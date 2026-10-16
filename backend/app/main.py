import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from ultralytics import YOLO

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
YOLO_MODEL = YOLO("yolo11n.pt")
app = FastAPI(
    title="Rulevia AI",
    description="Natural-language industrial safety intelligence.",
    version="1.0.0",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FRAMES_DIR = DATA_DIR / "extracted_frames"
REVIEW_FILE = DATA_DIR / "review_records.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/frames",
    StaticFiles(directory=str(FRAMES_DIR)),
    name="frames",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SafetyRuleRequest(BaseModel):
    instruction: str = Field(
        min_length=10,
        max_length=1000,
        description="Safety requirement written in plain English.",
    )


class SafetyRule(BaseModel):
    rule_name: str = Field(description="Short descriptive name for the rule.")
    description: str = Field(description="What the safety rule checks.")
    hazard_type: Literal[
        "ppe",
        "restricted_area",
        "fall",
        "unsafe_behavior",
        "other",
    ]
    detection_target: str = Field(
        description="Person, PPE item, object, or scene condition to detect."
    )
    condition: str = Field(
        description="A plain-language condition that would trigger review."
    )
    severity: Literal["low", "medium", "high", "critical"]
    required_evidence: list[str] = Field(
        description="Visual evidence that should be checked."
    )
    recommended_action: str = Field(
        description="Suggested action for a human reviewer."
    )
    confidence_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Initial proposed detector confidence threshold between 0 and 1. "
            "This is a configurable starting point, not a validated threshold."
        ),
    )
    limitations: list[str] = Field(
        description="Ambiguities and conditions that require human review."
    )


class VideoRuleEvaluationRequest(BaseModel):
    video_id: str
    rule: SafetyRule


class RuleValidationRequest(BaseModel):
    rule: SafetyRule


class ReviewDecisionRequest(BaseModel):
    video_id: str
    frame: str
    timestamp_seconds: float
    target_class: str
    confidence: float
    decision: str
    reviewer: str
    notes: str = ""


SUPPORTED_DETECTION_TERMS = {
    "hardhat": ("hardhat", "hard hat", "helmet"),
    "gloves": ("gloves", "glove"),
    "goggles": ("goggles", "goggle", "safety glasses"),
    "mask": ("mask", "respirator"),
    "safety vest": ("safety vest", "vest", "high-visibility vest"),
    "person": ("person", "worker", "people", "personnel"),
    "fall": ("fall", "fall-detected"),
    "ladder": ("ladder",),
    "safety cone": ("safety cone", "traffic cone"),
}


def validate_rule_draft(rule: SafetyRule) -> dict:
    findings = []
    combined_text = " ".join(
        [
            rule.rule_name,
            rule.description,
            rule.detection_target,
            rule.condition,
            *rule.required_evidence,
        ]
    ).lower()

    target = rule.detection_target.lower()
    detected_target = None

    # Find a target that the planned detector may support.
    for canonical_name, aliases in SUPPORTED_DETECTION_TERMS.items():
        if any(alias in target for alias in aliases):
            detected_target = canonical_name
            break

    # Some instructions describe the target in the condition rather
    # than in detection_target.
    if detected_target is None:
        for canonical_name, aliases in SUPPORTED_DETECTION_TERMS.items():
            if any(alias in combined_text for alias in aliases):
                detected_target = canonical_name
                break

    if detected_target:
        findings.append(
            {
                "check": "detection_target",
                "status": "passed",
                "message": (
                    f"Recognized a potentially supported target: {detected_target}."
                ),
            }
        )
    else:
        findings.append(
            {
                "check": "detection_target",
                "status": "needs_configuration",
                "message": (
                    "The target does not match a recognized detector label. "
                    "Map it to a supported class or configure another detector."
                ),
            }
        )

    spatial_terms = (
        "restricted area",
        "restricted zone",
        "zone",
        "boundary",
        "enters the area",
        "enters an area",
    )
    requires_spatial_setup = rule.hazard_type == "restricted_area" or any(
        term in combined_text for term in spatial_terms
    )

    if requires_spatial_setup:
        findings.append(
            {
                "check": "spatial_configuration",
                "status": "needs_configuration",
                "message": (
                    "This rule appears to depend on an area or boundary. "
                    "A reviewer must configure the region and its coordinates."
                ),
            }
        )
    else:
        findings.append(
            {
                "check": "spatial_configuration",
                "status": "passed",
                "message": "No explicit spatial boundary setup was identified.",
            }
        )

    if rule.condition.strip() and rule.required_evidence:
        findings.append(
            {
                "check": "rule_completeness",
                "status": "passed",
                "message": "A trigger condition and evidence requirements are present.",
            }
        )
    else:
        findings.append(
            {
                "check": "rule_completeness",
                "status": "failed",
                "message": "A condition and evidence requirements are required.",
            }
        )

    needs_configuration = any(
        finding["status"] == "needs_configuration" for finding in findings
    )
    has_failures = any(finding["status"] == "failed" for finding in findings)

    if has_failures:
        status = "invalid"
    elif needs_configuration:
        status = "needs_configuration"
    else:
        status = "ready_for_evaluation"

    return {
        "status": status,
        "rule_name": rule.rule_name,
        "findings": findings,
        "message": (
            "The draft needs changes before evaluation."
            if status == "invalid"
            else "Configure the outstanding requirements before evaluation."
            if status == "needs_configuration"
            else "Preliminary checks passed. The rule is ready for the next evaluation stage, not yet approved for production."
        ),
    }


@app.post("/rules/validate")
def validate_safety_rule(request: RuleValidationRequest):
    return validate_rule_draft(request.rule)


@app.get("/")
def home():
    return {
        "application": "Rulevia AI",
        "message": "Welcome to Rulevia AI",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Rulevia AI",
    }


@app.post("/rules/generate")
def generate_safety_rule(request: SafetyRuleRequest):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured in backend/.env.",
        )

    try:
        client = genai.Client(api_key=API_KEY)

        prompt = f"""
You are the safety-rule drafting assistant for Rulevia AI,
an industrial safety monitoring application.

Convert the user's instruction into a structured DRAFT rule.

Important requirements:
- Do not invent facts about the factory, camera, or footage.
- Do not claim that a person has actually violated a rule.
- Keep the rule focused on observable visual evidence.
- Explain ambiguity and detection limitations.
- Treat model detections as uncertain evidence that needs review.
- Do not invent a detector capability that may not exist.
- The confidence threshold is only a proposed starting value and
  must be validated against representative footage.
- The output will be reviewed and validated separately before execution.

User instruction:
{request.instruction}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SafetyRule,
            ),
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response. Please retry.",
            )

        rule = SafetyRule.model_validate_json(response.text)

        return {
            "status": "draft",
            "model": MODEL_NAME,
            "original_instruction": request.instruction,
            "rule": rule.model_dump(),
            "message": ("Draft generated. Validate it before enabling monitoring."),
        }

    except HTTPException:
        raise
    except ValidationError:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned a rule that failed schema validation.",
        )
    except Exception as exc:
        # Avoid returning API keys, request headers, or internal details.
        print(f"Rule generation failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not generate a rule. Check the backend logs, "
                "API key, model availability, and API quota."
            ),
        )




@app.post("/videos/upload-extract")
async def upload_and_extract_video(
    file: UploadFile = File(...),
):
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    max_size = 100 * 1024 * 1024  # 100 MB

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Upload an MP4, AVI, MOV, or MKV video.",
        )

    video_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{video_id}{extension}"
    frames_dir = FRAMES_DIR / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    size = 0
    capture = None

    try:
        # Save the upload with a size limit.
        with video_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 100 MB limit.",
                    )

                destination.write(chunk)

        # Open the video using OpenCV.
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise HTTPException(
                status_code=400,
                detail="The uploaded file could not be opened as a video.",
            )

        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="The video has invalid frame information.",
            )

        duration = total_frames / fps
        interval_seconds = 2
        frames = []
        timestamp = 0.0

        # Extract one frame every two seconds.
        while timestamp < duration:
            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp * 1000,
            )

            success, image = capture.read()

            if not success:
                break

            frame_number = len(frames) + 1
            filename = f"frame_{frame_number:05d}.jpg"
            frame_path = frames_dir / filename

            if not cv2.imwrite(str(frame_path), image):
                raise RuntimeError("Failed to save an extracted frame.")

            frames.append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": round(timestamp, 2),
                    "image_url": f"/frames/{video_id}/{filename}",
                }
            )

            timestamp += interval_seconds

        if not frames:
            raise HTTPException(
                status_code=400,
                detail="No frames could be extracted.",
            )

        return {
            "status": "success",
            "video_id": video_id,
            "filename": file.filename,
            "duration_seconds": round(duration, 2),
            "frame_interval_seconds": interval_seconds,
            "total_frames_extracted": len(frames),
            "frames": frames,
        }

    except Exception:
        # Remove incomplete uploads and extracted images on failure.
        video_path.unlink(missing_ok=True)

        for image_path in frames_dir.glob("*.jpg"):
            image_path.unlink(missing_ok=True)

        frames_dir.rmdir()
        raise

    finally:
        if capture is not None:
            capture.release()

        await file.close()


@app.post("/videos/{video_id}/detect")
def detect_video_objects(video_id: str):
    # Locate the extracted frames for this video
    video_frames_dir = FRAMES_DIR / video_id

    if not video_frames_dir.is_dir():
        raise HTTPException(
            status_code=404, detail="No extracted frames found for this video."
        )

    frame_paths = sorted(video_frames_dir.glob("frame_*.jpg"))

    if not frame_paths:
        raise HTTPException(
            status_code=404, detail="No frames available for detection."
        )

    detections = []

    for frame_path in frame_paths:
        # Frame 1 is timestamp 0, frame 2 is timestamp 2, etc.
        frame_number = int(frame_path.stem.replace("frame_", ""))
        timestamp_seconds = (frame_number - 1) * 2

        results = YOLO_MODEL.predict(source=str(frame_path), conf=0.25, verbose=False)

        frame_detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                coordinates = box.xyxy[0].tolist()

                frame_detections.append(
                    {
                        "class_name": YOLO_MODEL.names[class_id],
                        "confidence": round(confidence, 3),
                        "box": [round(value, 2) for value in coordinates],
                    }
                )

        detections.append(
            {
                "frame": frame_path.name,
                "timestamp_seconds": timestamp_seconds,
                "detections": frame_detections,
            }
        )

    return {
        "status": "success",
        "video_id": video_id,
        "total_frames_processed": len(frame_paths),
        "total_detections": sum(len(frame["detections"]) for frame in detections),
        "frames": detections,
    }


@app.post("/rules/evaluate-video")
def evaluate_video_rule(request: VideoRuleEvaluationRequest):
    # Only accept video IDs generated as UUIDs by the upload endpoint.
    try:
        safe_video_id = str(uuid.UUID(request.video_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID.")

    video_frames_dir = FRAMES_DIR / safe_video_id

    if not video_frames_dir.is_dir():
        raise HTTPException(
            status_code=404, detail="No extracted frames found for this video."
        )

    frame_paths = sorted(video_frames_dir.glob("frame_*.jpg"))

    if not frame_paths:
        raise HTTPException(
            status_code=404, detail="No frames available for evaluation."
        )

    rule = request.rule
    target_class = rule.detection_target.strip().lower().replace("_", " ")

    # Simple aliases for common object names.
    if target_class in {"people", "worker", "workers"}:
        target_class = "person"

    evaluated_frames = []
    matching_frames = []

    for frame_path in frame_paths:
        frame_number = int(frame_path.stem.replace("frame_", ""))
        timestamp_seconds = (frame_number - 1) * 2

        results = YOLO_MODEL.predict(
            source=str(frame_path), conf=rule.confidence_threshold, verbose=False
        )
        frame_height, frame_width = results[0].orig_shape
        frame_matches = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = YOLO_MODEL.names[class_id].lower()

                if class_name != target_class:
                    continue

                frame_matches.append(
                    {
                        "class_name": class_name,
                        "confidence": round(float(box.conf[0]), 3),
                        "box": [round(value, 2) for value in box.xyxy[0].tolist()],
                    }
                )

        frame_result = {
            "frame": frame_path.name,
            "timestamp_seconds": timestamp_seconds,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "target_observed": len(frame_matches) > 0,
            "matches": frame_matches,
        }

        evaluated_frames.append(frame_result)

        if frame_matches:
            matching_frames.append(frame_result)

    target_found = len(matching_frames) > 0

    condition = rule.condition.strip().lower()

    if condition == "present":
        condition_met = target_found

    elif condition == "not_observed":
        condition_met = not target_found

    elif any(
        phrase in condition
        for phrase in (
            "entering",
            "crossing",
            "remaining within",
            "restricted zone",
            "spatial boundaries",
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Restricted-area entry detection requires configured "
                "zone boundaries and spatial evaluation. "
                "Person presence alone cannot establish a zone violation."
            ),
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported condition: '{rule.condition}'.",
        )
    return {
        "status": "success",
        "video_id": safe_video_id,
        "rule_name": rule.rule_name,
        "target_class": target_class,
        "condition": rule.condition,
        "condition_met": condition_met,
        "evaluation_status": "needs_human_review",
        "human_approval_required": True,
        "total_frames_evaluated": len(evaluated_frames),
        "frames_with_target": len(matching_frames),
        "matching_frames": matching_frames,
        "frame_results": evaluated_frames,
        "limitations": [
            "Results depend on the detector and its confidence threshold.",
            "Not observing an object does not prove that it is absent.",
            "This evaluation does not establish helmet or PPE compliance.",
        ],
    }


@app.get("/reviews")
def get_reviews():
    if not REVIEW_FILE.exists():
        return {"status": "success", "total": 0, "reviews": []}

    try:
        with open(REVIEW_FILE, "r", encoding="utf-8") as file:
            reviews = json.load(file)

        if not isinstance(reviews, list):
            raise HTTPException(
                status_code=500,
                detail="Review records file must contain a JSON list.",
            )

        reviews = sorted(
            reviews,
            key=lambda review: review.get("reviewed_at", ""),
            reverse=True,
        )

        return {
            "status": "success",
            "total": len(reviews),
            "reviews": reviews,
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Review records file contains invalid JSON.",
        )


@app.post("/reviews")
def save_review(request: ReviewDecisionRequest):
    allowed_decisions = {
        "confirmed",
        "dismissed",
        "needs_further_review",
    }

    if request.decision not in allowed_decisions:
        raise HTTPException(
            status_code=400,
            detail=("Decision must be confirmed, dismissed, or needs_further_review."),
        )

    try:
        safe_video_id = str(uuid.UUID(request.video_id))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid video ID.",
        )

    if not (FRAMES_DIR / safe_video_id).is_dir():
        raise HTTPException(
            status_code=404,
            detail="Video frames not found.",
        )

    if not request.frame.startswith("frame_") or not request.frame.endswith(".jpg"):
        raise HTTPException(
            status_code=400,
            detail="Invalid frame filename.",
        )

    frame_path = FRAMES_DIR / safe_video_id / request.frame

    if not frame_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evidence frame not found.",
        )

    if not 0 <= request.confidence <= 1:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1.",
        )

    if request.timestamp_seconds < 0:
        raise HTTPException(
            status_code=400,
            detail="Timestamp cannot be negative.",
        )

    records = []

    if REVIEW_FILE.exists():
        try:
            records = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="Review record file contains invalid JSON.",
            )

    review = {
        "review_id": str(uuid.uuid4()),
        "video_id": safe_video_id,
        "frame": request.frame,
        "timestamp_seconds": request.timestamp_seconds,
        "target_class": request.target_class,
        "confidence": request.confidence,
        "decision": request.decision,
        "reviewer": request.reviewer,
        "notes": request.notes,
        "reviewed_at": datetime.now(UTC).isoformat(),
    }

    records.append(review)

    REVIEW_FILE.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    return {
        "status": "success",
        "message": "Review decision saved.",
        "review": review,
    }
