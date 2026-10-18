import json
import math
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from google.genai.errors import APIError as GeminiAPIError
from pydantic import BaseModel, Field, ValidationError, field_validator
from ultralytics import YOLO

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
YOLO_MODEL = YOLO("yolo11n.pt")
CORS_ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
}


@dataclass(frozen=True)
class DetectedObject:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectorFrame:
    frame_width: int
    frame_height: int
    detections: list[DetectedObject]


class VisualCapability(BaseModel):
    capability_id: str
    detector_id: str
    capability_type: Literal[
        "object_detection",
        "attribute_classification",
        "tracking",
        "spatial_analysis",
        "behaviour_recognition",
    ]
    description: str
    supported_object_classes: list[str] = Field(default_factory=list)
    supported_attributes: list[str] = Field(default_factory=list)
    supported_states: list[str] = Field(default_factory=list)
    supported_behaviours: list[str] = Field(default_factory=list)
    supported_relationships: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    configuration_schema: dict[str, Any] = Field(default_factory=dict)
    known_limitations: list[str] = Field(default_factory=list)
    validation_status: Literal[
        "unvalidated",
        "validated",
        "production_ready",
    ] = "unvalidated"
    validation_notes: list[str] = Field(default_factory=list)


class RuleConditionCapability(BaseModel):
    condition: str
    description: str
    requires_zone: bool = False
    required_capabilities: list[str] = Field(default_factory=list)
    required_capability_types: list[
        Literal[
            "object_detection",
            "attribute_classification",
            "tracking",
            "spatial_analysis",
            "behaviour_recognition",
        ]
    ] = Field(default_factory=list)
    required_parameters: list[str] = Field(default_factory=list)


CORE_SPATIAL_CAPABILITY = VisualCapability(
    capability_id="rulevia.spatial_analysis",
    detector_id="rulevia-core",
    capability_type="spatial_analysis",
    description="Evaluates detector geometry against a configured normalized polygon.",
    required_inputs=["object_detection_result", "normalized_zone_polygon"],
    known_limitations=[
        "Spatial results inherit errors from upstream detections and frame sampling.",
        "Spatial evaluation alone does not identify object attributes, states, or behaviours.",
    ],
    validation_status="unvalidated",
)


class DetectorAdapter(Protocol):
    detector_id: str
    display_name: str

    def available_classes(self) -> dict[int, str]: ...

    def capabilities(self) -> list[VisualCapability]: ...

    def supported_behaviors(self) -> set[str]: ...

    def predict(
        self,
        source: str | np.ndarray,
        confidence: float,
    ) -> list[DetectorFrame]: ...


class RuleEvaluationPlugin(Protocol):
    plugin_id: str

    def capabilities(self) -> list[VisualCapability]: ...

    def conditions(self) -> list[RuleConditionCapability]: ...

    def evaluate_video(
        self,
        request: "VideoRuleEvaluationRequest",
        frame_paths: list[Path],
    ) -> dict[str, Any]: ...


class YoloDetectorAdapter:
    detector_id = "yolo"
    display_name = "YOLO object detector"

    def __init__(self, model_provider: Callable[[], YOLO]):
        self._model_provider = model_provider

    def available_classes(self) -> dict[int, str]:
        names = self._model_provider().names
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        return {class_id: str(name) for class_id, name in enumerate(names)}

    def supported_behaviors(self) -> set[str]:
        return {"object_detection"}

    def capabilities(self) -> list[VisualCapability]:
        return [
            VisualCapability(
                capability_id=f"{self.detector_id}.object_detection",
                detector_id=self.detector_id,
                capability_type="object_detection",
                description="Detects classes listed in the loaded model metadata.",
                supported_object_classes=sorted(
                    self.available_classes().values(),
                    key=str.casefold,
                ),
                required_inputs=["image_frame"],
                configuration_schema={
                    "confidence_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    }
                },
                known_limitations=[
                    "Class availability does not establish attributes, states, tracking, relationships, or behaviour.",
                    "Detector accuracy has not been independently validated by this application.",
                ],
                validation_status="unvalidated",
            )
        ]

    def predict(
        self,
        source: str | np.ndarray,
        confidence: float,
    ) -> list[DetectorFrame]:
        model = self._model_provider()
        results = model.predict(source=source, conf=confidence, verbose=False)
        frames = []
        for result in results:
            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append(
                    DetectedObject(
                        class_id=class_id,
                        class_name=str(model.names[class_id]),
                        confidence=float(box.conf[0]),
                        box=tuple(float(value) for value in box.xyxy[0].tolist()),
                    )
                )
            frame_height, frame_width = result.orig_shape
            frames.append(
                DetectorFrame(
                    frame_width=int(frame_width),
                    frame_height=int(frame_height),
                    detections=detections,
                )
            )
        return frames


class DetectorCapabilityRegistry:
    def __init__(self):
        self._detectors: dict[str, DetectorAdapter] = {}
        self._evaluators: dict[str, RuleEvaluationPlugin] = {}

    def register(self, detector: DetectorAdapter) -> None:
        if detector.detector_id in self._detectors:
            raise ValueError(f"Detector '{detector.detector_id}' is already registered.")
        self._detectors[detector.detector_id] = detector

    def list_detectors(self) -> list[DetectorAdapter]:
        return list(self._detectors.values())

    def register_evaluator(self, evaluator: RuleEvaluationPlugin) -> None:
        if evaluator.plugin_id in self._evaluators:
            raise ValueError(
                f"Rule evaluator '{evaluator.plugin_id}' is already registered."
            )
        for condition in evaluator.conditions():
            if condition.condition in CONDITION_REGISTRY or any(
                condition.condition == registered.condition
                for plugin in self._evaluators.values()
                for registered in plugin.conditions()
            ):
                raise ValueError(
                    f"Condition '{condition.condition}' is already registered."
                )
        self._evaluators[evaluator.plugin_id] = evaluator

    def list_evaluators(self) -> list[RuleEvaluationPlugin]:
        return list(self._evaluators.values())

    def resolve_evaluator(
        self,
        condition: str,
    ) -> tuple[RuleEvaluationPlugin, RuleConditionCapability] | None:
        for evaluator in self.list_evaluators():
            for capability in evaluator.conditions():
                if capability.condition == condition:
                    return evaluator, capability
        return None

    def capability_catalog(self) -> list[VisualCapability]:
        return [
            capability
            for detector in self.list_detectors()
            for capability in detector.capabilities()
        ] + [
            capability
            for evaluator in self.list_evaluators()
            for capability in evaluator.capabilities()
        ] + [CORE_SPATIAL_CAPABILITY]

    def has_capability(self, capability_id: str) -> bool:
        return any(
            capability.capability_id == capability_id
            for capability in self.capability_catalog()
        )

    def supports_behavior(self, behavior: str, detector_id: str | None = None) -> bool:
        if detector_id is not None:
            detector = self._detectors.get(detector_id)
            return detector is not None and behavior in detector.supported_behaviors()
        return any(
            behavior in detector.supported_behaviors()
            for detector in self.list_detectors()
        )

    def resolve_target(
        self,
        target: str,
        detector_id: str | None = None,
    ) -> tuple[DetectorAdapter, int, str] | None:
        normalized_target = normalize_detection_target(target)
        matching = []
        if detector_id is not None and detector_id not in self._detectors:
            return None
        detectors = (
            [self._detectors[detector_id]]
            if detector_id in self._detectors
            else self.list_detectors()
        )
        for detector in detectors:
            for class_id, class_name in detector.available_classes().items():
                if normalize_detection_target(class_name) == normalized_target:
                    matching.append((detector, class_id, class_name))
        if len(matching) == 1:
            return matching[0]
        return None

    def available_targets(self) -> list[dict]:
        targets = []
        for detector in self.list_detectors():
            targets.extend(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "detector_id": detector.detector_id,
                    "detector_name": detector.display_name,
                    "detector_behaviors": sorted(detector.supported_behaviors()),
                }
                for class_id, class_name in detector.available_classes().items()
            )
        return sorted(
            targets,
            key=lambda target: (
                target["class_name"].casefold(),
                target["detector_id"],
            ),
        )


LEGACY_TARGET_ALIASES = {
    "people": "person",
    "worker": "person",
    "workers": "person",
    "personnel": "person",
}


def normalize_detection_target(target: str) -> str:
    normalized = " ".join(target.strip().lower().replace("_", " ").split())
    return LEGACY_TARGET_ALIASES.get(normalized, normalized)


DETECTOR_REGISTRY = DetectorCapabilityRegistry()
DETECTOR_REGISTRY.register(YoloDetectorAdapter(lambda: YOLO_MODEL))

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
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *sorted(CORS_ALLOWED_ORIGINS),
    ],
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
    hazard_type: str = Field(
        description="Generic hazard category; use a concise machine-readable value."
    )
    detection_target: str = Field(
        description="Requested detector class name, selected from capabilities when available."
    )
    detector_id: str | None = Field(
        default=None,
        description="Optional detector selected from the capability registry.",
    )
    condition: str = Field(
        description=(
            "An evaluator-supported key such as 'person_in_zone', "
            "'person_enters_zone', 'person_overlaps_zone', 'object_in_zone', "
            "'object_enters_zone', or 'object_overlaps_zone', or a "
            "plain-language condition."
        )
    )
    condition_parameters: dict[str, float | str | bool] = Field(
        default_factory=dict,
        description="Parameters required by the selected evaluation condition.",
    )
    required_capability_types: list[
        Literal[
            "object_detection",
            "attribute_classification",
            "tracking",
            "spatial_analysis",
            "behaviour_recognition",
        ]
    ] = Field(
        default_factory=list,
        description="Capability types explicitly required by the requested rule.",
    )
    required_capabilities: list[str] = Field(
        default_factory=list,
        description="Capability IDs required to execute this rule.",
    )
    zone_overlap_threshold: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "For person_overlaps_zone, minimum intersection area divided by "
            "person bounding-box area, strictly greater than 0 and at most 1."
        ),
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

    @field_validator("zone_overlap_threshold")
    @classmethod
    def reject_non_finite_overlap_threshold(
        cls,
        threshold: float | None,
    ) -> float | None:
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError("zone_overlap_threshold must be a finite number.")
        return threshold

    @field_validator("condition_parameters")
    @classmethod
    def reject_non_finite_condition_parameters(
        cls,
        parameters: dict[str, float | str | bool] | list,
    ) -> dict[str, float | str | bool] | list:
        values = (
            parameters.values()
            if isinstance(parameters, dict)
            else (parameter.value for parameter in parameters)
        )
        if any(isinstance(value, float) and not math.isfinite(value) for value in values):
            raise ValueError("Numeric condition parameters must be finite.")
        return parameters


class GeneratedRuleParameter(BaseModel):
    name: str
    value: str = Field(
        description="Parameter value as text; numeric and boolean strings are converted when parsing."
    )


class GeneratedSafetyRule(SafetyRule):
    condition_parameters: list[GeneratedRuleParameter] = Field(
        default_factory=list,
        description="Condition parameters as named entries; do not use a free-form object.",
    )
    zone_overlap_threshold: float | None = None


class ZonePoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


def validate_zone_polygon(zone: list[ZonePoint]) -> list[ZonePoint]:
    if len(zone) < 3:
        raise ValueError("A restricted zone must contain at least three points.")

    coordinates = [(point.x, point.y) for point in zone]

    if len(set(coordinates)) < 3:
        raise ValueError("A restricted zone must contain three distinct points.")

    area = sum(
        coordinates[index][0] * coordinates[(index + 1) % len(coordinates)][1]
        - coordinates[(index + 1) % len(coordinates)][0] * coordinates[index][1]
        for index in range(len(coordinates))
    )
    if abs(area) <= 1e-9:
        raise ValueError("A restricted zone must have a non-zero area.")

    return zone


def point_in_polygon(point: tuple[float, float], zone: list[ZonePoint]) -> bool:
    x, y = point
    inside = False

    for index, current in enumerate(zone):
        previous = zone[index - 1]
        x1, y1 = previous.x, previous.y
        x2, y2 = current.x, current.y

        cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross_product) <= 1e-9
            and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9
            and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9
        ):
            return True

        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside

    return inside


def _polygon_signed_area(polygon: list[tuple[float, float]]) -> float:
    return sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    ) / 2


def _cross(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    epsilon = 1e-12
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)

    if (
        ((ab_c > epsilon and ab_d < -epsilon) or (ab_c < -epsilon and ab_d > epsilon))
        and (
            (cd_a > epsilon and cd_b < -epsilon)
            or (cd_a < -epsilon and cd_b > epsilon)
        )
    ):
        return True

    def on_segment(
        start: tuple[float, float],
        end: tuple[float, float],
        point: tuple[float, float],
    ) -> bool:
        return (
            abs(_cross(start, end, point)) <= epsilon
            and min(start[0], end[0]) - epsilon
            <= point[0]
            <= max(start[0], end[0]) + epsilon
            and min(start[1], end[1]) - epsilon
            <= point[1]
            <= max(start[1], end[1]) + epsilon
        )

    return (
        on_segment(a, b, c)
        or on_segment(a, b, d)
        or on_segment(c, d, a)
        or on_segment(c, d, b)
    )


def _simple_polygon_coordinates(zone: list[ZonePoint]) -> list[tuple[float, float]]:
    coordinates = [(point.x, point.y) for point in zone]
    if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
        coordinates.pop()

    compacted = []
    for point in coordinates:
        if not compacted or point != compacted[-1]:
            compacted.append(point)

    if len(compacted) < 3:
        raise ValueError("A restricted zone must contain three distinct vertices.")

    edge_count = len(compacted)
    for first_index in range(edge_count):
        first_start = compacted[first_index]
        first_end = compacted[(first_index + 1) % edge_count]
        for second_index in range(first_index + 1, edge_count):
            if (
                second_index == first_index
                or second_index == (first_index + 1) % edge_count
                or first_index == (second_index + 1) % edge_count
            ):
                continue
            second_start = compacted[second_index]
            second_end = compacted[(second_index + 1) % edge_count]
            if _segments_intersect(
                first_start,
                first_end,
                second_start,
                second_end,
            ):
                raise ValueError(
                    "Bounding-box overlap requires a simple, non-self-intersecting "
                    "restricted-zone polygon."
                )

    changed = True
    while changed and len(compacted) > 3:
        changed = False
        for index, current in enumerate(compacted):
            previous = compacted[index - 1]
            following = compacted[(index + 1) % len(compacted)]
            if (
                abs(_cross(previous, current, following)) <= 1e-12
                and min(previous[0], following[0]) - 1e-12
                <= current[0]
                <= max(previous[0], following[0]) + 1e-12
                and min(previous[1], following[1]) - 1e-12
                <= current[1]
                <= max(previous[1], following[1]) + 1e-12
            ):
                del compacted[index]
                changed = True
                break

    return compacted


def _triangulate_simple_polygon(
    polygon: list[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    vertices = list(range(len(polygon)))
    orientation = 1 if _polygon_signed_area(polygon) > 0 else -1
    triangles = []
    epsilon = 1e-12

    while len(vertices) > 3:
        ear_found = False
        for position, current_index in enumerate(vertices):
            previous_index = vertices[position - 1]
            next_index = vertices[(position + 1) % len(vertices)]
            previous = polygon[previous_index]
            current = polygon[current_index]
            following = polygon[next_index]

            if orientation * _cross(previous, current, following) <= epsilon:
                continue

            contains_vertex = False
            for other_index in vertices:
                if other_index in {previous_index, current_index, next_index}:
                    continue
                p = polygon[other_index]
                if (
                    orientation * _cross(previous, current, p) >= -epsilon
                    and orientation * _cross(current, following, p) >= -epsilon
                    and orientation * _cross(following, previous, p) >= -epsilon
                ):
                    contains_vertex = True
                    break

            if contains_vertex:
                continue

            triangles.append([previous, current, following])
            del vertices[position]
            ear_found = True
            break

        if not ear_found:
            raise ValueError(
                "Restricted-zone polygon could not be triangulated for overlap "
                "evaluation."
            )

    triangles.append([polygon[index] for index in vertices])
    return triangles


def validate_overlap_zone(zone: list[ZonePoint]) -> list[list[tuple[float, float]]]:
    return _triangulate_simple_polygon(_simple_polygon_coordinates(zone))


def _clip_polygon_to_box(
    polygon: list[tuple[float, float]],
    box: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    left, top, right, bottom = box

    def clip_edge(
        points: list[tuple[float, float]],
        is_inside,
        intersect,
    ) -> list[tuple[float, float]]:
        if not points:
            return []

        output = []
        previous = points[-1]
        previous_inside = is_inside(previous)
        for current in points:
            current_inside = is_inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersect(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous = current
            previous_inside = current_inside
        return output

    def vertical_intersection(x: float, start, end):
        amount = (x - start[0]) / (end[0] - start[0])
        return (x, start[1] + amount * (end[1] - start[1]))

    def horizontal_intersection(y: float, start, end):
        amount = (y - start[1]) / (end[1] - start[1])
        return (start[0] + amount * (end[0] - start[0]), y)

    clipped = polygon
    clipped = clip_edge(
        clipped,
        lambda point: point[0] >= left,
        lambda start, end: vertical_intersection(left, start, end),
    )
    clipped = clip_edge(
        clipped,
        lambda point: point[0] <= right,
        lambda start, end: vertical_intersection(right, start, end),
    )
    clipped = clip_edge(
        clipped,
        lambda point: point[1] >= top,
        lambda start, end: horizontal_intersection(top, start, end),
    )
    return clip_edge(
        clipped,
        lambda point: point[1] <= bottom,
        lambda start, end: horizontal_intersection(bottom, start, end),
    )


def bounding_box_polygon_overlap_ratio(
    box: tuple[float, float, float, float],
    zone: list[ZonePoint],
) -> float:
    """Return exact intersection area / box area for a simple zone polygon."""
    polygon = _simple_polygon_coordinates(zone)
    triangles = _triangulate_simple_polygon(polygon)
    return _bounding_box_triangle_overlap_ratio(box, triangles)


def evaluate_spatial_detection(
    detected_object: DetectedObject,
    frame_width: int,
    frame_height: int,
    zone: list[ZonePoint],
    condition_kind: str,
    overlap_threshold: float | None = None,
    zone_triangles: list[list[tuple[float, float]]] | None = None,
) -> tuple[dict[str, Any], bool]:
    x1, y1, x2, y2 = detected_object.box
    anchor_point = (
        min(max((x1 + x2) / (2 * frame_width), 0.0), 1.0),
        min(max(y2 / frame_height, 0.0), 1.0),
    )
    inside_zone = point_in_polygon(anchor_point, zone)
    diagnostic: dict[str, Any] = {
        "anchor_point": {
            "x": round(anchor_point[0], 5),
            "y": round(anchor_point[1], 5),
        },
        "inside_zone": inside_zone,
    }
    if condition_kind != "overlaps_zone":
        return diagnostic, inside_zone

    normalized_box = (
        x1 / frame_width,
        y1 / frame_height,
        x2 / frame_width,
        y2 / frame_height,
    )
    triangles = (
        zone_triangles
        if zone_triangles is not None
        else validate_overlap_zone(zone)
    )
    overlap_ratio = _bounding_box_triangle_overlap_ratio(normalized_box, triangles)
    matches = (
        overlap_threshold is not None
        and overlap_ratio + 1e-12 >= overlap_threshold
    )
    diagnostic.update(
        {
            "zone_overlap_ratio": round(overlap_ratio, 6),
            "zone_overlap_threshold": overlap_threshold,
            "overlaps_zone": matches,
        }
    )
    return diagnostic, matches


def _bounding_box_triangle_overlap_ratio(
    box: tuple[float, float, float, float],
    triangles: list[list[tuple[float, float]]],
) -> float:
    left, top, right, bottom = box
    box_area = (right - left) * (bottom - top)
    if box_area <= 0:
        return 0.0

    intersection_area = 0.0
    for triangle in triangles:
        clipped = _clip_polygon_to_box(triangle, box)
        if len(clipped) >= 3:
            intersection_area += abs(_polygon_signed_area(clipped))

    return min(max(intersection_area / box_area, 0.0), 1.0)


class VideoRuleEvaluationRequest(BaseModel):
    video_id: str
    rule: SafetyRule
    zone: list[ZonePoint] | None = None

    @field_validator("zone")
    @classmethod
    def validate_zone(cls, zone: list[ZonePoint] | None) -> list[ZonePoint] | None:
        return validate_zone_polygon(zone) if zone is not None else None


class RuleValidationRequest(BaseModel):
    rule: SafetyRule
    zone: list[ZonePoint] | None = None

    @field_validator("zone")
    @classmethod
    def validate_zone(cls, zone: list[ZonePoint] | None) -> list[ZonePoint] | None:
        return validate_zone_polygon(zone) if zone is not None else None


class ReviewDecisionRequest(BaseModel):
    video_id: str
    frame: str
    timestamp_seconds: float
    target_class: str
    confidence: float
    decision: str
    reviewer: str
    notes: str = ""


SUPPORTED_SPATIAL_CONDITION = "person_in_zone"
SUPPORTED_ZONE_ENTRY_CONDITION = "person_enters_zone"
SUPPORTED_ZONE_OVERLAP_CONDITION = "person_overlaps_zone"
SUPPORTED_OBJECT_IN_ZONE_CONDITION = "object_in_zone"
SUPPORTED_OBJECT_ENTRY_CONDITION = "object_enters_zone"
SUPPORTED_OBJECT_OVERLAP_CONDITION = "object_overlaps_zone"

CONDITION_REGISTRY = {
    "present": {
        "kind": "presence",
        "spatial": False,
        "required_capability_types": ["object_detection"],
    },
    "not_observed": {
        "kind": "absence",
        "spatial": False,
        "required_capability_types": ["object_detection"],
    },
    SUPPORTED_SPATIAL_CONDITION: {
        "kind": "point_in_zone",
        "spatial": True,
        "target_class": "person",
        "required_capability_types": ["object_detection", "spatial_analysis"],
    },
    SUPPORTED_ZONE_ENTRY_CONDITION: {
        "kind": "enters_zone",
        "spatial": True,
        "target_class": "person",
        "required_capability_types": ["object_detection", "spatial_analysis"],
    },
    SUPPORTED_ZONE_OVERLAP_CONDITION: {
        "kind": "overlaps_zone",
        "spatial": True,
        "target_class": "person",
        "parameter": "overlap_threshold",
        "required_capability_types": ["object_detection", "spatial_analysis"],
    },
    SUPPORTED_OBJECT_IN_ZONE_CONDITION: {
        "kind": "point_in_zone",
        "spatial": True,
        "required_capability_types": ["object_detection", "spatial_analysis"],
    },
    SUPPORTED_OBJECT_ENTRY_CONDITION: {
        "kind": "enters_zone",
        "spatial": True,
        "required_capability_types": ["object_detection", "spatial_analysis"],
    },
    SUPPORTED_OBJECT_OVERLAP_CONDITION: {
        "kind": "overlaps_zone",
        "spatial": True,
        "parameter": "overlap_threshold",
        "required_capability_types": ["object_detection", "spatial_analysis"],
    },
}


def rule_condition_capability(
    condition: str,
) -> tuple[dict[str, Any] | RuleConditionCapability | None, RuleEvaluationPlugin | None]:
    builtin = CONDITION_REGISTRY.get(condition)
    if builtin is not None:
        return builtin, None
    resolved = DETECTOR_REGISTRY.resolve_evaluator(condition)
    if resolved is None:
        return None, None
    evaluator, capability = resolved
    return capability, evaluator


def capability_requirements(
    rule: SafetyRule,
    condition_capability: dict[str, Any] | RuleConditionCapability | None,
) -> tuple[set[str], set[str]]:
    required_types = set(rule.required_capability_types)
    required_ids = set(rule.required_capabilities)
    if isinstance(condition_capability, dict):
        required_types.update(
            condition_capability.get("required_capability_types", [])
        )
        required_ids.update(condition_capability.get("required_capabilities", []))
    elif condition_capability is not None:
        required_types.update(condition_capability.required_capability_types)
        required_ids.update(condition_capability.required_capabilities)
    return required_types, required_ids


def capabilities_executable_for_rule(
    target: str,
    detector_id: str | None,
    resolved_target: tuple[DetectorAdapter, int, str] | None,
    condition_capability: dict[str, Any] | RuleConditionCapability | None,
    evaluator_plugin: RuleEvaluationPlugin | None,
) -> list[VisualCapability]:
    normalized_target = normalize_detection_target(target)
    if evaluator_plugin is not None:
        if not isinstance(condition_capability, RuleConditionCapability):
            return []
        declared_types = set(condition_capability.required_capability_types)
        declared_ids = set(condition_capability.required_capabilities)
        return [
            capability
            for capability in evaluator_plugin.capabilities()
            if (detector_id is None or capability.detector_id == detector_id)
            and (
                capability.capability_id in declared_ids
                or capability.capability_type in declared_types
            )
            and any(
                normalize_detection_target(class_name) == normalized_target
                for class_name in capability.supported_object_classes
            )
        ]

    if resolved_target is None:
        return []

    detector = resolved_target[0]
    capabilities = [
        capability
        for capability in detector.capabilities()
        if capability.detector_id == detector.detector_id
        and (detector_id is None or capability.detector_id == detector_id)
        and any(
            normalize_detection_target(class_name) == normalized_target
            for class_name in capability.supported_object_classes
        )
    ]
    if (
        isinstance(condition_capability, dict)
        and condition_capability.get("spatial")
    ):
        capabilities.append(CORE_SPATIAL_CAPABILITY)
    return capabilities


def capability_implementation_guidance(capability_type: str) -> str:
    guidance = {
        "object_detection": "register a detector that detects the requested target",
        "attribute_classification": (
            "register an attribute/state classifier and evaluator for the requested target"
        ),
        "tracking": "register a tracker that supplies stable identities for the requested target",
        "spatial_analysis": (
            "register spatial evaluation logic and connect it to the selected condition"
        ),
        "behaviour_recognition": (
            "register a behaviour recognizer and evaluator for the requested target"
        ),
    }
    return guidance.get(
        capability_type,
        "register an implementation for this capability and connect it to the selected condition",
    )


def capability_requirement_issues(
    required_types: set[str],
    required_ids: set[str],
    executable_capabilities: list[VisualCapability],
    target: str,
    condition: str,
) -> list[dict[str, str]]:
    catalog = DETECTOR_REGISTRY.capability_catalog()
    global_types = {capability.capability_type for capability in catalog}
    global_ids = {capability.capability_id for capability in catalog}
    executable_types = {
        capability.capability_type for capability in executable_capabilities
    }
    executable_ids = {
        capability.capability_id for capability in executable_capabilities
    }
    issues = []

    for capability_type in sorted(required_types - executable_types):
        globally_registered = capability_type in global_types
        issue = f"type:{capability_type}"
        if globally_registered:
            reason = (
                f"Capability type '{capability_type}' exists in the global registry "
                f"but is not executable for target '{target}' with condition "
                f"'{condition}'."
            )
        else:
            reason = f"Capability type '{capability_type}' is not registered."
        issues.append(
            {
                "requirement": issue,
                "reason": reason,
                "needed": capability_implementation_guidance(capability_type),
            }
        )

    for capability_id in sorted(required_ids - executable_ids):
        globally_registered = capability_id in global_ids
        issue = f"id:{capability_id}"
        if globally_registered:
            reason = (
                f"Capability '{capability_id}' exists in the global registry but "
                f"is not executable for target '{target}' with condition "
                f"'{condition}'."
            )
        else:
            reason = f"Capability '{capability_id}' is not registered."
        issues.append(
            {
                "requirement": issue,
                "reason": reason,
                "needed": (
                    "Register this capability for the requested target and ensure "
                    "the selected condition evaluator executes it."
                ),
            }
        )

    return issues


def validate_rule_draft(rule: SafetyRule, zone_configured: bool = False) -> dict:
    findings = []
    condition = rule.condition.strip().lower()
    condition_capability, evaluator_plugin = rule_condition_capability(condition)
    required_types, required_ids = capability_requirements(
        rule,
        condition_capability,
    )
    resolved_target = DETECTOR_REGISTRY.resolve_target(
        rule.detection_target,
        rule.detector_id,
    )
    executable_capabilities = capabilities_executable_for_rule(
        rule.detection_target,
        rule.detector_id,
        resolved_target,
        condition_capability,
        evaluator_plugin,
    )
    capability_issues = capability_requirement_issues(
        required_types,
        required_ids,
        executable_capabilities,
        rule.detection_target,
        condition,
    )
    findings.append(
        {
            "check": "required_capabilities",
            "status": "needs_configuration" if capability_issues else "passed",
            "message": (
                "Required capabilities are not executable for this target and "
                "condition: "
                + " ".join(
                    f"{issue['requirement']}: {issue['reason']} "
                    f"Needed: {issue['needed']}"
                    for issue in capability_issues
                )
                if capability_issues
                else (
                    "All required capability types and IDs are executable for the "
                    "selected target and condition."
                )
            ),
            "missing_capabilities": [
                issue["requirement"] for issue in capability_issues
            ],
            "capability_issues": capability_issues,
        }
    )

    plugin_supports_target = any(
        any(
            normalize_detection_target(class_name)
            == normalize_detection_target(rule.detection_target)
            for class_name in capability.supported_object_classes
        )
        and (
            rule.detector_id is None
            or capability.detector_id == rule.detector_id
        )
        for capability in (
            evaluator_plugin.capabilities() if evaluator_plugin else []
        )
    )
    if resolved_target or plugin_supports_target:
        if resolved_target:
            detector, class_id, class_name = resolved_target
            detector_name = detector.display_name
            resolved_detector_id = detector.detector_id
        else:
            target_capability = next(
                capability
                for capability in executable_capabilities
                if any(
                    normalize_detection_target(class_name)
                    == normalize_detection_target(rule.detection_target)
                    for class_name in capability.supported_object_classes
                )
            )
            class_id = None
            class_name = rule.detection_target
            detector_name = target_capability.detector_id
            resolved_detector_id = target_capability.detector_id
        findings.append(
            {
                "check": "detection_target",
                "status": "passed",
                "message": f"Target '{class_name}' is available from {detector_name}.",
                "requested_target": rule.detection_target,
                "executable_target": class_name,
                "detector_id": resolved_detector_id,
                "class_id": class_id,
            }
        )
    else:
        findings.append(
            {
                "check": "detection_target",
                "status": "needs_configuration",
                "message": (
                    f"Requested target '{rule.detection_target}' is not currently "
                    "available from a registered detector. Add/register a "
                    "suitable detector; no alternate target will be substituted."
                ),
                "requested_target": rule.detection_target,
                "executable_target": None,
            }
        )

    if condition_capability is None:
        findings.append(
            {
                "check": "condition",
                "status": "failed",
                "message": f"Unsupported evaluation condition: '{rule.condition}'.",
            }
        )
    else:
        findings.append(
            {
                "check": "condition",
                "status": "passed",
                "message": f"Condition '{condition}' is supported.",
            }
        )
        required_class = (
            condition_capability.get("target_class")
            if isinstance(condition_capability, dict)
            else None
        )
        if required_class and normalize_detection_target(rule.detection_target) != required_class:
            findings.append(
                {
                    "check": "condition_target",
                    "status": "failed",
                    "message": f"'{condition}' preserves its legacy semantics and requires target '{required_class}'.",
                }
            )
        condition_parameter = (
            condition_capability.get("parameter")
            if isinstance(condition_capability, dict)
            else None
        )
        supported_parameters = (
            {condition_parameter}
            if condition_parameter
            else set(
                condition_capability.required_parameters
                if isinstance(condition_capability, RuleConditionCapability)
                else []
            )
        )
        unknown_parameters = set(rule.condition_parameters) - supported_parameters
        if unknown_parameters:
            findings.append(
                {
                    "check": "condition_parameters",
                    "status": "failed",
                    "message": (
                        "Unsupported condition parameter(s): "
                        + ", ".join(sorted(unknown_parameters))
                    ),
                }
            )
        missing_parameters = supported_parameters - set(rule.condition_parameters)
        if missing_parameters and isinstance(
            condition_capability,
            RuleConditionCapability,
        ):
            findings.append(
                {
                    "check": "condition_parameters",
                    "status": "needs_configuration",
                    "message": (
                        "Set the required condition parameter(s): "
                        + ", ".join(sorted(missing_parameters))
                    ),
                }
            )

    requires_spatial_setup = rule_requires_zone(rule)

    if (
        isinstance(condition_capability, dict)
        and condition_capability.get("parameter") == "overlap_threshold"
    ):
        overlap_threshold = rule.condition_parameters.get(
            "overlap_threshold",
            rule.zone_overlap_threshold,
        )
        if overlap_threshold is None:
            findings.append(
                {
                    "check": (
                        "zone_overlap_threshold"
                        if condition == SUPPORTED_ZONE_OVERLAP_CONDITION
                        else "overlap_threshold"
                    ),
                    "status": (
                        "failed"
                        if condition == SUPPORTED_ZONE_OVERLAP_CONDITION
                        else "needs_configuration"
                    ),
                    "message": (
                        "Set overlap_threshold (or legacy zone_overlap_threshold) to a value greater than 0 "
                        "and at most 1 for bounding-box overlap evaluation."
                    ),
                }
            )
        elif (
            isinstance(overlap_threshold, bool)
            or not isinstance(overlap_threshold, (int, float))
            or not math.isfinite(overlap_threshold)
            or not 0 < overlap_threshold <= 1
        ):
            findings.append(
                {
                    "check": "overlap_threshold",
                    "status": "failed",
                    "message": "overlap_threshold must be numeric, greater than 0, and at most 1.",
                }
            )
        else:
            findings.append(
                {
                    "check": "zone_overlap_threshold",
                    "status": "passed",
                    "message": (
                        "The threshold is the minimum fraction of the target bounding-box area that must overlap the zone."
                    ),
                }
            )
    elif rule.zone_overlap_threshold is not None or "overlap_threshold" in rule.condition_parameters:
        findings.append(
            {
                "check": "zone_overlap_threshold",
                "status": "failed",
                "message": (
                    "zone_overlap_threshold is only supported with "
                    f"'{SUPPORTED_ZONE_OVERLAP_CONDITION}'."
                ),
            }
        )

    if requires_spatial_setup and not zone_configured:
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
    elif requires_spatial_setup:
        findings.append(
            {
                "check": "spatial_configuration",
                "status": "passed",
                "message": "A valid restricted-zone polygon is configured.",
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


def rule_requires_zone(rule: SafetyRule) -> bool:
    condition_capability, _ = rule_condition_capability(
        rule.condition.strip().lower()
    )
    if isinstance(condition_capability, dict):
        return bool(condition_capability.get("spatial"))
    if isinstance(condition_capability, RuleConditionCapability):
        return condition_capability.requires_zone
    combined_text = " ".join(
        [
            rule.rule_name,
            rule.description,
            rule.detection_target,
            rule.condition,
            *rule.required_evidence,
        ]
    ).lower()
    spatial_terms = (
        "restricted area",
        "restricted zone",
        "zone",
        "boundary",
        "enters the area",
        "enters an area",
        "entering",
        "crossing",
        "remaining within",
        "spatial boundaries",
        "inside the area",
        "within the area",
    )
    return rule.hazard_type == "restricted_area" or any(
        term in combined_text for term in spatial_terms
    )


@app.post("/rules/validate")
def validate_safety_rule(request: RuleValidationRequest):
    if (
        CONDITION_REGISTRY.get(request.rule.condition.strip().lower(), {}).get("kind")
        == "overlaps_zone"
        and request.zone is not None
    ):
        try:
            validate_overlap_zone(request.zone)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return validate_rule_draft(request.rule, zone_configured=request.zone is not None)


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


@app.get("/capabilities")
def get_capabilities():
    detector_capabilities = []
    for detector in DETECTOR_REGISTRY.list_detectors():
        detector_capabilities.append(
            {
                "detector_id": detector.detector_id,
                "display_name": detector.display_name,
                "behaviors": sorted(detector.supported_behaviors()),
                "targets": [
                    {"class_id": class_id, "class_name": class_name}
                    for class_id, class_name in detector.available_classes().items()
                ],
            }
        )
    return {
        "detectors": detector_capabilities,
        "targets": DETECTOR_REGISTRY.available_targets(),
        "capabilities": [
            capability.model_dump()
            for capability in DETECTOR_REGISTRY.capability_catalog()
        ],
        "target_aliases": LEGACY_TARGET_ALIASES,
        "conditions": [
            *[
                {
                    "condition": condition,
                    "description": condition.replace("_", " "),
                    "spatial": capability["spatial"],
                    "required_target": capability.get("target_class"),
                    "parameter": capability.get("parameter"),
                    "required_capabilities": capability.get(
                        "required_capabilities", []
                    ),
                    "required_capability_types": capability.get(
                        "required_capability_types", []
                    ),
                }
                for condition, capability in CONDITION_REGISTRY.items()
            ],
            *[
                {
                    **capability.model_dump(),
                    "spatial": capability.requires_zone,
                    "required_target": None,
                    "parameter": None,
                }
                for evaluator in DETECTOR_REGISTRY.list_evaluators()
                for capability in evaluator.conditions()
            ],
        ],
        "limitations": [
            "Available object targets come from registered detector metadata.",
            "Natural-language parsing creates a structured request; it does not add visual detection capabilities.",
            "Object detection does not establish attributes, states, tracking, spatial relationships, or behaviour unless a registered capability explicitly provides them.",
        ],
        "behaviors": sorted(
            {
                behavior
                for detector in DETECTOR_REGISTRY.list_detectors()
                for behavior in detector.supported_behaviors()
            }
        ),
    }


def parse_generated_parameter_value(value: str) -> float | str | bool:
    normalized = value.strip().lower()
    if normalized in {"true", "false"}:
        return normalized == "true"
    try:
        return float(value)
    except ValueError:
        return value


def is_simple_spatial_presence_instruction(
    instruction: str,
    detection_target: str,
) -> bool:
    normalized = " ".join(re.findall(r"[a-z0-9_]+", instruction.lower()))
    normalized = re.sub(
        r"^(?:please\s+)?(?:alert(?:\s+me)?\s+when|flag(?:\s+when)?|"
        r"detect(?:\s+when)?|notify(?:\s+me)?\s+when|monitor(?:\s+for)?|"
        r"identify(?:\s+when)?)\s+",
        "",
        normalized,
    )
    target = normalize_detection_target(detection_target)
    normalized = re.sub(
        rf"\b(?:a|an|the)\s+{re.escape(target)}\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return bool(
        re.fullmatch(
            r"(?:is|are|remains?|stays?|located|present)?\s*"
            r"(?:inside|within|in)\s+(?:(?:the|a)\s+)?"
            r"(?:restricted\s+)?(?:area|zone)",
            normalized,
        )
    )


def remove_unrequested_behaviour_requirements(
    rule_data: dict[str, Any],
    instruction: str,
) -> None:
    if not is_simple_spatial_presence_instruction(
        instruction,
        rule_data["detection_target"],
    ):
        return

    behaviour_ids = {
        capability.capability_id
        for capability in DETECTOR_REGISTRY.capability_catalog()
        if capability.capability_type == "behaviour_recognition"
    }
    behaviour_ids.update(
        capability_id
        for capability_id in rule_data["required_capabilities"]
        if re.search(r"behavio[u]?r", capability_id, re.IGNORECASE)
    )
    rule_data["required_capability_types"] = [
        capability_type
        for capability_type in rule_data["required_capability_types"]
        if capability_type != "behaviour_recognition"
    ]
    rule_data["required_capabilities"] = [
        capability_id
        for capability_id in rule_data["required_capabilities"]
        if capability_id not in behaviour_ids
    ]


@app.post("/rules/generate")
def generate_safety_rule(request: SafetyRuleRequest):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "Natural-language generation is unavailable because GEMINI_API_KEY "
                "is not configured on the backend. The deterministic restricted-area "
                "and bounding-box-overlap presets remain available."
            ),
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
- Classify each requested visual ability separately as object_detection, attribute_classification, tracking, spatial_analysis, or behaviour_recognition. Do not infer one from another.
- Set required_capability_types and required_capabilities from the actual user request. Use the provided capability IDs and types; include missing requirements too, so validation can explain what is unavailable.
- The currently registered targets, capability contracts, and conditions are:
{json.dumps({"targets": DETECTOR_REGISTRY.available_targets(), "capabilities": [item.model_dump() for item in DETECTOR_REGISTRY.capability_catalog()], "conditions": get_capabilities()["conditions"]}, ensure_ascii=True)}
- Preserve the requested target/condition even if unsupported. Do not substitute a supported class or capability; describe the limitation in the draft.
- Return condition_parameters as a list of objects with name and value (value as text); use an empty list when no parameters are needed.
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
                response_schema=GeneratedSafetyRule,
            ),
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response. Please retry.",
            )

        generated_rule = GeneratedSafetyRule.model_validate_json(response.text)
        rule_data = generated_rule.model_dump()
        remove_unrequested_behaviour_requirements(
            rule_data,
            request.instruction,
        )
        parameter_entries = rule_data["condition_parameters"]
        parameter_names = [entry["name"] for entry in parameter_entries]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Generated condition parameter names must be unique.")
        rule_data["condition_parameters"] = {
            entry["name"]: parse_generated_parameter_value(entry["value"])
            for entry in parameter_entries
        }
        rule = SafetyRule.model_validate(rule_data)

        return {
            "status": "draft",
            "model": MODEL_NAME,
            "original_instruction": request.instruction,
            "rule": rule.model_dump(),
            "capability_assessment": validate_rule_draft(rule),
            "message": ("Draft generated. Validate it before enabling monitoring."),
        }

    except HTTPException:
        raise
    except ValidationError as exc:
        invalid_fields = sorted(
            {
                ".".join(str(part) for part in error["loc"])
                for error in exc.errors()
            }
        )
        print(
            "Rule generation schema validation failed for field(s): "
            + ", ".join(invalid_fields)
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini returned a draft that did not match Rulevia's required "
                "rule format. Retry or use the deterministic restricted-area "
                "and bounding-box-overlap presets."
            ),
        )
    except GeminiAPIError as exc:
        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        print(
            "Rule generation failed: "
            f"{type(exc).__name__} status={status_code or 'unknown'}"
        )
        if status_code == 429:
            detail = (
                "Gemini is rate-limited or its quota is exhausted. Try again later; "
                "the deterministic restricted-area and bounding-box-overlap presets "
                "remain available."
            )
        elif status_code in {401, 403}:
            detail = (
                "Gemini rejected the backend credentials or access to the configured "
                "model. Verify backend GEMINI_API_KEY and model access; the "
                "deterministic restricted-area and bounding-box-overlap presets "
                "remain available."
            )
        elif status_code == 404:
            detail = (
                "The configured Gemini model is unavailable. Check GEMINI_MODEL; "
                "the deterministic restricted-area and bounding-box-overlap "
                "presets remain available."
            )
        else:
            detail = (
                "Gemini returned an upstream error. Check backend logs, model "
                "availability, and quota; the deterministic restricted-area and "
                "bounding-box-overlap presets remain available."
            )
        raise HTTPException(
            status_code=502,
            detail=detail,
        )
    except ValueError as exc:
        print(f"Rule generation request rejected: {type(exc).__name__}")
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini could not process the structured rule request. This can "
                "indicate an unsupported response schema or model configuration. "
                "The deterministic restricted-area and bounding-box-overlap "
                "presets remain available."
            ),
        )




@app.post("/videos/upload-extract")
async def upload_and_extract_video(
    file: Annotated[UploadFile, File(...)],
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


@app.post("/rules/evaluate-frame")
async def evaluate_live_frame(
    file: Annotated[UploadFile, File(...)],
    rule_json: Annotated[str, Form(alias="rule")],
    zone_json: Annotated[str | None, Form(alias="zone")] = None,
    timestamp_seconds: Annotated[float, Form()] = 0.0,
):
    max_payload_bytes = 5 * 1024 * 1024
    max_dimension = 8192
    max_pixels = 12_000_000

    try:
        if file.content_type not in {"image/jpeg", "image/jpg"}:
            raise HTTPException(
                status_code=415,
                detail="Upload a JPEG webcam frame.",
            )
        encoded_image = await file.read(max_payload_bytes + 1)
        if len(encoded_image) > max_payload_bytes:
            raise HTTPException(
                status_code=413,
                detail="Webcam frame exceeds the 5 MB limit.",
            )
        image = cv2.imdecode(
            np.frombuffer(encoded_image, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None or image.size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded frame is not a decodable JPEG image.",
            )
        frame_height, frame_width = image.shape[:2]
        if (
            frame_width <= 0
            or frame_height <= 0
            or frame_width > max_dimension
            or frame_height > max_dimension
            or frame_width * frame_height > max_pixels
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Webcam frame dimensions exceed the supported limit "
                    "(8192 pixels per side and 12 megapixels total)."
                ),
            )

        try:
            rule = SafetyRule.model_validate_json(rule_json)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "The submitted rule is invalid.",
                    "errors": exc.errors(),
                },
            ) from exc

        zone = None
        if zone_json:
            try:
                zone_data = json.loads(zone_json)
                if not isinstance(zone_data, list):
                    raise TypeError("The zone must be a JSON array of points.")
                zone = validate_zone_polygon(
                    [ZonePoint.model_validate(point) for point in zone_data]
                )
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid zone polygon: {exc}",
                ) from exc

        condition = rule.condition.strip().lower()
        condition_capability, _ = rule_condition_capability(condition)
        if isinstance(condition_capability, RuleConditionCapability):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Live frame evaluation does not support plugin condition "
                    f"'{condition}' yet."
                ),
            )
        if condition_capability is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported evaluation condition: '{rule.condition}'.",
            )
        condition_kind = condition_capability["kind"]
        if condition_kind == "enters_zone":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Entry conditions require adjacent observations from a live "
                    "session and are not supported by single-frame evaluation. "
                    "Select a presence or overlap condition instead."
                ),
            )
        if condition_kind not in {
            "presence",
            "absence",
            "point_in_zone",
            "overlaps_zone",
        }:
            raise HTTPException(
                status_code=400,
                detail=f"Condition '{condition}' is not supported for live frames.",
            )
        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise HTTPException(
                status_code=422,
                detail="timestamp_seconds must be a finite non-negative value.",
            )

        assessment = validate_rule_draft(rule, zone_configured=zone is not None)
        if assessment["status"] != "ready_for_evaluation":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "The selected rule is not ready for live evaluation. "
                        "Resolve its capability or configuration findings first."
                    ),
                    "validation": assessment,
                },
            )

        spatial_condition = bool(condition_capability.get("spatial"))
        evaluation_zone = zone if spatial_condition else None
        if spatial_condition and evaluation_zone is None:
            raise HTTPException(
                status_code=400,
                detail=f"Condition '{condition}' requires a valid zone polygon.",
            )

        resolved_target = DETECTOR_REGISTRY.resolve_target(
            rule.detection_target,
            rule.detector_id,
        )
        if resolved_target is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        f"Requested target '{rule.detection_target}' is not "
                        "available from the selected detector."
                    ),
                    "missing_capability": "detector_class",
                },
            )
        detector, target_class_id, target_class = resolved_target
        overlap_condition = condition_kind == "overlaps_zone"
        overlap_threshold = rule.condition_parameters.get(
            "overlap_threshold",
            rule.zone_overlap_threshold,
        )
        zone_triangles = None
        if overlap_condition:
            if (
                isinstance(overlap_threshold, bool)
                or not isinstance(overlap_threshold, (int, float))
                or not math.isfinite(overlap_threshold)
                or not 0 < overlap_threshold <= 1
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{condition}' requires overlap_threshold greater than 0 "
                        "and at most 1."
                    ),
                )
            if evaluation_zone is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Condition '{condition}' requires a valid zone polygon.",
                )
            try:
                zone_triangles = validate_overlap_zone(evaluation_zone)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        if condition_kind == "point_in_zone" and evaluation_zone is None:
            raise HTTPException(
                status_code=400,
                detail=f"Condition '{condition}' requires a valid zone polygon.",
            )

        detector_frames = detector.predict(
            source=image,
            confidence=rule.confidence_threshold,
        )
        detections = []
        for detector_frame in detector_frames:
            for detected_object in detector_frame.detections:
                if detected_object.class_id != target_class_id:
                    continue
                evidence = {
                    "class_name": detected_object.class_name.lower(),
                    "class_id": detected_object.class_id,
                    "detector_id": detector.detector_id,
                    "confidence": round(detected_object.confidence, 3),
                    "box": [round(value, 2) for value in detected_object.box],
                }
                if evaluation_zone is not None:
                    spatial_diagnostic, spatial_match = evaluate_spatial_detection(
                        detected_object,
                        frame_width,
                        frame_height,
                        evaluation_zone,
                        condition_kind,
                        float(overlap_threshold) if overlap_condition else None,
                        zone_triangles,
                    )
                    evidence.update(spatial_diagnostic)
                    evidence["matches"] = spatial_match
                else:
                    evidence["matches"] = True
                detections.append(evidence)

        if condition_kind == "absence":
            condition_met = not detections
            matches = []
        else:
            matches = [detection for detection in detections if detection["matches"]]
            condition_met = bool(matches)

        return {
            "status": "success",
            "rule_name": rule.rule_name,
            "target_class": target_class,
            "detector_id": detector.detector_id,
            "condition": condition,
            "timestamp_seconds": timestamp_seconds,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "condition_met": condition_met,
            "evaluation_status": "needs_human_review",
            "human_approval_required": True,
            "detection_count": len(detections),
            "detections": detections,
            "matches": matches,
            "zone": (
                [point.model_dump() for point in evaluation_zone]
                if evaluation_zone
                else None
            ),
            "zone_overlap_threshold": overlap_threshold if overlap_condition else None,
            "evidence": {
                "frame_retained": False,
                "description": (
                    "Detection boxes and spatial diagnostics are transient; the "
                    "webcam image was not stored."
                ),
            },
            "limitations": [
                "Detections are uncertain evidence and require human review.",
                "A single live frame cannot establish an entry transition.",
                "The webcam frame is not stored by this endpoint.",
            ],
        }
    finally:
        await file.close()


@app.post("/rules/evaluate-video")
def evaluate_video_rule(request: VideoRuleEvaluationRequest):
    # Only accept video IDs generated as UUIDs by the upload endpoint.
    try:
        safe_video_id = str(uuid.UUID(request.video_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID.")

    rule = request.rule
    condition = rule.condition.strip().lower()
    condition_capability, evaluator_plugin = rule_condition_capability(condition)
    if condition_capability is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported evaluation condition: '{rule.condition}'.",
        )
    required_types, required_ids = capability_requirements(
        rule,
        condition_capability,
    )
    resolved_target = DETECTOR_REGISTRY.resolve_target(
        rule.detection_target,
        rule.detector_id,
    )
    capability = (
        condition_capability if isinstance(condition_capability, dict) else None
    )
    executable_capabilities = capabilities_executable_for_rule(
        rule.detection_target,
        rule.detector_id,
        resolved_target,
        condition_capability,
        evaluator_plugin,
    )
    plugin_supports_target = any(
        any(
            normalize_detection_target(class_name)
            == normalize_detection_target(rule.detection_target)
            for class_name in item.supported_object_classes
        )
        for item in executable_capabilities
    )
    if evaluator_plugin and not plugin_supports_target:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"Evaluator '{evaluator_plugin.plugin_id}' does not advertise "
                    f"target '{rule.detection_target}'. Register a compatible "
                    "capability implementation for this target."
                ),
                "requested_target": rule.detection_target,
                "missing_capability": "object_detection",
                "capability_issues": [
                    {
                        "requirement": f"target:{rule.detection_target}",
                        "reason": (
                            "The selected evaluator has no capability descriptor "
                            "declaring this target."
                        ),
                        "needed": (
                            "Advertise the target in a capability descriptor and "
                            "implement it in this evaluator."
                        ),
                    }
                ],
            },
        )
    required_parameters = (
        {capability["parameter"]}
        if capability and capability.get("parameter")
        else set(
            condition_capability.required_parameters
            if isinstance(condition_capability, RuleConditionCapability)
            else []
        )
    )
    supported_parameters = required_parameters
    unknown_parameters = set(rule.condition_parameters) - supported_parameters
    if unknown_parameters:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported condition parameter(s): "
                + ", ".join(sorted(unknown_parameters))
            ),
        )
    missing_parameters = supported_parameters - set(rule.condition_parameters)
    if missing_parameters and isinstance(
        condition_capability,
        RuleConditionCapability,
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Required condition parameter(s) are missing.",
                "missing_parameters": sorted(missing_parameters),
            },
        )
    if resolved_target is None and not evaluator_plugin:
        available = DETECTOR_REGISTRY.available_targets()
        available_names = sorted({target["class_name"] for target in available})
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"Requested target '{rule.detection_target}' is not available "
                    "from the selected registered detector. No target substitution was made."
                ),
                "requested_target": rule.detection_target,
                "detector_id": rule.detector_id,
                "available_targets": available_names,
                "missing_capability": "detector_class",
            },
        )
    capability_issues = capability_requirement_issues(
        required_types,
        required_ids,
        executable_capabilities,
        rule.detection_target,
        condition,
    )
    if capability_issues:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Required capabilities are not executable for the selected "
                    "target and condition. No substitute capability was used."
                ),
                "requested_capabilities": [
                    issue["requirement"] for issue in capability_issues
                ],
                "missing_capability": capability_issues[0]["requirement"],
                "capability_issues": capability_issues,
            },
        )
    if resolved_target:
        detector, target_class_id, target_class = resolved_target
    else:
        detector = None
        target_class_id = None
        target_class = rule.detection_target
    required_class = capability.get("target_class") if capability else None
    if required_class and normalize_detection_target(target_class) != required_class:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{condition}' retains its backward-compatible semantics and "
                f"requires target '{required_class}'. Use a generic object condition "
                "for other detector classes."
            ),
        )

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

    if evaluator_plugin is not None:
        if (
            condition_capability.requires_zone
            and request.zone is None
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Condition '{condition}' requires its configured spatial inputs."
                ),
            )
        plugin_result = evaluator_plugin.evaluate_video(request, frame_paths)
        if not isinstance(plugin_result, dict) or not isinstance(
            plugin_result.get("condition_met"), bool
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Evaluator '{evaluator_plugin.plugin_id}' returned an invalid "
                    "result; condition_met must be boolean."
                ),
            )
        return {
            **plugin_result,
            "status": "success",
            "video_id": safe_video_id,
            "rule_name": rule.rule_name,
            "requested_target": rule.detection_target,
            "target_class": target_class,
            "condition": rule.condition,
            "evaluation_status": "needs_human_review",
            "human_approval_required": True,
            "total_frames_evaluated": plugin_result.get(
                "total_frames_evaluated",
                len(frame_paths),
            ),
        }

    if capability is None:
        raise HTTPException(
            status_code=400,
            detail=f"Condition '{condition}' has no registered execution logic.",
        )

    spatial_evaluation = bool(capability.get("spatial"))
    zone = request.zone if spatial_evaluation else None

    if spatial_evaluation and zone is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "This rule requires a configured restricted-zone polygon "
                "before it can be evaluated."
            ),
        )

    overlap_condition = capability.get("parameter") == "overlap_threshold"
    overlap_threshold = rule.condition_parameters.get(
        "overlap_threshold",
        rule.zone_overlap_threshold,
    )
    if overlap_condition:
        if (
            isinstance(overlap_threshold, bool)
            or not isinstance(overlap_threshold, (int, float))
            or not math.isfinite(overlap_threshold)
            or overlap_threshold <= 0
            or overlap_threshold > 1
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{condition}' requires numeric overlap_threshold (legacy zone_overlap_threshold) greater than 0 and at most 1."
                ),
            )
        try:
            zone_triangles = validate_overlap_zone(zone)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        if rule.zone_overlap_threshold is not None or "overlap_threshold" in rule.condition_parameters:
            raise HTTPException(
                status_code=400,
                detail=(
                    "overlap_threshold is only supported with overlap conditions."
                ),
            )
        zone_triangles = None

    condition_kind = capability["kind"]
    entry_target_label = (
        "person" if normalize_detection_target(target_class) == "person" else "target"
    )
    evaluated_frames = []
    matching_frames = []
    entry_evidence = []
    frames_with_target = 0
    previous_single_target = None
    previous_target_unavailable_reason = "no_previous_frame_in_evaluation"

    for frame_path in frame_paths:
        frame_number = int(frame_path.stem.replace("frame_", ""))
        timestamp_seconds = (frame_number - 1) * 2

        detector_frames = detector.predict(
            source=str(frame_path),
            confidence=rule.confidence_threshold,
        )
        if detector_frames:
            detector_frame = detector_frames[0]
            frame_height = detector_frame.frame_height
            frame_width = detector_frame.frame_width
        else:
            frame_image = cv2.imread(str(frame_path))
            if frame_image is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not read extracted frame '{frame_path.name}'.",
                )
            frame_height, frame_width = frame_image.shape[:2]
        frame_detections = []
        frame_matches = []
        point_zone_matches = []
        overlap_frame_matches = []

        frame_detector_objects = detector_frame.detections if detector_frames else []
        for detected_object in frame_detector_objects:
            if detected_object.class_id != target_class_id:
                continue
            x1, y1, x2, y2 = detected_object.box
            class_name = detected_object.class_name.lower()
            match = {
                "class_name": class_name,
                "class_id": detected_object.class_id,
                "detector_id": detector.detector_id,
                "confidence": round(detected_object.confidence, 3),
                "box": [round(value, 2) for value in (x1, y1, x2, y2)],
            }

            if zone is not None:
                spatial_diagnostic, spatial_match = evaluate_spatial_detection(
                    detected_object,
                    frame_width,
                    frame_height,
                    zone,
                    condition_kind,
                    overlap_threshold,
                    zone_triangles,
                )
                match.update(spatial_diagnostic)
                inside_zone = spatial_diagnostic["inside_zone"]
                if inside_zone:
                    point_zone_matches.append(match)
                    if condition_kind != "overlaps_zone":
                        frame_matches.append(match)

                if condition_kind == "overlaps_zone" and spatial_match:
                    frame_matches.append(match)
                    overlap_frame_matches.append(match)
            else:
                frame_matches.append(match)

            frame_detections.append(match)

        if frame_detections:
            frames_with_target += 1

        zone_diagnostic = None
        if zone is not None:
            zone_diagnostic = {
                "detection_count": len(frame_detections),
                "observations": [
                    {
                        "reference_point": detection["anchor_point"],
                        "inside_zone": detection["inside_zone"],
                    }
                    for detection in frame_detections
                ],
                "matching_detection_count": len(point_zone_matches),
                "matching_status": bool(point_zone_matches),
            }

        overlap_diagnostic = None
        if condition_kind == "overlaps_zone":
            overlap_diagnostic = {
                "detection_count": len(frame_detections),
                "threshold": overlap_threshold,
                "observations": [
                    {
                        "box": detection["box"],
                        "overlap_ratio": detection["zone_overlap_ratio"],
                        "matches": detection["overlaps_zone"],
                    }
                    for detection in frame_detections
                ],
                "matching_detection_count": len(overlap_frame_matches),
                "matching_status": bool(overlap_frame_matches),
            }

        entry_detected = False
        entry_transition = None
        if condition_kind == "enters_zone":
            observations = [
                {
                    "reference_point": detection["anchor_point"],
                    "inside_zone": detection["inside_zone"],
                }
                for detection in frame_detections
            ]
            entry_transition = {
                "detection_count": len(frame_detections),
                "observations": observations,
                "transition_evaluated": False,
                "transition_result": "not_evaluated",
                "reason": "",
            }

            if len(frame_detections) == 1:
                current_person = frame_detections[0]
                if previous_single_target is None:
                    entry_transition["reason"] = previous_target_unavailable_reason
                elif previous_single_target["frame_number"] != frame_number - 1:
                    entry_transition["reason"] = (
                        "previous_detection_not_from_adjacent_frame"
                    )
                else:
                    entry_transition["transition_evaluated"] = True
                    entry_transition["transition_result"] = "no_entry"
                    previous_inside = previous_single_target["inside_zone"]
                    current_inside = current_person["inside_zone"]
                    if not previous_inside and current_inside:
                        entry_detected = True
                        entry_transition["transition_result"] = "entry_candidate"
                        entry_transition["reason"] = "observed_outside_to_inside"
                        entry_evidence.append(
                            {
                                "event_type": "observed_zone_entry_candidate",
                                "target_class": target_class,
                                "detector_id": detector.detector_id,
                                "frame": frame_path.name,
                                "frame_number": frame_number,
                                "timestamp_seconds": timestamp_seconds,
                                "previous_frame": previous_single_target["frame"],
                                "previous_timestamp_seconds": previous_single_target[
                                    "timestamp_seconds"
                                ],
                                "previous_box": previous_single_target["box"],
                                "previous_anchor_point": previous_single_target[
                                    "anchor_point"
                                ],
                                "box": current_person["box"],
                                "confidence": current_person["confidence"],
                                "anchor_point": current_person["anchor_point"],
                                "zone": [point.model_dump() for point in zone],
                            }
                        )
                    elif previous_inside and current_inside:
                        entry_transition["reason"] = "observed_inside_to_inside"
                    elif previous_inside and not current_inside:
                        entry_transition["reason"] = "observed_inside_to_outside"
                    else:
                        entry_transition["reason"] = "observed_outside_to_outside"

                previous_single_target = {
                    "frame": frame_path.name,
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "inside_zone": current_person["inside_zone"],
                    "box": current_person["box"],
                    "anchor_point": current_person["anchor_point"],
                }
                previous_target_unavailable_reason = (
                    f"previous_frame_had_single_{entry_target_label}"
                )
            else:
                previous_single_target = None
                previous_target_unavailable_reason = (
                    f"previous_frame_had_no_{entry_target_label}_detection"
                    if not frame_detections
                    else f"previous_frame_had_multiple_{entry_target_label}_detections"
                )
                entry_transition["reason"] = (
                    f"no_{entry_target_label}_detected"
                    if not frame_detections
                    else f"multiple_{entry_target_label}_detections"
                )

        frame_result = {
            "frame": frame_path.name,
            "timestamp_seconds": timestamp_seconds,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "target_observed": bool(frame_detections),
            "zone_match": bool(frame_matches) if zone is not None else None,
            "zone_diagnostic": zone_diagnostic,
            "overlap_diagnostic": overlap_diagnostic,
            "entry_detected": entry_detected,
            "entry_transition": entry_transition,
            "detections": frame_detections,
            "matches": frame_matches,
        }

        evaluated_frames.append(frame_result)

        if frame_matches:
            matching_frames.append(frame_result)

    if condition_kind == "enters_zone":
        matching_frames = [
            frame
            for frame in evaluated_frames
            if frame["entry_detected"]
        ]
        target_found = bool(entry_evidence)
    else:
        target_found = len(matching_frames) > 0

    if condition_kind == "presence":
        condition_met = target_found

    elif condition_kind == "absence":
        condition_met = not target_found

    elif condition_kind == "enters_zone":
        condition_met = target_found

    else:
        condition_met = target_found
    return {
        "status": "success",
        "video_id": safe_video_id,
        "rule_name": rule.rule_name,
        "target_class": target_class,
        "requested_target": rule.detection_target,
        "detector_id": detector.detector_id,
        "condition": rule.condition,
        "condition_met": condition_met,
        "evaluation_status": "needs_human_review",
        "human_approval_required": True,
        "total_frames_evaluated": len(evaluated_frames),
        "frames_with_target": frames_with_target,
        "frames_matching_condition": len(matching_frames),
        "zone": [point.model_dump() for point in zone] if zone is not None else None,
        "zone_overlap_threshold": overlap_threshold if overlap_condition else None,
        "entry_evidence": entry_evidence,
        "matching_frames": matching_frames,
        "frame_results": evaluated_frames,
        "limitations": [
            "Results depend on the detector and its confidence threshold.",
            "Not observing an object does not prove that it is absent.",
            (
                "This evaluation does not establish visual attributes, states, or "
                "compliance unless registered capabilities explicitly evaluate them."
            ),
            (
                "Zone presence is evaluated only on the extracted frames. "
                "It does not verify a continuous boundary crossing."
                if spatial_evaluation
                else "Spatial entry evaluation was not requested."
            ),
            (
                "Entry candidates use a conservative heuristic: exactly one "
                f"{target_class} must be detected in each adjacent sampled frame. "
                "The detector provides no stable track IDs, so the same "
                "identity cannot be guaranteed; missing or multiple detections "
                "break the comparison. Human review is required."
                if condition_kind == "enters_zone"
                else "Entry-transition detection was not requested."
            ),
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
