
from pathlib import Path

from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo11n.pt")

# Find an extracted frame from your videos
project_root = Path(__file__).resolve().parents[1]
frames_dir = project_root / "data" / "extracted_frames"

frames = list(frames_dir.glob("*/*.jpg"))

if not frames:
    print("No extracted frames found. Upload a video first.")
else:
    frame_path = frames[0]
    print(f"Testing frame: {frame_path}")

    results = model.predict(
        source=str(frame_path),
        conf=0.25,
        save=True
    )

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            print(
                f"Detected: {model.names[class_id]} "
                f"| Confidence: {confidence:.2f}"
            )

    print("YOLO detection test completed!")
