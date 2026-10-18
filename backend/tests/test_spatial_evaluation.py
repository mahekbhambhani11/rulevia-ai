import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

import cv2
import numpy as np
from app.main import (
    SUPPORTED_SPATIAL_CONDITION,
    SUPPORTED_ZONE_ENTRY_CONDITION,
    SUPPORTED_ZONE_OVERLAP_CONDITION,
    DetectorCapabilityRegistry,
    GeneratedSafetyRule,
    RuleConditionCapability,
    SafetyRule,
    SafetyRuleRequest,
    VideoRuleEvaluationRequest,
    VisualCapability,
    YoloDetectorAdapter,
    ZonePoint,
    app,
    bounding_box_polygon_overlap_ratio,
    evaluate_video_rule,
    generate_safety_rule,
    get_capabilities,
    point_in_polygon,
    validate_overlap_zone,
    validate_rule_draft,
    validate_zone_polygon,
)
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError


def make_rule(
    condition: str = SUPPORTED_SPATIAL_CONDITION,
    zone_overlap_threshold: float | None = None,
) -> SafetyRule:
    return SafetyRule(
        rule_name="Restricted zone entry",
        description="Flag people entering a restricted area.",
        hazard_type="restricted_area",
        detection_target="person",
        condition=condition,
        zone_overlap_threshold=zone_overlap_threshold,
        severity="high",
        required_evidence=["person_detection", "zone"],
        recommended_action="Review the evidence.",
        confidence_threshold=0.25,
        limitations=["Sampled frames may miss brief events."],
    )


class FakeCoordinate:
    def __init__(self, values: list[float]):
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeBox:
    def __init__(self, coordinates: list[float], class_id: int = 0):
        self.cls = [class_id]
        self.conf = [0.9]
        self.xyxy = [FakeCoordinate(coordinates)]


class FakeModel:
    names: ClassVar[dict[int, str]] = {0: "person"}

    def __init__(
        self,
        detections_by_frame: dict[int, list[list[float]]] | None = None,
        names: dict[int, str] | None = None,
        class_id: int = 0,
    ) -> None:
        self.calls: list[str | np.ndarray] = []
        self.detections_by_frame = detections_by_frame
        self.names = names or {0: "person"}
        self.class_id = class_id

    def predict(
        self,
        source: str | np.ndarray,
        conf: float,
        verbose: bool,
    ) -> list[SimpleNamespace]:
        self.calls.append(source)
        frame_number = (
            1
            if isinstance(source, np.ndarray)
            else int(Path(source).stem.removeprefix("frame_"))
        )
        if self.detections_by_frame is None:
            boxes = [
                [40, 40, 60, 70]
                if frame_number == 1
                else [80, 40, 90, 70]
            ]
        else:
            boxes = self.detections_by_frame.get(frame_number, [])
        return [
            SimpleNamespace(
                orig_shape=(100, 100),
                boxes=[FakeBox(coordinates, self.class_id) for coordinates in boxes],
            )
        ]


class FakeRulePlugin:
    plugin_id = "visual-attribute-plugin"

    def capabilities(self) -> list[VisualCapability]:
        return [
            VisualCapability(
                capability_id="visual-attribute-plugin.attribute_classification",
                detector_id=self.plugin_id,
                capability_type="attribute_classification",
                description="Classifies a configured visual attribute.",
                supported_object_classes=["person"],
                supported_attributes=["uniform_color"],
                supported_states=["visible", "not_visible"],
                required_inputs=["image_frame", "object_region"],
                configuration_schema={"attribute": {"type": "string"}},
                known_limitations=["Unvalidated demonstration plugin."],
                validation_status="unvalidated",
            )
        ]

    def conditions(self) -> list[RuleConditionCapability]:
        return [
            RuleConditionCapability(
                condition="object_has_configured_attribute",
                description="Evaluates a plugin-defined visual attribute.",
                required_capabilities=[
                    "visual-attribute-plugin.attribute_classification"
                ],
                required_capability_types=["attribute_classification"],
                required_parameters=["attribute"],
            )
        ]

    def evaluate_video(
        self,
        request: VideoRuleEvaluationRequest,
        frame_paths: list[Path],
    ) -> dict:
        return {
            "condition_met": True,
            "total_frames_evaluated": len(frame_paths),
            "frame_results": [
                {"frame": frame.name, "plugin_evidence": "test-only"}
                for frame in frame_paths
            ],
        }


class SpatialEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zone = [
            ZonePoint(x=0.25, y=0.25),
            ZonePoint(x=0.75, y=0.25),
            ZonePoint(x=0.75, y=0.75),
            ZonePoint(x=0.25, y=0.75),
        ]

    def post_live_frame(
        self,
        rule: SafetyRule | None = None,
        zone: list[ZonePoint] | None = None,
        image_bytes: bytes | None = None,
        content_type: str = "image/jpeg",
    ):
        if image_bytes is None:
            encoded, buffer = cv2.imencode(
                ".jpg",
                np.zeros((100, 100, 3), dtype=np.uint8),
            )
            self.assertTrue(encoded)
            image_bytes = buffer.tobytes()
        data = {
            "rule": (rule or make_rule("object_in_zone")).model_dump_json(),
            "timestamp_seconds": "1.25",
        }
        if zone is not None:
            data["zone"] = json.dumps([point.model_dump() for point in zone])
        return TestClient(app).post(
            "/rules/evaluate-frame",
            files={"file": ("webcam.jpg", image_bytes, content_type)},
            data=data,
        )

    def evaluate_detections(
        self,
        detections_by_frame: dict[int, list[list[float]]],
        frame_count: int,
        condition: str = SUPPORTED_ZONE_ENTRY_CONDITION,
        zone_overlap_threshold: float | None = None,
    ) -> dict:
        video_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as directory:
            frames_dir = Path(directory) / video_id
            frames_dir.mkdir()
            for frame_number in range(1, frame_count + 1):
                (frames_dir / f"frame_{frame_number:05d}.jpg").touch()

            request = VideoRuleEvaluationRequest(
                video_id=video_id,
                rule=make_rule(condition, zone_overlap_threshold),
                zone=self.zone,
            )
            with (
                patch("app.main.FRAMES_DIR", Path(directory)),
                patch(
                    "app.main.YOLO_MODEL",
                    FakeModel(detections_by_frame),
                ),
            ):
                return evaluate_video_rule(request)

    def test_point_in_polygon_includes_interior_and_boundary(self) -> None:
        self.assertTrue(point_in_polygon((0.5, 0.5), self.zone))
        self.assertTrue(point_in_polygon((0.25, 0.5), self.zone))
        self.assertFalse(point_in_polygon((0.9, 0.5), self.zone))

    def test_zone_requires_three_distinct_points_and_nonzero_area(self) -> None:
        with self.assertRaises(ValueError):
            validate_zone_polygon(self.zone[:2])

        with self.assertRaises(ValueError):
            validate_zone_polygon(
                [
                    ZonePoint(x=0.1, y=0.1),
                    ZonePoint(x=0.2, y=0.2),
                    ZonePoint(x=0.3, y=0.3),
                ]
            )

    def test_zone_coordinates_are_bounded(self) -> None:
        with self.assertRaises(ValueError):
            ZonePoint(x=1.1, y=0.5)

    def test_zone_overlap_ratio_uses_actual_concave_polygon(self) -> None:
        concave_zone = [
            ZonePoint(x=0.1, y=0.1),
            ZonePoint(x=0.9, y=0.1),
            ZonePoint(x=0.9, y=0.3),
            ZonePoint(x=0.3, y=0.3),
            ZonePoint(x=0.3, y=0.9),
            ZonePoint(x=0.1, y=0.9),
        ]
        triangles = validate_overlap_zone(concave_zone)

        self.assertEqual(len(triangles), 4)
        self.assertEqual(
            bounding_box_polygon_overlap_ratio((0.6, 0.6, 0.8, 0.8), concave_zone),
            0.0,
        )
        self.assertAlmostEqual(
            bounding_box_polygon_overlap_ratio((0.2, 0.2, 0.4, 0.4), concave_zone),
            0.75,
        )

    def test_overlap_zone_rejects_self_intersecting_polygon(self) -> None:
        crossed_zone = [
            ZonePoint(x=0.1, y=0.1),
            ZonePoint(x=0.9, y=0.1),
            ZonePoint(x=0.2, y=0.8),
            ZonePoint(x=0.8, y=0.8),
            ZonePoint(x=0.5, y=0.05),
        ]

        with self.assertRaisesRegex(ValueError, "non-self-intersecting"):
            validate_overlap_zone(crossed_zone)

    def test_evaluation_matches_only_detections_inside_the_zone(self) -> None:
        video_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as directory:
            frames_dir = Path(directory) / video_id
            frames_dir.mkdir()
            for frame_name in ("frame_00001.jpg", "frame_00002.jpg"):
                (frames_dir / frame_name).touch()

            request = VideoRuleEvaluationRequest(
                video_id=video_id,
                rule=make_rule(),
                zone=self.zone,
            )
            model = FakeModel()
            with (
                patch("app.main.FRAMES_DIR", Path(directory)),
                patch("app.main.YOLO_MODEL", model),
            ):
                result = evaluate_video_rule(request)

        self.assertTrue(result["condition_met"])
        self.assertEqual(result["condition"], SUPPORTED_SPATIAL_CONDITION)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(result["frames_with_target"], 2)
        self.assertEqual(result["frames_matching_condition"], 1)
        self.assertEqual(len(result["matching_frames"]), 1)
        self.assertEqual(
            result["matching_frames"][0]["frame"],
            "frame_00001.jpg",
        )
        self.assertTrue(result["matching_frames"][0]["matches"][0]["inside_zone"])
        self.assertFalse(result["frame_results"][1]["zone_match"])
        self.assertIsNone(result["frame_results"][0]["entry_transition"])

    def test_person_in_zone_matches_each_frame_with_any_inside_reference_point(self) -> None:
        result = self.evaluate_detections(
            {
                1: [[40, 40, 60, 70]],
                2: [[42, 40, 62, 70]],
                3: [[44, 40, 64, 70]],
            },
            frame_count=3,
            condition=SUPPORTED_SPATIAL_CONDITION,
        )

        self.assertTrue(result["condition_met"])
        self.assertEqual(result["frames_matching_condition"], 3)
        self.assertEqual(len(result["matching_frames"]), 3)
        self.assertEqual(
            [frame["frame"] for frame in result["matching_frames"]],
            ["frame_00001.jpg", "frame_00002.jpg", "frame_00003.jpg"],
        )
        self.assertEqual(
            result["zone"],
            [point.model_dump() for point in self.zone],
        )
        self.assertEqual(
            result["frames_matching_condition"],
            len(result["matching_frames"]),
        )
        for frame in result["frame_results"]:
            diagnostic = frame["zone_diagnostic"]
            self.assertEqual(diagnostic["detection_count"], 1)
            self.assertEqual(diagnostic["matching_detection_count"], 1)
            self.assertTrue(diagnostic["matching_status"])
            self.assertTrue(diagnostic["observations"][0]["inside_zone"])

    def test_person_in_zone_frame_matches_when_any_person_is_inside(self) -> None:
        result = self.evaluate_detections(
            {
                1: [
                    [80, 40, 90, 70],
                    [40, 40, 60, 70],
                ]
            },
            frame_count=1,
            condition=SUPPORTED_SPATIAL_CONDITION,
        )

        self.assertTrue(result["condition_met"])
        self.assertEqual(result["frames_matching_condition"], 1)
        self.assertEqual(result["frame_results"][0]["zone_diagnostic"]["detection_count"], 2)
        self.assertEqual(
            result["frame_results"][0]["zone_diagnostic"]["matching_detection_count"],
            1,
        )
        self.assertEqual(
            len(result["matching_frames"][0]["matches"]),
            1,
        )

    def test_person_in_zone_does_not_match_frames_with_only_outside_points(self) -> None:
        result = self.evaluate_detections(
            {
                1: [[80, 40, 90, 70]],
                2: [[82, 40, 92, 70]],
                3: [[84, 40, 94, 70]],
            },
            frame_count=3,
            condition=SUPPORTED_SPATIAL_CONDITION,
        )

        self.assertFalse(result["condition_met"])
        self.assertEqual(result["frames_matching_condition"], 0)
        self.assertEqual(result["matching_frames"], [])
        for frame in result["frame_results"]:
            diagnostic = frame["zone_diagnostic"]
            self.assertEqual(diagnostic["detection_count"], 1)
            self.assertEqual(diagnostic["matching_detection_count"], 0)
            self.assertFalse(diagnostic["matching_status"])
            self.assertFalse(diagnostic["observations"][0]["inside_zone"])

    def test_overlapping_box_does_not_match_when_bottom_center_is_outside(self) -> None:
        result = self.evaluate_detections(
            {1: [[60, 40, 90, 90]]},
            frame_count=1,
            condition=SUPPORTED_SPATIAL_CONDITION,
        )

        frame = result["frame_results"][0]
        self.assertTrue(result["frame_results"][0]["target_observed"])
        self.assertFalse(frame["zone_match"])
        self.assertEqual(result["frames_matching_condition"], 0)
        observation = frame["zone_diagnostic"]["observations"][0]
        self.assertEqual(observation["reference_point"], {"x": 0.75, "y": 0.9})
        self.assertFalse(observation["inside_zone"])

    def test_person_overlaps_zone_does_not_match_without_overlap(self) -> None:
        result = self.evaluate_detections(
            {1: [[80, 40, 90, 60]]},
            frame_count=1,
            condition=SUPPORTED_ZONE_OVERLAP_CONDITION,
            zone_overlap_threshold=0.2,
        )

        diagnostic = result["frame_results"][0]["overlap_diagnostic"]
        self.assertEqual(diagnostic["detection_count"], 1)
        self.assertEqual(diagnostic["observations"][0]["overlap_ratio"], 0)
        self.assertEqual(diagnostic["threshold"], 0.2)
        self.assertFalse(diagnostic["observations"][0]["matches"])
        self.assertFalse(diagnostic["matching_status"])
        self.assertEqual(result["frames_matching_condition"], 0)

    def test_person_overlaps_zone_reports_threshold_without_detections(self) -> None:
        result = self.evaluate_detections(
            {1: []},
            frame_count=1,
            condition=SUPPORTED_ZONE_OVERLAP_CONDITION,
            zone_overlap_threshold=0.2,
        )

        diagnostic = result["frame_results"][0]["overlap_diagnostic"]
        self.assertEqual(diagnostic["detection_count"], 0)
        self.assertEqual(diagnostic["observations"], [])
        self.assertEqual(diagnostic["threshold"], 0.2)
        self.assertFalse(diagnostic["matching_status"])

    def test_person_overlaps_zone_below_threshold_does_not_match(self) -> None:
        result = self.evaluate_detections(
            {1: [[70, 40, 90, 60]]},
            frame_count=1,
            condition=SUPPORTED_ZONE_OVERLAP_CONDITION,
            zone_overlap_threshold=0.3,
        )

        observation = result["frame_results"][0]["overlap_diagnostic"][
            "observations"
        ][0]
        self.assertEqual(observation["overlap_ratio"], 0.25)
        self.assertEqual(
            result["frame_results"][0]["overlap_diagnostic"]["threshold"],
            0.3,
        )
        self.assertFalse(observation["matches"])
        self.assertEqual(result["frames_matching_condition"], 0)

    def test_person_overlaps_zone_meeting_threshold_matches(self) -> None:
        result = self.evaluate_detections(
            {1: [[70, 40, 90, 60]]},
            frame_count=1,
            condition=SUPPORTED_ZONE_OVERLAP_CONDITION,
            zone_overlap_threshold=0.25,
        )

        observation = result["frame_results"][0]["overlap_diagnostic"][
            "observations"
        ][0]
        self.assertEqual(observation["overlap_ratio"], 0.25)
        self.assertTrue(observation["matches"])
        self.assertEqual(result["frames_matching_condition"], 1)
        self.assertEqual(len(result["matching_frames"]), 1)
        self.assertTrue(result["human_approval_required"])
        self.assertEqual(result["evaluation_status"], "needs_human_review")

    def test_person_overlaps_zone_full_containment_has_ratio_one(self) -> None:
        result = self.evaluate_detections(
            {1: [[40, 40, 60, 60]]},
            frame_count=1,
            condition=SUPPORTED_ZONE_OVERLAP_CONDITION,
            zone_overlap_threshold=1.0,
        )

        observation = result["frame_results"][0]["overlap_diagnostic"][
            "observations"
        ][0]
        self.assertEqual(observation["overlap_ratio"], 1.0)
        self.assertTrue(observation["matches"])

    def test_person_overlaps_zone_diagnostics_report_multiple_detections(self) -> None:
        result = self.evaluate_detections(
            {
                1: [
                    [70, 40, 90, 60],
                    [80, 40, 90, 60],
                ]
            },
            frame_count=1,
            condition=SUPPORTED_ZONE_OVERLAP_CONDITION,
            zone_overlap_threshold=0.2,
        )

        diagnostic = result["frame_results"][0]["overlap_diagnostic"]
        self.assertEqual(diagnostic["detection_count"], 2)
        self.assertEqual(diagnostic["matching_detection_count"], 1)
        self.assertEqual(
            [item["matches"] for item in diagnostic["observations"]],
            [True, False],
        )
        self.assertEqual(result["frames_matching_condition"], 1)

    def test_person_overlaps_zone_requires_valid_threshold(self) -> None:
        with self.assertRaises(HTTPException) as missing_error:
            self.evaluate_detections(
                {1: [[40, 40, 60, 60]]},
                frame_count=1,
                condition=SUPPORTED_ZONE_OVERLAP_CONDITION,
            )
        self.assertEqual(missing_error.exception.status_code, 400)
        self.assertIn("zone_overlap_threshold", missing_error.exception.detail)

        for threshold in (0.0, 1.01, float("inf"), float("nan")):
            with (
                self.subTest(threshold=threshold),
                self.assertRaises(ValidationError),
            ):
                make_rule(
                    SUPPORTED_ZONE_OVERLAP_CONDITION,
                    zone_overlap_threshold=threshold,
                )

    def test_overlap_threshold_is_only_supported_for_overlap_condition(self) -> None:
        with self.assertRaises(HTTPException) as error:
            self.evaluate_detections(
                {1: [[40, 40, 60, 60]]},
                frame_count=1,
                condition=SUPPORTED_SPATIAL_CONDITION,
                zone_overlap_threshold=0.2,
            )

        self.assertEqual(error.exception.status_code, 400)
        self.assertIn("only supported", error.exception.detail)

    def test_rule_validation_requires_overlap_threshold(self) -> None:
        result = validate_rule_draft(
            make_rule(SUPPORTED_ZONE_OVERLAP_CONDITION),
            zone_configured=True,
        )

        self.assertEqual(result["status"], "invalid")
        threshold_finding = next(
            finding
            for finding in result["findings"]
            if finding["check"] == "zone_overlap_threshold"
        )
        self.assertEqual(threshold_finding["status"], "failed")

    def test_entry_is_detected_for_outside_then_inside(self) -> None:
        result = self.evaluate_detections(
            {
                1: [[80, 40, 90, 70]],
                2: [[40, 40, 60, 70]],
            },
            frame_count=2,
        )

        self.assertTrue(result["condition_met"])
        self.assertEqual(result["condition"], SUPPORTED_ZONE_ENTRY_CONDITION)
        self.assertEqual(result["frames_matching_condition"], 1)
        self.assertEqual(len(result["entry_evidence"]), 1)
        evidence = result["entry_evidence"][0]
        self.assertEqual(evidence["frame"], "frame_00002.jpg")
        self.assertEqual(evidence["timestamp_seconds"], 2)
        self.assertEqual(evidence["previous_frame"], "frame_00001.jpg")
        self.assertEqual(evidence["previous_box"], [80, 40, 90, 70])
        self.assertEqual(
            evidence["previous_anchor_point"],
            {"x": 0.85, "y": 0.7},
        )
        self.assertEqual(evidence["box"], [40, 40, 60, 70])
        self.assertEqual(evidence["zone"], [point.model_dump() for point in self.zone])
        self.assertEqual(evidence["event_type"], "observed_zone_entry_candidate")
        self.assertTrue(result["human_approval_required"])
        self.assertEqual(result["evaluation_status"], "needs_human_review")
        first_transition = result["frame_results"][0]["entry_transition"]
        self.assertEqual(first_transition["detection_count"], 1)
        self.assertEqual(first_transition["transition_result"], "not_evaluated")
        self.assertEqual(first_transition["reason"], "no_previous_frame_in_evaluation")
        self.assertEqual(
            first_transition["observations"],
            [{"reference_point": {"x": 0.85, "y": 0.7}, "inside_zone": False}],
        )
        entry_transition = result["frame_results"][1]["entry_transition"]
        self.assertIsNotNone(entry_transition)
        self.assertTrue(entry_transition["transition_evaluated"])
        self.assertEqual(entry_transition["transition_result"], "entry_candidate")
        self.assertEqual(entry_transition["reason"], "observed_outside_to_inside")

    def test_outside_then_outside_is_not_an_entry(self) -> None:
        result = self.evaluate_detections(
            {
                1: [[80, 40, 90, 70]],
                2: [[80, 40, 90, 70]],
            },
            frame_count=2,
        )

        self.assertFalse(result["condition_met"])
        self.assertEqual(result["entry_evidence"], [])
        transition = result["frame_results"][1]["entry_transition"]
        self.assertTrue(transition["transition_evaluated"])
        self.assertEqual(transition["transition_result"], "no_entry")
        self.assertEqual(transition["reason"], "observed_outside_to_outside")

    def test_inside_then_inside_does_not_repeat_entry(self) -> None:
        result = self.evaluate_detections(
            {
                1: [[40, 40, 60, 70]],
                2: [[45, 40, 65, 70]],
            },
            frame_count=2,
        )

        self.assertFalse(result["condition_met"])
        self.assertEqual(result["entry_evidence"], [])
        transition = result["frame_results"][1]["entry_transition"]
        self.assertTrue(transition["transition_evaluated"])
        self.assertEqual(transition["reason"], "observed_inside_to_inside")

    def test_inside_then_outside_is_not_an_entry(self) -> None:
        result = self.evaluate_detections(
            {
                1: [[40, 40, 60, 70]],
                2: [[80, 40, 90, 70]],
            },
            frame_count=2,
        )

        self.assertFalse(result["condition_met"])
        self.assertEqual(result["entry_evidence"], [])
        transition = result["frame_results"][1]["entry_transition"]
        self.assertTrue(transition["transition_evaluated"])
        self.assertEqual(
            transition["reason"],
            "observed_inside_to_outside",
        )

    def test_missing_detection_breaks_entry_comparison(self) -> None:
        result = self.evaluate_detections(
            {
                1: [[80, 40, 90, 70]],
                3: [[40, 40, 60, 70]],
            },
            frame_count=3,
        )

        self.assertFalse(result["condition_met"])
        self.assertEqual(result["entry_evidence"], [])
        missing_transition = result["frame_results"][1]["entry_transition"]
        self.assertEqual(missing_transition["detection_count"], 0)
        self.assertFalse(missing_transition["transition_evaluated"])
        self.assertEqual(
            missing_transition["reason"],
            "no_person_detected",
        )
        following_transition = result["frame_results"][2]["entry_transition"]
        self.assertFalse(following_transition["transition_evaluated"])
        self.assertEqual(
            following_transition["reason"],
            "previous_frame_had_no_person_detection",
        )

    def test_entry_diagnostics_follow_chronological_frames_and_configured_zone(self) -> None:
        result = self.evaluate_detections(
            {1: [[80, 40, 90, 70]], 2: [[80, 40, 90, 70]]},
            frame_count=2,
        )

        self.assertEqual(result["condition"], SUPPORTED_ZONE_ENTRY_CONDITION)
        self.assertEqual(result["zone"], [point.model_dump() for point in self.zone])
        self.assertEqual(
            [frame["frame"] for frame in result["frame_results"]],
            ["frame_00001.jpg", "frame_00002.jpg"],
        )
        self.assertEqual(
            [frame["timestamp_seconds"] for frame in result["frame_results"]],
            [0, 2],
        )
        self.assertTrue(
            all(
                frame["entry_transition"]["observations"][0]["inside_zone"] is False
                for frame in result["frame_results"]
            )
        )

    def test_multiple_detections_do_not_assume_person_identity(self) -> None:
        result = self.evaluate_detections(
            {
                1: [[80, 40, 90, 70]],
                2: [
                    [80, 40, 90, 70],
                    [40, 40, 60, 70],
                ],
                3: [[40, 40, 60, 70]],
            },
            frame_count=3,
        )

        self.assertFalse(result["condition_met"])
        self.assertEqual(result["entry_evidence"], [])
        ambiguous_transition = result["frame_results"][1]["entry_transition"]
        self.assertEqual(ambiguous_transition["detection_count"], 2)
        self.assertFalse(ambiguous_transition["transition_evaluated"])
        self.assertEqual(
            ambiguous_transition["reason"],
            "multiple_person_detections",
        )
        following_transition = result["frame_results"][2]["entry_transition"]
        self.assertFalse(following_transition["transition_evaluated"])
        self.assertEqual(
            following_transition["reason"],
            "previous_frame_had_multiple_person_detections",
        )

    def test_exit_then_reentry_emits_a_second_candidate(self) -> None:
        result = self.evaluate_detections(
            {
                1: [[80, 40, 90, 70]],
                2: [[40, 40, 60, 70]],
                3: [[45, 40, 65, 70]],
                4: [[80, 40, 90, 70]],
                5: [[40, 40, 60, 70]],
            },
            frame_count=5,
        )

        self.assertEqual(len(result["entry_evidence"]), 2)
        self.assertEqual(
            [entry["frame"] for entry in result["entry_evidence"]],
            ["frame_00002.jpg", "frame_00005.jpg"],
        )

    def test_spatial_rule_requires_zone_before_model_evaluation(self) -> None:
        video_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as directory:
            frames_dir = Path(directory) / video_id
            frames_dir.mkdir()
            (frames_dir / "frame_00001.jpg").touch()
            request = VideoRuleEvaluationRequest(video_id=video_id, rule=make_rule())

            with (
                patch("app.main.FRAMES_DIR", Path(directory)),
                self.assertRaises(HTTPException) as error,
            ):
                evaluate_video_rule(request)

        self.assertEqual(error.exception.status_code, 400)

    def test_invalid_zone_polygon_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            VideoRuleEvaluationRequest(
                video_id=str(uuid.uuid4()),
                rule=make_rule(SUPPORTED_ZONE_ENTRY_CONDITION),
                zone=self.zone[:2],
            )

    def test_natural_language_spatial_condition_is_not_reinterpreted(self) -> None:
        video_id = str(uuid.uuid4())
        unsupported_rule = make_rule().model_copy(
            update={"condition": "A person enters the restricted area."}
        )
        request = VideoRuleEvaluationRequest(
            video_id=video_id,
            rule=unsupported_rule,
            zone=self.zone,
        )
        model = FakeModel()

        with tempfile.TemporaryDirectory() as directory:
            frames_dir = Path(directory) / video_id
            frames_dir.mkdir()
            (frames_dir / "frame_00001.jpg").touch()
            with (
                patch("app.main.FRAMES_DIR", Path(directory)),
                patch("app.main.YOLO_MODEL", model),
                self.assertRaises(HTTPException) as error,
            ):
                evaluate_video_rule(request)

        self.assertEqual(error.exception.status_code, 400)
        self.assertIn("Unsupported evaluation condition", error.exception.detail)
        self.assertEqual(model.calls, [])

    def test_detector_capabilities_are_discovered_from_model_metadata(self) -> None:
        model = FakeModel(names={0: "person", 1: "forklift"})
        with patch("app.main.YOLO_MODEL", model):
            capabilities = get_capabilities()

        self.assertEqual(
            {target["class_name"] for target in capabilities["targets"]},
            {"person", "forklift"},
        )
        self.assertIn("object_in_zone", {
            item["condition"] for item in capabilities["conditions"]
        })
        object_capability = next(
            item
            for item in capabilities["capabilities"]
            if item["capability_type"] == "object_detection"
        )
        self.assertEqual(
            object_capability["supported_object_classes"],
            ["forklift", "person"],
        )
        self.assertEqual(object_capability["validation_status"], "unvalidated")
        capability_types = {
            item["capability_type"] for item in capabilities["capabilities"]
        }
        self.assertIn("object_detection", capability_types)
        self.assertIn("spatial_analysis", capability_types)

    def test_live_frame_inference_uses_in_memory_image_and_does_not_store_it(self) -> None:
        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("app.main.FRAMES_DIR", Path(directory)),
                patch("app.main.YOLO_MODEL", model),
            ):
                response = self.post_live_frame(
                    rule=make_rule("object_in_zone"),
                    zone=self.zone,
                )
            files_after_request = list(Path(directory).iterdir())

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["condition_met"])
        self.assertEqual(result["condition"], "object_in_zone")
        self.assertEqual(result["target_class"], "person")
        self.assertEqual(result["timestamp_seconds"], 1.25)
        self.assertEqual(result["frame_width"], 100)
        self.assertEqual(result["detection_count"], 1)
        self.assertEqual(result["matches"][0]["anchor_point"], {"x": 0.5, "y": 0.7})
        self.assertEqual(result["evaluation_status"], "needs_human_review")
        self.assertTrue(result["human_approval_required"])
        self.assertFalse(result["evidence"]["frame_retained"])
        self.assertEqual(files_after_request, [])
        self.assertEqual(len(model.calls), 1)
        self.assertIsInstance(model.calls[0], np.ndarray)

    def test_live_frame_overlap_condition_uses_existing_polygon_metric(self) -> None:
        with patch("app.main.YOLO_MODEL", FakeModel()):
            response = self.post_live_frame(
                rule=make_rule("person_overlaps_zone", 0.5),
                zone=self.zone,
            )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["condition_met"])
        self.assertGreaterEqual(
            result["matches"][0]["zone_overlap_ratio"],
            result["zone_overlap_threshold"],
        )

    def test_live_frame_presence_condition_does_not_apply_unrequested_zone(self) -> None:
        rule = make_rule("present").model_copy(
            update={
                "hazard_type": "other",
                "description": "Detect a person in this frame.",
            }
        )
        model = FakeModel(detections_by_frame={1: [[80, 40, 90, 70]]})
        with patch("app.main.YOLO_MODEL", model):
            response = self.post_live_frame(rule=rule)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["condition_met"])
        self.assertIsNone(result["zone"])
        self.assertTrue(result["matches"][0]["matches"])

    def test_live_frame_rejects_invalid_jpeg(self) -> None:
        with patch("app.main.YOLO_MODEL", FakeModel()):
            response = self.post_live_frame(
                image_bytes=b"not a jpeg",
                zone=self.zone,
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("decodable JPEG", response.json()["detail"])

    def test_live_frame_rejects_oversized_payload(self) -> None:
        with patch("app.main.YOLO_MODEL", FakeModel()):
            response = self.post_live_frame(
                image_bytes=b"x" * (5 * 1024 * 1024 + 1),
                zone=self.zone,
            )

        self.assertEqual(response.status_code, 413)
        self.assertIn("5 MB limit", response.json()["detail"])

    def test_live_frame_rejects_unreasonable_image_dimensions(self) -> None:
        encoded, buffer = cv2.imencode(
            ".jpg",
            np.zeros((1, 8193, 3), dtype=np.uint8),
        )
        self.assertTrue(encoded)
        with patch("app.main.YOLO_MODEL", FakeModel()):
            response = self.post_live_frame(
                image_bytes=buffer.tobytes(),
                zone=self.zone,
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("dimensions exceed", response.json()["detail"])

    def test_live_frame_rejects_invalid_confidence_threshold(self) -> None:
        rule = make_rule("object_in_zone").model_copy(
            update={"confidence_threshold": 1.5}
        )
        with patch("app.main.YOLO_MODEL", FakeModel()):
            response = self.post_live_frame(rule=rule, zone=self.zone)

        self.assertEqual(response.status_code, 422)
        self.assertIn("submitted rule is invalid", response.json()["detail"]["message"])

    def test_live_frame_rejects_unsupported_target(self) -> None:
        rule = make_rule("object_in_zone").model_copy(
            update={"detection_target": "excavator"}
        )
        with patch(
            "app.main.YOLO_MODEL",
            FakeModel(names={0: "person", 1: "forklift"}),
        ):
            response = self.post_live_frame(rule=rule, zone=self.zone)

        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertIn("validation", detail)
        target_finding = next(
            finding
            for finding in detail["validation"]["findings"]
            if finding["check"] == "detection_target"
        )
        self.assertEqual(target_finding["status"], "needs_configuration")

    def test_live_frame_rejects_entry_condition_without_temporal_session(self) -> None:
        with patch("app.main.YOLO_MODEL", FakeModel()):
            response = self.post_live_frame(
                rule=make_rule(SUPPORTED_ZONE_ENTRY_CONDITION),
                zone=self.zone,
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("adjacent observations", response.json()["detail"])

    def test_live_frame_rejects_unavailable_capability(self) -> None:
        rule = make_rule("object_in_zone").model_copy(
            update={
                "required_capability_types": ["behaviour_recognition"],
                "required_capabilities": ["missing.behaviour_recognition"],
            }
        )
        with patch("app.main.YOLO_MODEL", FakeModel()):
            response = self.post_live_frame(rule=rule, zone=self.zone)

        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        capability_finding = next(
            finding
            for finding in detail["validation"]["findings"]
            if finding["check"] == "required_capabilities"
        )
        self.assertEqual(capability_finding["status"], "needs_configuration")
        self.assertIn(
            "type:behaviour_recognition",
            capability_finding["missing_capabilities"],
        )

    def test_live_frame_rejects_invalid_zone_polygon(self) -> None:
        invalid_zone = [
            ZonePoint(x=0.1, y=0.1),
            ZonePoint(x=0.2, y=0.2),
            ZonePoint(x=0.3, y=0.3),
        ]
        with patch("app.main.YOLO_MODEL", FakeModel()):
            response = self.post_live_frame(
                rule=make_rule("object_in_zone"),
                zone=invalid_zone,
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("Invalid zone polygon", response.json()["detail"])

    def test_capability_and_condition_plugin_registration(self) -> None:
        registry = DetectorCapabilityRegistry()
        plugin = FakeRulePlugin()
        registry.register_evaluator(plugin)

        self.assertEqual(
            registry.capability_catalog()[0].capability_type,
            "attribute_classification",
        )
        self.assertEqual(
            registry.resolve_evaluator("object_has_configured_attribute")[0],
            plugin,
        )
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register_evaluator(plugin)

    def test_existing_spatial_rule_requirements_are_executable(self) -> None:
        rule = make_rule().model_copy(
            update={
                "required_capability_types": [
                    "object_detection",
                    "spatial_analysis",
                ],
            }
        )
        with patch("app.main.YOLO_MODEL", FakeModel()):
            assessment = validate_rule_draft(rule, zone_configured=True)

        self.assertEqual(assessment["status"], "ready_for_evaluation")
        capability_finding = next(
            item
            for item in assessment["findings"]
            if item["check"] == "required_capabilities"
        )
        self.assertEqual(capability_finding["status"], "passed")
        self.assertEqual(capability_finding["capability_issues"], [])

    def test_globally_registered_capability_must_be_executable_for_rule(self) -> None:
        registry = DetectorCapabilityRegistry()
        registry.register(YoloDetectorAdapter(lambda: FakeModel()))
        plugin = FakeRulePlugin()
        registry.register_evaluator(plugin)
        rule = make_rule().model_copy(
            update={
                "condition": "object_in_zone",
                "required_capability_types": ["attribute_classification"],
                "required_capabilities": [
                    "visual-attribute-plugin.attribute_classification"
                ],
            }
        )

        with patch("app.main.DETECTOR_REGISTRY", registry):
            assessment = validate_rule_draft(rule, zone_configured=True)
            capability_finding = next(
                item
                for item in assessment["findings"]
                if item["check"] == "required_capabilities"
            )
            video_id = str(uuid.uuid4())
            request = VideoRuleEvaluationRequest(
                video_id=video_id,
                rule=rule,
                zone=self.zone,
            )
            with (
                tempfile.TemporaryDirectory() as directory,
                patch("app.main.FRAMES_DIR", Path(directory)),
                self.assertRaises(HTTPException) as error,
            ):
                evaluate_video_rule(request)

        self.assertEqual(assessment["status"], "needs_configuration")
        self.assertEqual(
            capability_finding["missing_capabilities"],
            [
                "type:attribute_classification",
                "id:visual-attribute-plugin.attribute_classification",
            ],
        )
        self.assertIn(
            "exists in the global registry but is not executable",
            capability_finding["capability_issues"][0]["reason"],
        )
        self.assertIn(
            "evaluator executes it",
            capability_finding["capability_issues"][1]["needed"],
        )
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(
            error.exception.detail["requested_capabilities"],
            capability_finding["missing_capabilities"],
        )
        self.assertTrue(error.exception.detail["capability_issues"])

    def test_missing_visual_capability_types_explain_required_implementation(self) -> None:
        missing_capabilities = {
            "attribute_classification": "classifier",
            "tracking": "tracker",
            "behaviour_recognition": "behaviour",
        }
        for capability_type, expected_implementation in missing_capabilities.items():
            with self.subTest(capability_type=capability_type):
                rule = make_rule().model_copy(
                    update={"required_capability_types": [capability_type]}
                )
                assessment = validate_rule_draft(rule, zone_configured=True)
                finding = next(
                    item
                    for item in assessment["findings"]
                    if item["check"] == "required_capabilities"
                )

                self.assertEqual(finding["status"], "needs_configuration")
                self.assertEqual(
                    finding["missing_capabilities"],
                    [f"type:{capability_type}"],
                )
                self.assertIn(
                    expected_implementation,
                    finding["capability_issues"][0]["needed"],
                )

    def test_plugin_capability_must_be_declared_by_selected_condition(self) -> None:
        class PluginWithUnwiredCapability(FakeRulePlugin):
            def capabilities(self) -> list[VisualCapability]:
                return [
                    *super().capabilities(),
                    VisualCapability(
                        capability_id=f"{self.plugin_id}.tracking",
                        detector_id=self.plugin_id,
                        capability_type="tracking",
                        description="A registered but unvalidated tracker.",
                        supported_object_classes=["person"],
                    ),
                ]

        registry = DetectorCapabilityRegistry()
        registry.register_evaluator(PluginWithUnwiredCapability())
        rule = make_rule().model_copy(
            update={
                "condition": "object_has_configured_attribute",
                "required_capability_types": ["tracking"],
                "required_capabilities": ["visual-attribute-plugin.tracking"],
            }
        )

        with patch("app.main.DETECTOR_REGISTRY", registry):
            assessment = validate_rule_draft(rule)

        finding = next(
            item
            for item in assessment["findings"]
            if item["check"] == "required_capabilities"
        )
        self.assertEqual(finding["status"], "needs_configuration")
        self.assertEqual(
            finding["missing_capabilities"],
            [
                "type:tracking",
                "id:visual-attribute-plugin.tracking",
            ],
        )
        self.assertIn(
            "not executable for target",
            finding["capability_issues"][0]["reason"],
        )

    def test_safety_rule_accepts_generic_hazard_categories(self) -> None:
        rule = make_rule().model_copy(
            update={"hazard_type": "object_obstructing_exit"}
        )

        self.assertEqual(rule.hazard_type, "object_obstructing_exit")
        hazard_schema = SafetyRule.model_json_schema()["properties"]["hazard_type"]
        self.assertNotIn("enum", hazard_schema)

    def test_plugin_evaluator_runs_and_result_requires_human_review(self) -> None:
        registry = DetectorCapabilityRegistry()
        plugin = FakeRulePlugin()
        registry.register_evaluator(plugin)
        rule = make_rule().model_copy(
            update={
                "hazard_type": "other",
                "condition": "object_has_configured_attribute",
                "condition_parameters": {"attribute": "uniform_color"},
                "required_capability_types": ["attribute_classification"],
                "required_capabilities": [
                    "visual-attribute-plugin.attribute_classification"
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            video_id = str(uuid.uuid4())
            frames_dir = Path(directory) / video_id
            frames_dir.mkdir()
            (frames_dir / "frame_00001.jpg").touch()
            request = VideoRuleEvaluationRequest(video_id=video_id, rule=rule)
            with (
                patch("app.main.FRAMES_DIR", Path(directory)),
                patch("app.main.DETECTOR_REGISTRY", registry),
            ):
                assessment = validate_rule_draft(rule)
                result = evaluate_video_rule(request)

        self.assertEqual(assessment["status"], "ready_for_evaluation")
        capability_finding = next(
            item
            for item in assessment["findings"]
            if item["check"] == "required_capabilities"
        )
        self.assertEqual(capability_finding["status"], "passed")
        self.assertEqual(capability_finding["capability_issues"], [])
        self.assertTrue(result["condition_met"])
        self.assertEqual(result["evaluation_status"], "needs_human_review")
        self.assertTrue(result["human_approval_required"])
        self.assertEqual(result["frame_results"][0]["plugin_evidence"], "test-only")

    def test_missing_attribute_capability_is_explicitly_rejected(self) -> None:
        rule = make_rule().model_copy(
            update={
                "required_capability_types": ["attribute_classification"],
                "required_capabilities": ["attribute_classifier.visual_state"],
            }
        )
        assessment = validate_rule_draft(rule, zone_configured=True)

        self.assertEqual(assessment["status"], "needs_configuration")
        missing = next(
            item
            for item in assessment["findings"]
            if item["check"] == "required_capabilities"
        )["missing_capabilities"]
        self.assertEqual(
            missing,
            [
                "type:attribute_classification",
                "id:attribute_classifier.visual_state",
            ],
        )

        video_id = str(uuid.uuid4())
        request = VideoRuleEvaluationRequest(
            video_id=video_id,
            rule=rule,
            zone=self.zone,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("app.main.FRAMES_DIR", Path(directory)),
            self.assertRaises(HTTPException) as error,
        ):
            evaluate_video_rule(request)

        self.assertEqual(error.exception.status_code, 400)
        self.assertIn(
            "type:attribute_classification",
            error.exception.detail["requested_capabilities"],
        )

    def test_supported_dynamic_target_is_validated_and_evaluated(self) -> None:
        model = FakeModel(
            detections_by_frame={1: [[40, 40, 60, 70]]},
            names={0: "person", 1: "forklift"},
            class_id=1,
        )
        rule = make_rule().model_copy(
            update={"detection_target": "forklift", "condition": "object_in_zone"}
        )
        with patch(
            "app.main.YOLO_MODEL",
            FakeModel(names={0: "person", 1: "forklift"}),
        ):
            assessment = validate_rule_draft(rule, zone_configured=True)
        self.assertEqual(assessment["status"], "ready_for_evaluation")

        video_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as directory:
            frames_dir = Path(directory) / video_id
            frames_dir.mkdir()
            (frames_dir / "frame_00001.jpg").touch()
            request = VideoRuleEvaluationRequest(
                video_id=video_id,
                rule=rule,
                zone=self.zone,
            )
            with (
                patch("app.main.FRAMES_DIR", Path(directory)),
                patch("app.main.YOLO_MODEL", model),
            ):
                result = evaluate_video_rule(request)

        self.assertTrue(result["condition_met"])
        self.assertEqual(result["target_class"], "forklift")
        self.assertEqual(result["requested_target"], "forklift")
        self.assertEqual(result["detector_id"], "yolo")
        self.assertEqual(result["frames_matching_condition"], 1)
        self.assertEqual(
            result["matching_frames"][0]["matches"][0]["class_name"],
            "forklift",
        )

    def test_unsupported_target_is_reported_without_substitution(self) -> None:
        model = FakeModel(names={0: "person", 1: "forklift"})
        rule = make_rule().model_copy(
            update={
                "detection_target": "excavator",
                "condition": "object_in_zone",
            }
        )
        with patch(
            "app.main.YOLO_MODEL",
            FakeModel(names={0: "person", 1: "forklift"}),
        ):
            assessment = validate_rule_draft(rule, zone_configured=True)
        self.assertEqual(assessment["status"], "needs_configuration")

        video_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as directory:
            frames_dir = Path(directory) / video_id
            frames_dir.mkdir()
            (frames_dir / "frame_00001.jpg").touch()
            request = VideoRuleEvaluationRequest(
                video_id=video_id,
                rule=rule,
                zone=self.zone,
            )
            with (
                patch("app.main.FRAMES_DIR", Path(directory)),
                patch("app.main.YOLO_MODEL", model),
                self.assertRaises(HTTPException) as error,
            ):
                evaluate_video_rule(request)

        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(
            error.exception.detail["missing_capability"],
            "detector_class",
        )
        self.assertEqual(error.exception.detail["requested_target"], "excavator")
        self.assertIn("forklift", error.exception.detail["available_targets"])

    def test_natural_language_rule_generation_preserves_requested_capability(self) -> None:
        model = FakeModel(names={0: "person", 1: "forklift"})
        generated_rule = make_rule().model_copy(
            update={
                "detection_target": "forklift",
                "condition": "object_overlaps_zone",
            }
        )
        client = Mock()
        generated_data = generated_rule.model_dump()
        generated_data["condition_parameters"] = [
            {"name": "overlap_threshold", "value": "0.25"}
        ]
        generated_data["condition"] = "object_overlaps_zone"
        client.models.generate_content.return_value.text = json.dumps(generated_data)
        with (
            patch("app.main.API_KEY", "configured-for-test"),
            patch("app.main.YOLO_MODEL", model),
            patch("app.main.genai.Client", return_value=client),
        ):
            response = generate_safety_rule(
                SafetyRuleRequest(instruction="Alert when a forklift enters the zone.")
            )

        self.assertEqual(response["rule"]["detection_target"], "forklift")
        self.assertEqual(response["rule"]["condition"], "object_overlaps_zone")
        self.assertEqual(
            response["rule"]["condition_parameters"]["overlap_threshold"],
            0.25,
        )
        target_finding = next(
            finding
            for finding in response["capability_assessment"]["findings"]
            if finding["check"] == "detection_target"
        )
        self.assertEqual(target_finding["status"], "passed")
        self.assertEqual(target_finding["executable_target"], "forklift")
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertIs(config.response_schema, GeneratedSafetyRule)

    def test_spatial_rule_generation_discards_unrequested_behaviour_requirement(self) -> None:
        generated_rule = make_rule().model_copy(
            update={
                "condition": "object_in_zone",
                "required_capability_types": [
                    "object_detection",
                    "spatial_analysis",
                    "behaviour_recognition",
                ],
                "required_capabilities": [
                    "rulevia.behaviour_recognition",
                ],
            }
        )
        client = Mock()
        generated_data = generated_rule.model_dump()
        generated_data["condition_parameters"] = []
        client.models.generate_content.return_value.text = json.dumps(
            generated_data
        )

        with (
            patch("app.main.API_KEY", "configured-for-test"),
            patch("app.main.YOLO_MODEL", FakeModel()),
            patch("app.main.genai.Client", return_value=client),
        ):
            response = generate_safety_rule(
                SafetyRuleRequest(
                    instruction=(
                        "Alert me when a person is inside the restricted zone."
                    )
                )
            )

        self.assertEqual(response["rule"]["condition"], "object_in_zone")
        self.assertEqual(
            set(response["rule"]["required_capability_types"]),
            {"object_detection", "spatial_analysis"},
        )
        self.assertEqual(response["rule"]["required_capabilities"], [])
        capability_finding = next(
            item
            for item in response["capability_assessment"]["findings"]
            if item["check"] == "required_capabilities"
        )
        self.assertEqual(capability_finding["status"], "passed")
        self.assertEqual(capability_finding["missing_capabilities"], [])

    def test_explicit_behaviour_request_remains_unsupported_without_recognizer(self) -> None:
        generated_rule = make_rule().model_copy(
            update={
                "condition": "object_in_zone",
                "required_capability_types": [
                    "object_detection",
                    "spatial_analysis",
                    "behaviour_recognition",
                ],
                "required_capabilities": [
                    "rulevia.behaviour_recognition",
                ],
            }
        )
        client = Mock()
        generated_data = generated_rule.model_dump()
        generated_data["condition_parameters"] = []
        client.models.generate_content.return_value.text = json.dumps(
            generated_data
        )

        with (
            patch("app.main.API_KEY", "configured-for-test"),
            patch("app.main.YOLO_MODEL", FakeModel()),
            patch("app.main.genai.Client", return_value=client),
        ):
            response = generate_safety_rule(
                SafetyRuleRequest(
                    instruction=(
                        "Alert me when a person is running inside the restricted zone."
                    )
                )
            )

        self.assertIn(
            "behaviour_recognition",
            response["rule"]["required_capability_types"],
        )
        self.assertIn(
            "rulevia.behaviour_recognition",
            response["rule"]["required_capabilities"],
        )
        capability_finding = next(
            item
            for item in response["capability_assessment"]["findings"]
            if item["check"] == "required_capabilities"
        )
        self.assertEqual(capability_finding["status"], "needs_configuration")
        self.assertIn(
            "type:behaviour_recognition",
            capability_finding["missing_capabilities"],
        )
        self.assertIn(
            "behaviour recognizer",
            capability_finding["message"],
        )
        self.assertEqual(
            next(
                item
                for item in response["capability_assessment"]["findings"]
                if item["check"] == "condition"
            )["status"],
            "passed",
        )

    def test_generated_rule_schema_uses_named_parameters_not_free_form_object(self) -> None:
        parameter_schema = GeneratedSafetyRule.model_json_schema()["properties"][
            "condition_parameters"
        ]

        self.assertEqual(parameter_schema["type"], "array")
        self.assertIn("items", parameter_schema)
        self.assertNotIn("additionalProperties", parameter_schema)
        schema = GeneratedSafetyRule.model_json_schema()
        self.assertNotIn("exclusiveMinimum", json.dumps(schema))

    def test_generation_schema_errors_return_cors_visible_preset_message(self) -> None:
        client = Mock()
        client.models.generate_content.side_effect = ValueError(
            "unsupported structured response schema"
        )

        with (
            patch("app.main.API_KEY", "configured-for-test"),
            patch("app.main.genai.Client", return_value=client),
        ):
            response = TestClient(app).post(
                "/rules/generate",
                headers={"Origin": "http://localhost:5173"},
                json={
                    "instruction": "Create a draft rule when a person enters a zone."
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://localhost:5173",
        )
        self.assertIn("structured rule request", response.json()["detail"])
        self.assertIn("presets remain available", response.json()["detail"])
        self.assertNotIn(
            "unsupported structured response schema",
            response.json()["detail"],
        )

    def test_legacy_person_condition_rejects_non_person_target(self) -> None:
        rule = make_rule().model_copy(update={"detection_target": "forklift"})

        assessment = validate_rule_draft(rule, zone_configured=True)

        self.assertEqual(assessment["status"], "invalid")
        self.assertTrue(
            any(finding["check"] == "condition_target" for finding in assessment["findings"])
        )


if __name__ == "__main__":
    unittest.main()
