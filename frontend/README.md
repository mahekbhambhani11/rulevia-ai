# Rulevia AI

Rulevia AI is an industrial safety intelligence workspace for drafting safety
rules, evaluating recorded footage, and reviewing visual evidence.

## Run locally

Start the FastAPI backend from the repository's backend environment:

```sh
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

Start the Vite frontend in a second terminal:

```sh
cd frontend
npm run dev
```

Rule generation also requires `GEMINI_API_KEY` in `backend/.env`. Video
processing uses the local YOLO model at `backend/yolo11n.pt`.

The frontend API URL defaults to `http://127.0.0.1:8000`. For a different
backend host, create `frontend/.env` from `.env.example` and set
`VITE_API_BASE_URL` to the browser-reachable API URL. Add the frontend origin to
`CORS_ALLOWED_ORIGINS` in `backend/.env` as a comma-separated list, then restart
both dev servers. Keep Gemini credentials only in the backend environment.
If Gemini is unavailable, Rule Builder's deterministic restricted-area and
bounding-box-overlap presets do not require generation.

## Live webcam monitoring

Open **Rule Builder** and select a rule supported by the current detector, then
configure a zone in **Video Library** if the rule is spatial. Open **Live
Monitoring** and choose **Start Monitoring**; the browser requests camera
permission only after that action. Use localhost or HTTPS for browser camera
access. Frames are resized and sent as JPEGs to `POST /rules/evaluate-frame`
one at a time, at approximately 1.5-second intervals. The backend evaluates
each frame in memory and does not save it. Use **Stop Monitoring** to abort an
in-flight request and release the webcam.

Live mode supports per-frame presence and spatial-zone/overlap evaluation.
Entry-transition rules are not supported because they need session-level
temporal observations. Webcam candidates are transient and are not written to
the Review Queue or Audit History; review persistence remains available for
recorded-video evidence.

## Restricted-zone evaluation

1. Create a safety rule in **Rule Builder** and continue to **Video Library**.
2. Upload footage to extract timestamped frames.
3. In **Restricted-zone configuration**, choose a representative frame and
   draw a polygon by clicking its vertices. Finish the zone with at least three
   points; use **Redraw zone** to replace it.
4. Evaluate the selected rule. For a spatial rule, a detection matches the
   zone when the bottom-center of its bounding box falls inside the polygon.
5. Inspect the matching frame, timestamp, confidence, zone outline, and
   detection box, then submit a human review decision if appropriate.

The polygon is stored in normalized frame coordinates in the current workspace
session and is sent with evaluation requests. Evaluation uses frames sampled
every two seconds, so it can miss brief events and does not establish a
continuous boundary crossing. Results require human review and are not
independent confirmation of a safety violation.

## Detector capabilities

Rule Builder loads executable targets and conditions from the backend's
`GET /capabilities` endpoint. Each capability contract describes its type,
supported classes and visual concepts, required inputs/configuration,
limitations, and validation status. Object detection, attribute/state
classification, tracking, spatial analysis, and behaviour recognition are
separate capability types. YOLO currently registers only object detection;
the core separately advertises its polygon analysis. Neither implies that
unregistered attributes, states, tracking, relationships, or behaviours can be
detected.

To add a detector, implement `DetectorAdapter` in a backend module, expose its
actual available classes and `VisualCapability` descriptors, and implement
`predict(...)` using the existing normalized `DetectorFrame` contract. Register
it with `DETECTOR_REGISTRY.register(...)` during backend setup. To add a new
condition or capability that needs custom logic, separately implement
`RuleEvaluationPlugin`: return its capability descriptors and
`RuleConditionCapability` entries, and implement
`evaluate_video(request, frame_paths)` to return at least a boolean
`condition_met` plus evidence. Register it with
`DETECTOR_REGISTRY.register_evaluator(...)`. The core validates advertised
capability IDs/types, rejects unavailable requirements, and enforces the
standard human-review response. Keep `validation_status` honest: the current
YOLO and built-in spatial entries are explicitly marked `unvalidated`, not
production-ready.

The built-in `object_*_zone` conditions use detected-object bottom-center
points, sampled-frame transitions, or polygon/bounding-box overlap as their
respective condition defines. Legacy `person_*_zone` conditions retain their
person-only semantics. Natural-language rules are structured requests, not
visual capabilities: validation reports missing capability types or IDs, and
video evaluation never substitutes a different target or capability. Every
result continues to require human review.

## Validation

Frontend lint and type-check:

```sh
npm run lint
./node_modules/.bin/tsc --noEmit -p tsconfig.app.json --incremental false --tsBuildInfoFile /dev/null
```

Focused backend spatial tests:

```sh
cd backend
.venv/bin/python -m unittest discover -s tests -v
```
