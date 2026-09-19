# PLAN_PROJECT.md

Working notes for the app half of cat-face-studio. Decisions, not a spec. Details
get filled in when I get there.

Companion: PLAN_RESEARCH.md. That produces `models/expression_head.onnx`. This
consumes it. Nothing here waits on it.

## What it is

Upload a cat photo → 48 landmarks + a predicted expression class + two derived
geometry readouts (eye aperture, ear angle) → pick a target expression and a slider,
get a warped image with the expression shifted.

Not a vet tool. No pain claims. Ears are predicted but not editable in v1, since a
2D warp can't do out-of-plane rotation.

## Course requirements

Needed: frontend, backend, API contract, persistent storage, tests,
containerization, CI/CD, public deployment, docs (spec, architecture, setup, tests,
deployment), a written account of how AI tools were used, plus observability,
an alert, a read-only agent incident responder, agent skills, and security work.

**Read the rubric file in the course repo and list the scored items with their
weights before writing code.** Order the work by weight, not by what's fun.

Note: the course expects Postgres. Plan is SQLite locally, Postgres in the prod
compose file, switched by `DATABASE_URL`. CI runs the tests against both.

## Layout

```
src/catface/
  core/     # shared: geometry, graph/adjacency, schemas, states, warp
  api/      # fastapi, routes, db, inference engines
  ml/       # training only. see PLAN_RESEARCH.md
models/     # gitignored except manifest.json
scripts/    # fetch_weights, export
web/        # plain html/js, no build step
tests/  docs/  ops/  alembic/  .github/workflows/
```

Rule: `api` never imports `ml`. Write a test that greps for it.

`core` is the contract. Letterboxing, crop margin, Procrustes all live there and get
called by both training and serving. Duplicating them is how you get a model that
scores well in the notebook and badly in prod.

## Dependencies (uv)

Runtime deps in `[project]`: fastapi, uvicorn, pydantic, sqlalchemy, alembic, numpy,
opencv-python-headless, openvino, pillow.

Groups: `dev` (pytest, ruff, mypy, httpx), `train` (torch, sklearn, pandas,
matplotlib), `convert` (tensorflow-cpu, tf2onnx). Extras: `postgres`, `otel`.

`uv sync` gives a reviewer the app with no torch download. Prod image builds with
`--frozen --no-dev`.

## API

`/api/v1/` from day one.

- `POST /predict` — multipart image → upload_id, face_box, 48 landmarks, class
  probabilities, derived geometry (eye_aperture, ear_angle), model_version, latency
- `POST /edit` — upload_id, target_class, intensity 0..1 → warped image URL and the
  source/target landmarks
- `GET /uploads`, `GET /uploads/{id}`, `GET /models`
- `/healthz`, `/readyz`, `/metrics`

Model returns class probabilities. `core/states.py` picks the display label and
applies the low-confidence and extreme-pose caveats. The model never emits a
user-facing string.

Derived geometry is computed in `core`, not predicted: eye aperture from the eyelid
contour points, ear angle from base-to-tip against the inter-ocular axis. These need
no model and work even when the classifier is missing.

Snapshot `/openapi.json` in tests so contract changes show up in the diff.

## Storage

Tables: `uploads` (sha256 unique, storage key, dims), `predictions` (landmarks JSON,
three scores, label, model_version, latency), `edits` (target state, intensity,
target landmarks, storage key).

Portability rules so the SQLite→Postgres swap is real:
- `sqlalchemy.types.JSON`, not JSONB
- uuid as `String(36)`, not the PG type
- `DateTime(timezone=True)`, always tz-aware
- no PG-only SQL
- Alembic from the first migration

SQLite under concurrent requests needs WAL mode and a `busy_timeout` of a few
seconds, set as pragmas on connect. It still allows only one writer at a time, which
is fine at this scale but is the main reason prod runs Postgres. Keep write
transactions short: never hold one open across an inference call.

## Inference

`InferenceEngine` protocol with two impls: `StubEngine` (deterministic fake, used in
tests and until M2) and `OpenVINOEngine`.

Pipeline: downscale to max 1024px → letterbox 224 → localiser → box in image space →
expand 0.1, crop, resize 384 → landmark model → map back → Procrustes →
expression head.

Inputs are float32 in [0,1]. The graphs rescale internally, don't do it twice.

OpenVINO reads `.tflite` directly, so no ONNX conversion for the vendored models.
Verify this in M2. If a custom op blocks it, swap in ONNX Runtime behind the same
protocol.

Load and compile models once at startup. Use the 11 MB landmark model, not the 55 MB
one.

**Concurrency.** Inference is CPU-bound, so it must never run inside an async
coroutine or it blocks the event loop and every other request stalls behind it. Two
ways to do this and either is fine: declare the route with `def` instead of
`async def` so FastAPI runs it in the threadpool, or keep the route async and hand
the work to an executor. Pick one, write down which.

Concurrency is bounded, not unbounded:
- one compiled model, a small pool of InferRequests sized to the vCPU count
- a semaphore capping concurrent inferences at 2 on a 2-vCPU box
- a bounded wait queue on top of that. When the queue is full, return 503 with a
  `Retry-After` header rather than accepting work the box can't do. A fast refusal
  beats a slow timeout for everyone already in the queue.
- per-request timeout so a pathological image can't hold a slot forever

OpenVINO settings: 2 streams × 1 thread, or 1 stream × 2 threads. Two streams
generally gives better throughput with concurrent users, one stream gives better
single-request latency. Measure both under the load test and keep the numbers.

**Cache by content hash.** `uploads.sha256` is already unique. If the same image
comes in again, return the stored prediction instead of re-running the pipeline.
Under peer review several people will upload the same sample photos, so this is
worth more than it looks.

## Load and resources

Target host: t3.micro, 2 vCPU, 1 GiB RAM, x86. Assume around 5 concurrent reviewers
poking the endpoint, not real traffic.

**Memory budget, roughly.** OS and Docker ~150 MB, API process with Python, numpy,
opencv and the OpenVINO runtime ~300 MB, compiled models and activation buffers
~150 MB, Postgres tuned small ~100 MB. That lands near 700 MB of 1024 MB. It fits,
with little headroom.

Consequences:
- **2 GB swap file, mandatory.** Without it a single traffic spike OOM-kills the
  container.
- **One uvicorn worker.** Each worker loads its own copy of the models, so two
  workers roughly doubles model memory for no throughput gain on 2 vCPU. Scale with
  the InferRequest pool inside one process instead.
- **Container memory limits and `restart: unless-stopped`** in the prod compose, so
  a leak restarts the API instead of taking down Postgres with it.
- **Postgres tuned down**: `shared_buffers=32MB`, `max_connections=20`, and a
  SQLAlchemy pool of about 5. Default Postgres settings assume a much bigger box.

**Image size is the real memory risk.** A 4000×3000 JPEG decodes to ~36 MB of RGB
array, and three of those at once is most of the headroom. So:
- 5 MB upload cap, enforced while streaming to disk, not after reading into memory
- reject images above a max dimension outright
- downscale to max 1024px as the first pipeline step, before anything else touches it
- cap the warp output resolution too

**Disk.** Free tier covers 30 GB of EBS. Uploads and warped outputs accumulate and a
full disk takes the app down harder than a memory spike. Add a retention job that
deletes stored images older than N days, and an alert on disk usage above 80%.

**Cost traps on the free tier.** The 750 hours are shared across instances, so never
leave a second one running. Public IPv4 costs about $3.60/month. Egress is free up to
100 GB/month, which serving warped images will not approach.

**Standing caveat, already in M6:** accounts created after July 2025 are on the
credit-based plan, which closes the account when the credits run out. Everything
above assumes t3.micro sizing regardless of provider, so the same numbers apply to a
similarly sized VPS.

**Load test.** Add `bench/load.py` (locust or k6): N concurrent users uploading a
mix of small and large fixtures. Record p50, p95, error rate, 503 rate, and peak RSS
at 1, 5 and 20 concurrent users. Run it before and after tuning the stream settings
and put the table in the README. This doubles as the evidence that the sizing
decisions above were measured rather than guessed.

Acceptance: at 5 concurrent users, no 5xx other than deliberate 503 backpressure, no
OOM kill, p95 under 3 seconds.

## Model artifacts

No weights in git. `models/manifest.json` (committed) lists name, file, sha256,
source, licence, input shape, metrics. `scripts/fetch_weights.py` downloads and
verifies. Runs in the Docker build and in `make setup`.

Licence: the vendored weights and CatFLW are both CC BY-NC 4.0. Don't redistribute
them, say so in the README, cite the papers, and state separately that my own code
is MIT.

## Tests

Unit on `core`: letterbox round-trips to identity, Procrustes aligns a
rotated copy, adjacency is symmetric, identity warp returns the same image.

Integration with `StubEngine` + in-memory db: every endpoint, bad uploads, oversized
uploads, intensity=0 returns the source landmarks, full predict→edit→list round trip.
Run the suite on SQLite and on Postgres.

E2E marked slow: real weights, three fixture photos. Assert landmark count and that
points sit inside the image. Don't assert coordinates, they change when I retrain.

Concurrency tests: 10 simultaneous `/predict` calls all return 200 or a clean 503,
never a 500 and never a hang. Oversized upload is rejected before the body is fully
read. Two concurrent writes of the same image hash don't violate the unique
constraint — the second should get the cached row.

## Docker / CI

Multi-stage Dockerfile, uv in the builder, non-root, `alembic upgrade head` on start.
Two compose files: dev (sqlite, reload) and prod (postgres + otel stack).

CI: ruff, mypy, pytest on both backends, build image and curl `/healthz`, dependency
audit, image scan, secret scan.

Deploy job on main with `needs: [ci]` so it literally cannot run before tests pass.
Poll `/readyz`, roll back on failure.

## Observability

OTel FastAPI + SQLAlchemy instrumentation → collector → Prometheus/Loki/Tempo/Grafana.
Custom metrics: inference latency by stage, predictions by outcome, no-face count,
inference queue depth, 503 backpressure count, cache hit rate, process RSS.

One committed Grafana dashboard. Alerts: no-face rate above 50% for 10 minutes
(fires when the model breaks and when users upload junk), plus queue saturation and
disk above 80%. Write a runbook line for each explaining the threshold.

`/readyz` must stay cheap. Never run inference in a health check, or the load
balancer will mark the box unhealthy precisely when it's busy.

## Agents

Incident responder: read-only, logs and metrics tools only, no write or deploy
access. Document the allowlist, break the model path on purpose, save the transcript.

Skills as SKILL.md: `add-endpoint` (route + schema + test + snapshot),
`export-model`, `review-checklist`.

## Security

Content-type allowlist plus magic-byte check, 5 MB cap enforced during streaming,
max dimension cap, Pillow decode-bomb limits. Generated uuid filenames on disk. Rate
limit predict and edit per IP — on a 1 GiB box this is a resource control as much as
a security one. CORS to the deploy origin. Non-root container. Secrets from env,
`.env.example` has placeholders. Scanning in CI.

## Milestones

Each ends green and tagged.

- **M0** skeleton: uv init, layout, lint + one test in CI, Dockerfile builds
- **M1** contract first: all schemas and routes against `StubEngine`, db + first
  migration, integration tests, OpenAPI snapshot. Whole API works with a fake model.
- **M2** real landmarks: fetch_weights, manifest, OpenVINOEngine, tflite load
  verified, E2E on fixtures
- **M3** warper: Delaunay with boundary anchors (image corners + ring around the face
  box, or the warp tears at the edges), per-triangle affine, wire up `/edit`
- **M4** frontend: upload, canvas overlay, state readout, target + slider,
  before/after, history
- **M5** expression head from the research side. If it's late, the class picker in
  `/edit` becomes manual and everything else still works.
- **M6** deploy: prod compose with Postgres, deploy workflow, public URL.
  Prefer a fixed-price VPS over AWS free tier — new AWS accounts run on credits that
  expire and then close the account, which kills the demo before review. If AWS,
  x86 not Graviton, since OpenVINO targets Intel.
- **M6.5** load: swap file, memory limits, tuned Postgres, bounded semaphore and
  queue, `bench/load.py`. Tune streams, record the table, fix whatever the test
  breaks. Do this before observability, since the load test tells you which metrics
  are worth having.
- **M7** observability + agents
- **M8** docs, screenshots, self-score against the rubric, fix the worst gaps

## Fallbacks

| Risk | Fallback |
|---|---|
| tflite won't load in OpenVINO | ONNX Runtime, same protocol |
| classifier late or bad | ship the landmark viewer, the derived geometry readouts, and the warper with a manual class picker. All three work without it. |
| ear warp smears | read-only channel, documented |
| Postgres too heavy for the host | SQLite in prod with WAL, Postgres still proven in CI, tradeoff documented |
| no time for full observability | metrics + one alert, note what was cut |
| OOM under load even with swap | drop the otel stack off the app host, ship metrics to a free hosted Grafana instead |
| latency unacceptable at 5 users | pre-resize harder (768px), or accept the upload and process it as a background job with a polling status endpoint |

## Docs to keep as I go

`docs/AI_WORKFLOW.md` is graded, so write it daily, not at the end: which tool for
what, how I gave it context, two or three cases where I rejected generated code and
why, what tests caught what. A caught mistake reads better than a clean transcript.

`docs/DECISIONS.md`: one dated paragraph per decision. Cheap daily, impossible to
reconstruct later.
