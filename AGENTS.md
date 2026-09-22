Commands

uv sync - install runtime + dev deps
uv sync --group train --group dev - add the research track's deps (torch, sklearn, pandas)
uv run pytest - the whole app-track suite
uv run pytest tests/test_smoke.py - one test file
uv run python manage.py migrate - apply database migrations
uv run python -m catface.ml.<script> - research track scripts (E0–E5, export)

Rules

- Dependencies are added in pyproject.toml. Do not add one without
  asking.
- `src/catface/api` never imports `src/catface/ml`. This is
  grep-tested; don't work around the test, fix the import.
- `src/catface/core` is shared by both tracks. Neither track
  reimplements letterboxing, crop margin, or Procrustes locally — if
  you need it, import it from `core`.
- Two tracks, two processes. Figure out which one an issue belongs to before picking a process file: PLAN_RESEARCH.md scope (E0–E5, model questions Q1–Q5, `models/expression_head.onnx`) uses `docs/research_process.md`; everything else (API, storage, warper, frontend, deploy, ops) uses `docs/project_process.md`.

Tracks

- **Research** — `docs/plan-research.md` (PLAN_RESEARCH.md), process in `docs/research_process.md`
- **Project** — `docs/plan-project.md` (PLAN_PROJECT.md), process in `docs/project_process.md`

Documents

- `docs/backlog.md` — the backlog, tagged by track
- `docs/MODEL_REPORT.md` — research track's running report, written by
  Research PM as experiments close
- `docs/AI_WORKFLOW.md`, `docs/DECISIONS.md` — graded, both tracks
  write to these as they go, not at the end

Orchestrator

The main session is the orchestrator for whichever track it's currently running. It never grooms, implements, tests, or reviews itself — that's what the subagents in each track's role docs are for. It does not mix the two tracks' lifecycles in one loop: pick a track, run that track's process file's lifecycle to completion or to its next natural stopping point, then switch.

Cross-track dependency

There are two handoffs between tracks. `models/class_means.json` (per-class Procrustes mean shapes) shipped at E1, ahead of the model — the app's warper (PLAN_PROJECT.md M3) depends on it, not on the trained classifier, which is what makes M5's fallback (manual class picker) actually work. The second is `models/expression_head.onnx` + its manifest entry, gated by the research track's Export Verifier (`docs/team/export-verifier.md`). The project track does not wait on the second one to make progress — that's exactly what M5's fallback is for. If the project track picks up work that consumes the real model, check the manifest and the Export Verifier's PASS exist first; if they don't, that work isn't groomable yet and goes back to the backlog.
