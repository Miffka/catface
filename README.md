# catface

Upload a cat photo → 48 facial landmarks, a predicted expression class, and two
geometry readouts (eye aperture, ear angle). Pick a target expression and a
slider, get the image warped toward it.

Not a vet tool. No pain or welfare claims. Ears are predicted but not editable
in v1 — a 2D warp can't do out-of-plane rotation.

## Setup

```
uv sync                                  # app: runtime + dev deps, no torch
uv sync --group train --group dev        # research track too (torch CPU, sklearn, pandas)
uv run pytest                            # app-track tests
uv run python manage.py migrate          # database migrations
uv run python -m catface.ml.<script>     # research scripts (E0–E5, export)
```

## Layout

```
src/catface/
  core/     # shared: geometry, graph/adjacency, schemas, states, warp
  api/      # fastapi, routes, db, inference engines
  ml/       # training only
models/     # gitignored except manifest.json
scripts/    # fetch_weights, export
web/        # plain html/js, no build step
tests/  docs/  ops/  alembic/  .github/workflows/
```

`api` never imports `ml`. Both import `core`.

## Docs

- [AGENTS.md](AGENTS.md) — rules, commands, how the two tracks are run
- [docs/PLAN_PROJECT.md](docs/PLAN_PROJECT.md) — the app: API, storage, warper, deploy
- [docs/PLAN_RESEARCH.md](docs/PLAN_RESEARCH.md) — the model: experiments E0–E5, export
- [docs/backlog.md](docs/backlog.md) — groomed work, tagged by track
- [docs/DECISIONS.md](docs/DECISIONS.md), [docs/AI_WORKFLOW.md](docs/AI_WORKFLOW.md) — running logs
