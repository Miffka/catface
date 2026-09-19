# catface

Upload a cat photo and get 48 facial landmarks, an expression class, and two geometry readouts (eye aperture, ear angle). Pick a target expression, drag a slider, and the app warps the image toward it. Not a vet tool; it makes no pain or welfare claims.

## Run

```
uv sync                                  # app: runtime + dev deps
uv sync --group train --group dev        # add the research track (torch CPU, sklearn, pandas)
uv run pytest                            # app tests
uv run python manage.py migrate          # database migrations
uv run python -m catface.ml.<script>     # research scripts
```

## Docs

- [AGENTS.md](AGENTS.md): rules and how the two tracks run
- [docs/PLAN_PROJECT.md](docs/PLAN_PROJECT.md): the app
- [docs/PLAN_RESEARCH.md](docs/PLAN_RESEARCH.md): the model
- [docs/backlog.md](docs/backlog.md): work queue
