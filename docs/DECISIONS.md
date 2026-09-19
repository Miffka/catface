# DECISIONS.md

Decisions and the reasons for them, written as they're made. Not a design
doc — one entry per decision: what was decided, why, what the alternative
was. Newest on top.

---

## 2026-09-19 — `models/manifest.json` holds the detector pair now; RSCH-6 appends `expression_head`, doesn't replace the file

**Decided:** `models/manifest.json` is committed today with two entries, `cat_face_localizer` and `cat_face_landmarks` (the CatFLW-trained HF detector pair `fetch_weights.py` downloads), in the exact shape RSCH-6 already specifies for its own entry: name, file, sha256, source, licence, input shape, metrics.

**Why:** the detector pair has to exist before E0's "run the detector once over both Roboflow sets" step can run, and PLAN_PROJECT's inference path needs the same files. RSCH-6 was always going to write to this path in this shape, so starting the file now instead of waiting for RSCH-6 avoids a rename or a second manifest file later.

**Alternative considered:** a separate `models/detector_manifest.json` for the app-track weights, keeping `manifest.json` for RSCH-6 alone. Rejected — one manifest is one place to check sha256s from, and the entry shape already matches.

**How to apply:** RSCH-6 adds a third top-level key, `expression_head`, to this same file. It doesn't touch the two keys above.

---

## 2026-09-19 — `.env` loading uses `python-dotenv`, not a hand-rolled parser

**Decided:** `fetch_data.py` calls `dotenv.load_dotenv` and `dotenv.set_key`; `python-dotenv` is now a `dev` group dependency.

**Why:** the approved plan specified a four-line stdlib `KEY=VALUE` reader to avoid adding a dependency for something that small. `set_key` also has to handle the append-if-missing, quoting, and rewrite cases a hand-rolled version would either skip or reinvent, and dotenv is already the standard tool for exactly this job.

**Alternative considered:** the stdlib parser as planned. Rejected on review — small enough to write, but the quoting edge cases are exactly the kind of already-solved problem that justifies reaching for a dependency instead of stdlib.

**How to apply:** AGENTS.md's "dependencies are added in pyproject.toml, do not add one without asking" rule was followed — asked, and the user named `python-dotenv` directly.

---

## 2026-09-19 — E0 fetch scripts go in `scripts/`, not `catface.ml`, because one of them is app-track infra too

**Decided:** `fetch_weights.py` and `fetch_data.py` live in `scripts/`, run as `uv run python scripts/fetch_weights.py` and `uv run python scripts/fetch_data.py`, not under `catface.ml` as AGENTS.md's research-script command line describes.

**Why:** `fetch_weights.py` downloads the CatFLW-trained detector pair the app needs to run inference at all — PLAN_PROJECT's M2 build step calls this same script, not a research-only one. Filing it under `catface.ml` would put an app-track build dependency behind the import path the `api` track is forbidden from touching. `fetch_data.py` only serves E0, but splitting the two fetch scripts across two locations for that reason is worse than keeping both build-time fetchers in one place.

**Alternative considered:** `catface/ml/fetch_data.py`, per AGENTS.md's stated convention. Built first, then moved once the shared-infra point above became obvious mid-review.

**How to apply:** AGENTS.md's `uv run python -m catface.ml.<script>` line no longer covers either fetch script; it wants a `uv run python scripts/<script>.py` line next to it.

---

## 2026-09-19 — `models/class_means.json` is an E1 deliverable, and the second cross-track artifact

**Decided:** the per-class Procrustes mean shapes the editor maths subtracts get
written at E1 as `models/class_means.json` (committed, not fetched), rather than
falling out of the export step with the model.

**Why:** `/edit` computes `mean_shape[target] - mean_shape[predicted]`, so the
warper depends on the mean shapes, not on the classifier. Filing them behind the
model would have made PLAN_PROJECT.md's M5 fallback ("manual class picker,
everything else still works") false — with a manual picker you still need the
deltas. E1 already Procrustes-aligns every cached landmark, so the means are one
groupby over work that's happening anyway.

**Alternative considered:** deliver them with the ONNX export at RSCH-6.
Rejected — it puts a research artifact on M3's critical path and leaves the app
track's advertised independence untested until the end.

**How to apply:** AGENTS.md still says the only handoff between tracks is
`expression_head.onnx` + its manifest entry. That's now wrong and wants one line.

---

## 2026-09-19 — Cut four pieces of premature machinery from PLAN_PROJECT.md

**Decided:** each of these loses its oversized form and keeps a one-line
`add when:` trigger — the `convert` dependency group (tensorflow-cpu, tf2onnx),
the self-hosted Prometheus/Loki/Tempo/Grafana stack, the InferRequest pool plus
bounded wait queue on top of the semaphore, and locust/k6 for the load test. The
C++ bench CLI is deleted outright from both plans. `predictions.scores` becomes
one JSON column instead of three, and `edits` stops storing derivable landmarks.

**Why:** each was specified before the thing that justifies it exists. The
monitoring stack is the clearest case — it's several hundred MB on a box the same
document budgets at ~700 MB of 1024 MB, so it would OOM the app it observes; the
fallback table already conceded this, so plan and fallback are now swapped. The
`convert` group is ~600 MB for a conversion line 111 says is unnecessary. A
bounded semaphore already is a bounded queue. And three score columns bake a class
count into the schema while E4 is still deciding whether to merge classes.

**Alternative considered:** keep everything, mark it deferred in a subsection.
Rejected — a deferred section is a plan you still have to read past every time,
and the `add when:` triggers make each cut as reversible as a subsection would.

**How to apply:** don't re-add any of these because it feels professional. Re-add
when the named trigger fires, which for most of them is a measurement.

---

## 2026-09-19 — Backlog stays research-only; project work stays in the milestone list

**Decided:** don't write `[project]` backlog items for M0–M8 yet. The project
track runs off `PLAN_PROJECT.md`'s milestone list until M0 grooming starts.
Requirements that were stated in the plan but sat in no milestone (read the rubric
and weight the work, the image retention job, the five graded docs) were added to
the milestones themselves.

**Why:** `project_process.md` has PM groom one task immediately before it's
implemented. Groomed text for M8 written today is stale before it's read, and the
grooming happens anyway when the task is picked up — so writing it now is work
done twice.

**Alternative considered:** thin one-liner items per milestone. Rejected as a
second copy of the milestone list that would drift from it.

---

## 2026-09-19 — Backlog grooming shape: question/method/controls/stop-condition, not `_docs/task-template.md`

**Decided:** groom `docs/backlog.md` research items using the shape
`research_process.md` step 2 already specifies (question, method, required
controls, stop condition, checkable acceptance criteria) rather than the
four-section template `docs/team/pm.md` points at
(`_docs/task-template.md`).

**Why:** that template file doesn't exist anywhere in the repo. Inventing
one to satisfy a dangling reference would be scaffolding nobody asked for;
`research_process.md` already names the exact fields an experiment issue
needs, and it's the doc the research lifecycle actually runs on.

**Alternative considered:** write `_docs/task-template.md` first, then
groom against it. Rejected — out of scope for this pass, and PM's template
is meant for issues in general, not specifically research experiments,
which already have a more specific shape defined.

---

## 2026-09-19 — One backlog task per experiment (E0–E5), plus separate export and report tasks

**Decided:** 8 backlog items for the research track: E0 through E5 one each,
plus a standalone export task and a standalone `MODEL_REPORT.md`
compilation task, rather than one task per plan subsection (e.g. splitting
E2's three models or E4's two analyses into separate items).

**Why:** `research_process.md` states "one issue per experiment, in the
order PLAN_RESEARCH.md lists them" and treats export and the report as
separate lifecycle steps (step 8 for export, step 6 for report entries).
Splitting inside an experiment would fight the plan's own time-boxing,
which scores E2 and E4 as single 1-day units with multiple required parts,
not as multiple independent tasks.

**How to apply:** if a future experiment needs splitting (e.g. E3 turns out
to need more than 2 days), split it as a re-groom of that one item with a
Research PM note, not a silent backlog rewrite.
