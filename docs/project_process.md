Scope: PLAN_PROJECT.md. Delivers the FastAPI app, storage, warper,
frontend, deploy, observability, and the rest of the M0–M8 milestone
list. Shares `catface.core` with the research track; `api` never
imports `ml`.

Status: roles for this track are not finalized yet. PLAN_PROJECT.md's
scope is wider than the research track's — API contract, storage,
concurrency, Docker/CI, observability, agents, security — and it likely
needs more than three roles once it's broken down (e.g. something that
owns the memory/concurrency budget specifically, something that owns
the incident-responder agent, may not be the same person implementing
routes). Don't invent that split speculatively; define it against the
actual milestone breakdown when M0 grooming starts.

Until dedicated roles exist, this track borrows the existing generic
roles as a placeholder:

- PM — grooms a task before implementation, follows `docs/team/pm.md`
- Engineer — implements one groomed task, follows
  `docs/team/software-engineer.md`
- QA — checks the result against acceptance criteria, follows
  `docs/team/qa-engineer.md`

Orchestrator

The main session is the orchestrator. It launches PM, Engineer, and QA
as subagents. It does not groom, implement, or test itself.

Lifecycle

1. Pick the next open issue, milestone order first (M0 before M1, etc.
   — PLAN_PROJECT.md says order by rubric weight within that, not by
   what's fun)
2. PM grooms it
3. Engineer implements it
4. QA verifies it
5. On FAIL, back to step 3 with the QA comment as input
6. On PASS, close the issue
7. Repeat until the milestone's issues are empty, then move to the
   next milestone

Rules

- Do not skip step 2
- The engineer does not close the issue
- QA does not fix the code, only outputs PASS or FAIL
- The orchestrator closes the issue only after QA outputs PASS
- `api` never imports `ml` — this is grep-testable and the test is part
  of what M0's CI sets up, not optional later cleanup
- If a groomed task depends on `models/expression_head.onnx` and the
  research track hasn't delivered it, that's expected per
  PLAN_PROJECT.md's fallback table (M5: "class picker becomes manual")
  — don't block on research, implement the fallback and file a
  follow-up to swap the real model in later

Revisit this file once M0–M1 are underway and the actual shape of the
work (route-per-issue vs. infra-per-issue vs. the concurrency/ops work
that doesn't fit the route template) is clearer. The research track's
roles (`docs/team/research-*.md`) are a template for how to specialize
away from the generic three if the same problem shows up here.
