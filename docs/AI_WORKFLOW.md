# AI_WORKFLOW.md

Log of how AI assistance was used, written as it happens. Newest entry on top.
One entry per session/task: what was asked, which tool/agent did it, what a
human checked or changed before accepting it.

---

## 2026-09-19 — Reviewed the three planning docs for gaps and over-engineering

**Asked:** review `PLAN_RESEARCH.md`, `PLAN_PROJECT.md` and `backlog.md`; find
missing tasks, find over-engineered schemas that don't make sense yet, rewrite
them safely.

**Tool:** Claude Code, main session, `/ponytail full` (a skill that forces the
minimum solution and makes you justify anything speculative). No subagents — the
repo is 16 files and fully readable in one pass, so the plan-mode workflow's
explore agents would have re-derived context the session already had.

**What it did:** found four unowned artifacts (`core.geometry` had no owner
though both tracks import it; per-class mean shapes were needed by `/edit` and
produced by nothing; the rubric read, the retention job and the five graded docs
were requirements sitting in no milestone), and six pieces of machinery specified
before the thing justifying them existed. Both sets are written up in
`DECISIONS.md` with the reasoning and, for every cut, the condition that brings it
back.

**Checked by human:** two scope calls were mine, not the model's — it proposed a
thin `[project]` backlog covering M0–M8 and I declined it, since
`project_process.md` grooms project tasks just-in-time and the pre-written text
would be stale before it was read. I also chose deletion-with-a-trigger over a
"deferred" subsection.

**What it caught that I'd have missed:** the M5 fallback was quietly false. It
claims the warper works without the research track, but the warper needs the
per-class mean shapes, which only the research track can produce — so the
"independent" app track had a hidden dependency on an artifact nobody had
scheduled. The fix moves that artifact to E1, where the alignment it needs is
already being computed, instead of to export time behind the model.

**Correction it made to its own plan:** the first version filed `class_means.json`
as an export-time deliverable (RSCH-6). That would have put it after E5 and left
M3 blocked on the whole research track. Moved to RSCH-1 before implementing.

---

## 2026-09-19 — Groomed research backlog, seeded AI_WORKFLOW.md / DECISIONS.md

**Asked:** turn `PLAN_RESEARCH.md` into a fine-grained, groomed backlog and
start these two log files.

**Tool:** Claude Code, main session, no subagents (file-authoring task, not a
research-track lifecycle step — `research_process.md`'s Research PM role
applies once real experiments are being groomed one at a time against live
results, not to this one-shot doc pass).

**What it did:** read `AGENTS.md` and `PLAN_RESEARCH.md`, cross-checked
against `research_process.md` and the team role docs (`pm.md`,
`ml-engineer.md`, `research-qa.md`, `export-verifier.md`, `data-steward.md`)
for the grooming shape actually used downstream, then wrote
`docs/backlog.md` with one task per experiment (E0–E5) plus export and
report-compilation as their own tasks — 8 items total, each with question,
method, required controls, checkable acceptance criteria, and stop
condition.

**Checked by human:** not yet — first pass, pending review before E0 is
picked up for real.
