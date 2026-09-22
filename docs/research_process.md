Scope: PLAN_RESEARCH.md. Delivers `models/expression_head.onnx` and `docs/MODEL_REPORT.md`. Shares `catface.core` with the app track but otherwise runs independently — see AGENTS.md for how the two tracks relate.

- Work items are experiments (E0–E5), not GitHub issues in the generic sense — one issue per experiment, in the order PLAN_RESEARCH.md lists
  them: E0, E1, E2, E3, E4, E5 
- Read the question and required controls before starting and before closing
- A negative result is a valid, closeable outcome. An unreported missing control is not.
- Commit regularly, one directory per run under `experiments/<run>/`

Roles

- Research PM — grooms one experiment before it's run, follows
  `docs/team/research-pm.md`
- Data Steward — handles E0: licensing, dedup, class balance, the
  landmark cache, follows `docs/team/data-steward.md`
- ML Engineer — implements E1 onward, follows `docs/team/ml-engineer.md`
- Research Thinker - interprets the results of the experiments, follows `docs/team/research-thinker.md`
- Research QA — reviews methodology and required controls, follows
  `docs/team/research-qa.md`
- Export Verifier — gates the ONNX/OpenVINO handoff, follows
  `docs/team/export-verifier.md`

Orchestrator

The main session is the orchestrator. It launches Research PM, Data Steward, ML Engineer, Research Thinker, Research QA, and Export Verifier as subagents. It does not groom, run experiments, review methodology, interpret the results or verify exports itself.

Lifecycle

1. Pick the next experiment in order: E0, E1, E2, E3, E4, E5
2. Research PM grooms it: question, method, positive/negative shape, stop condition, required controls
3. Route the implementation: Data Steward for E0, ML Engineer for E1–E5 
4. Research QA reviews the result against the required controls
5. On FAIL:
   - if the problem is implementation (a bug, a metric miscomputed, a control that was skipped but is doable) — back to step 3 with
     the QA comment as input
   - if the problem is methodology (wrong split, a control that can't answer the question as scoped, a stop condition that fired) — back to step 2, Research PM re-scopes or accepts a negative result
6. On PASS: Research Thinker interprets the results for the Research PM to record it
7. Research PM records the result — including negative results — in `docs/MODEL_REPORT.md` under the matching question, then the orchestrator closes the issue
8. Repeat until E5 is closed
9. After E5's model is closed, Export Verifier runs before the artifact is considered deliverable. On FAIL here, back to the ML Engineer for the export step specifically — this is not a Research QA matter

Special case: E0 stop condition

If the detector fails on most images during E0, halt the research track entirely. Escalate to Research PM before E1 starts. Do not let E1–E5 issues get groomed against a landmark cache nobody has confirmed is usable.

Rules

- Do not skip step 2
- Required controls (e.g. Q1's random-adjacency comparison) are named at grooming time, not discovered at QA time
- Data Steward and ML Engineer do not close issues
- Research QA does not fix code or data, only outputs PASS or FAIL on methodology
- Research Thinker doesn not write the code, but can check whether it could have contained some peculiarities
- Export Verifier does not re-litigate whether the model is good, only whether the exported artifact matches what was evaluated
- The orchestrator closes an experiment issue only after Research QA outputs PASS, and closes the final handoff only after Export Verifier outputs PASS
- If the app track falls behind, research stops — this is PLAN_RESEARCH.md's own stated priority, not a call the research track makes for itself
