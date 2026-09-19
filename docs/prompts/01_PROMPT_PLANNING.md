# PROMPT_PLANNING.md

Documentation of how `PLAN_PROJECT.md` and `PLAN_RESEARCH.md` were produced with an
AI assistant, for the AI-usage section of the project.

## Honest note on what this is

The plans did not come from one prompt. They came from a conversation of about ten
turns that went: brainstorm ideas → research datasets → argue about scope → pick a
stack → write plans → cut them down → catch a modelling error → rewrite.

The prompt below is a reconstruction. It is what I would send if I were starting
over knowing what I know now, and it is written to be reusable rather than to
pretend the process was clean. The section after it records what the earlier
attempts got wrong, which is the part that actually matters for documenting
AI-assisted work.

## The prompt

```
I'm an ML engineer with limited software engineering experience. I'm doing the
DataTalks.Club AI Dev Tools Zoomcamp final project. Read the requirements at
https://github.com/DataTalksClub/ai-dev-tools-zoomcamp/tree/main/project — if
GitHub blocks you, the mirror at datatalks.club/docs works. Tell me if you can't
read the scoring rubric rather than guessing at it.

## The project

A web app for cat facial analysis:
- upload a photo, get 48 facial landmarks from a pretrained detector
- predict a facial expression class from those landmarks with a graph neural
  network I train myself
- let the user pick a target expression and a slider, then warp the landmarks and
  the image toward it (Delaunay triangulation, piecewise affine)

I want the GNN because I've never used one and want the experience. I want the
model exported to ONNX/OpenVINO and running on CPU.

## Decisions already made, don't relitigate them

- one repo, ML and app together, with an enforced import boundary
- uv for dependencies, groups so the runtime image doesn't pull torch
- SQLAlchemy, SQLite locally, Postgres in prod via DATABASE_URL
- OpenVINO for inference, called from Python in-process (no C++ subprocess)
- plain HTML/JS frontend, no build step
- FastAPI backend

## What I want you to produce

Two markdown files meant to live in the repo and be read by a coding agent:

1. PLAN_PROJECT.md — the shippable app. Everything here must work. Map it to the
   course requirements. Include layout, dependency setup, API shape, storage,
   inference, tests, CI/CD, deployment, observability, security, milestones, and a
   fallback for every risky component.

2. PLAN_RESEARCH.md — the model work. This is exploratory: open questions,
   experiments with stop conditions, time boxes. It delivers one artifact to the
   app and is never allowed to block it.

They share the repo, the pyproject, and a `core` package holding anything used by
both training and serving.

## How to write them

- Working notes, not a specification. Decisions and the reasons for them. Don't
  write code I haven't asked for — no full Dockerfiles, no complete pyproject, no
  exhaustive test lists. I'll fill those in when I get there.
- No em dashes, no "not X but Y", no throat-clearing.
- Every claim about a dataset, licence, model, or API limit must come from a search
  you actually ran. If you can't verify it, say so in the file.

## Before you write, research and report back

- What open cat-face datasets exist, with links, licences, instance counts, label
  types, and class balance. Be specific about what is NOT available.
- What pretrained cat-face models exist, with format, size, reported metrics, and
  how to convert them to something CPU-fast.
- What the 48-landmark scheme actually contains, point by point, so we know which
  quantities are computable from geometry and which need labels.

## Push back on me

If a piece of my plan doesn't survive contact with the available data, say so
before writing anything. Specifically: check that every quantity the model is
supposed to predict has a real label source, and that anything computable by
formula isn't being turned into a learning problem.
```

## What earlier versions got wrong

Recording these because a review of AI-generated work is worth more than a clean
transcript.

**Invented a target with no labels.** The first plan had the model predict three
"action unit" scores: ear rotation, eye aperture, muzzle tension. I asked where the
labels would come from. Two of the three are formulas computable directly from the
landmarks, so training a network for them was pointless. The third had no label
source anywhere and was effectively made up. The whole plan then included an evening
of hand-labelling 500 photos to measure a quantity that didn't exist.

Fix: predict expression class from the public labelled datasets, compute the two
geometric quantities as formulas, drop the third.

**Dismissed data that was fine.** The assistant treated the public cat emotion
datasets as unusable because the categories aren't scientifically validated for
cats. For a portfolio project they're perfectly usable with a one-line limitation in
the README. Rejecting them is what forced the invented-label detour.

**Over-engineered the first draft.** The first pair of plans ran roughly twice as
long, with full Dockerfiles, complete dependency files, and enumerated test cases
before a line of code existed. Asked for a compressed, rougher version and got
something more useful to actually work from.

**Got a current fact wrong by default.** Initial advice assumed the old AWS free
tier (750 hours/month for 12 months). A search showed accounts created after
July 2025 get a credit-based plan that closes the account when credits run out,
which would have taken the deployed demo offline before peer review.

**Couldn't read the rubric.** GitHub blocks automated fetching, so the requirement
mapping came from the course docs mirror and module descriptions, not the scored
rubric itself. Both plans say so explicitly and flag reading it as a task for me.
That's the right behaviour: an unverifiable claim marked as unverified rather than
filled in confidently.
