# Rung 8 — The recent claims, adjudicated (capstone)

## The idea

You now have every tool the 2025–2026 papers use. The capstone is not to *believe* their
claims — global workspace, introspection, societies of thought, metacognition — but to
**reproduce each on open weights and run the control that could falsify it.** This is the last
rung of the ladder and the first cells of the
[Consciousness-Indicator Scorecard](https://github.com/m9h/jacobian-lens).

## Read

- *Verbalizable Representations Form a Global Workspace…* (Gurnee et al., 2026)
- *Emergent Introspective Awareness…* (Lindsey, 2025)
- *Reasoning Models Generate Societies of Thought* (Kim et al., 2026)
- *When Models Manipulate Manifolds* (Gurnee et al., 2025)
- Background for the reasoning claims: Raschka, *Build a Reasoning Model from Scratch*.

## Build (choose one or more cells)

All on **OLMo-3 / Ministral**, using the companion repos; each has a worked reference in
[`jacobian-lens/results`](https://github.com/m9h/jacobian-lens):

- **Workspace / point-of-view.** Fit lenses across the OLMo-3 post-training ladder; measure how
  far post-training moves the J-space, capability-controlled.
- **Introspection.** Inject a concept and test whether the model *reports* it — with the
  strength sweep and asked-vs-neutral contrast that separate introspection from steering.
- **Societies of thought.** Reproduce the steering accuracy gain — then check it on a *second*
  benchmark.
- **Metacognition.** Does the workspace covertly encode the model's own errors, and when does
  reportability of that signal emerge across training?

## The control (the whole point)

Every cell ships its null, and **a negative is a first-class result**:
- workspace geometry vs a **distance null** (most "structure" is drift);
- the paper's quality metric vs a **logit-lens baseline** (it can reward noise);
- introspection vs the **negation / neutral-continuation** control (ours came out: *steering,
  not introspection*);
- society-of-thought's gain on a **second benchmark** (ours: −22 on MATH-Hard — it reverses);
- metacognition's internal signal **beyond output confidence**, and its emergence localized to
  a training stage.

## You are done when

You can take *any* new interpretability claim, reproduce its method on an open model, attach a
null, and say — with evidence — whether it survives. That is the skill the whole ladder exists
to build, and the function the field is missing.
