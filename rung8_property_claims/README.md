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
- **Background for the post-training claims** (near-required for the workspace cell): Lambert,
  [*RLHF Book*](https://rlhfbook.com) — **Lecture 6 (DPO)** and **Conversation 2, a case study
  in DPO for OLMo 3**. The workspace cell below measures what SFT / DPO / RLVR *do to a model's
  representations*; that course explains what those stages *are* and how they were actually
  built for this exact model. Read it before deciding what our numbers mean.

## Build (choose one or more cells)

All on **OLMo-3 / Ministral**, using the companion repos; each has a worked reference in
[`jacobian-lens/results`](https://github.com/m9h/jacobian-lens):

- **Workspace / point-of-view.** Fit lenses across the OLMo-3 post-training ladder; measure how
  far post-training moves the J-space, capability-controlled.

  > Our result: post-training moves the J-space ~31% (cos 0.69 from base) while MMLU is flat to
  > slightly *down* — a large representational change with no competence gain. **Method sets the
  > magnitude, not domain:** SFT+DPO moves it ~5× more than RLVR, and varying the RLVR domain at
  > matched capability adds ~1%. Read against a **per-layer** refit floor (rung 6), not a pooled
  > one.
  >
  > This is why the DPO reading matters: the stage our measurement singles out is the one
  > Lambert's Conversation 2 walks through building, for this same model.
- **Introspection.** Inject a concept and test whether the model *reports* it — with the
  strength sweep and asked-vs-neutral contrast that separate introspection from steering.
- **Societies of thought.** Reproduce the steering accuracy gain — then check it on a *second*
  benchmark.
- **Metacognition.** Does the workspace covertly encode the model's own errors, and when does
  reportability of that signal emerge across training?

- **Instrument convergence (new cell).** Rather than adjudicating one claim, ask whether two
  *independent* instruments read the same thing from the same activation. Run a J-lens (a
  linear readout) and a Natural Language Autoencoder (a trained RL'd verbalizer) on the same
  layer-41 residual of `gemma-3-27b-it`, score their agreement, and re-pair the readouts across
  *different prompts* as the null.

  > Our result: matched agreement **0.298** vs mismatched **0.007** (z = **+8.05**), with **79%**
  > of items beating every one of ~40 cross-prompt mispairings against a 2.4% chance rate. Two
  > instruments sharing no machinery recover the same content — convergent validity for
  > "verbalizable content" as a property of the *representation*, which neither paper can claim
  > alone.

  Then make it **causal** with SAE steering as ground truth: inject a known feature and ask
  whether both instruments report it.

  > Steered **0.188** (J-lens) / **0.118** (NLA); negation, random direction of matched norm, and
  > unsteered baseline all at floor (0.000–0.008). Injecting a *known* concept is detected;
  > injecting nothing is not.

  Write-ups: `tri-lens/results/PHASE1_RESULT.md`, `PHASE2_RESULT.md`.

### Build your own ladder, then measure it

Every cell above reads AI2's *pre-trained* OLMo-3 ladder. You can also close the loop yourself:
[TRL](https://github.com/huggingface/trl) v1.0 gives you SFT, reward modelling, DPO and GRPO in
one stack, so on a small open model you can **train your own miniature post-training ladder**
and then run rung 6's lenses and rung 8's controls on the checkpoints you produced.

That is the strongest version of this rung, because you control the independent variable. Our
finding — that method (SFT+DPO vs RLVR) sets how far the J-space moves, while task *domain*
adds ~1% at matched capability — is exactly the kind of claim a self-built ladder can check
without asking anyone for access.

## The control (the whole point)

Every cell ships its null, and **a negative is a first-class result**:
- workspace geometry vs a **distance null** (most "structure" is drift);
- the paper's quality metric vs a **logit-lens baseline** (it can reward noise);
- introspection vs the **negation / neutral-continuation** control (ours came out: *steering,
  not introspection*);
- society-of-thought's gain on a **second benchmark** (ours: −22 on MATH-Hard — it reverses);
- instrument agreement vs a **mismatch null**, re-pairing readouts across different prompts;
- and before any of it, a **positive control**: reproduce a published number, and confirm your
  red cases fail. We reported a null result about a 2026 method that was two bugs of our own —
  see [READING_A_PAPER.md](../READING_A_PAPER.md), question zero;
- metacognition's internal signal **beyond output confidence**, and its emergence localized to
  a training stage.

## You are done when

You can take *any* new interpretability claim, reproduce its method on an open model, attach a
null, and say — with evidence — whether it survives. That is the skill the whole ladder exists
to build, and the function the field is missing.
