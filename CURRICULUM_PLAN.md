# Spinning Up in Mechanistic Interpretability — curriculum plan

## Thesis
Walk the mech-interp lineage — Distill vision circuits (2020) → Anthropic transformer circuits
(2021+) → the 2025–26 property papers — on **open-weight models**, with **a control at every
rung**, so a learner ends able to *adjudicate* the recent claims, not just read them.

## Structure (one directory per rung; each: README = the reading+build+control, starter.py = the project)
- **rung0 — build the model first** (prereq): nanoGPT / Raschka LLM & Reasoning from Scratch / Mike X Cohen (50 ML projects to understand LLMs, github.com/mikexcohen/ML4LLM_book).
- **rung1 — features & circuits** (vision, InceptionV1) — control: model-randomization sanity check.
- **rung2 — residual stream** (GPT-2, TransformerLens) — control: ablate-vs-random-head.
- **rung3 — superposition** (toy model, laptop; runnable) — control: random-dictionary null.
- **rung4 — induction heads** (GPT-2/Pythia) — control: selective causal ablation.
- **rung5 — sparse autoencoders** (Gemma Scope) — control: random-direction / negation steering null.
- **rung6 — lenses** (jlens/jlens-lab on OLMo-3) — controls: randomization, distance null, logit-lens.
- **rung7 — attribution graphs** (circuit-tracer, Gemma-2/Llama) — control: intervention faithfulness.
- **rung8 — property claims** (OLMo-3/Ministral, our results) — control: the null per claim; negatives first-class.

## Each rung README follows: The idea → Read → Build → The control → Toward the recent papers.

## Build order / status
- rung3 has a complete, laptop-runnable reference (`starter.py`) + its control.
- rungs 6 & 8 run on the working companion code (jacobian-lens, jlens-lab, the Hub lenses).
- rungs 0,1,2,4,5,7 ship a README + a scaffolded starter pointing at the canonical open target.

## Verification
Every rung's `starter.py` should, when complete, (a) reproduce the technique on the named open
model and (b) run the control and print pass/fail. rung3 already does both.
