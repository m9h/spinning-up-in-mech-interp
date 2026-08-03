# Rung 5 — Sparse autoencoders

## The idea

If features are superposed (rung 3), you cannot read them off neurons — but you can *learn a
dictionary* that unfolds them. A **sparse autoencoder (SAE)** is trained to reconstruct a
model's activations from a large, sparse set of latent features; those latents turn out to be
far more **monosemantic** than neurons, and you can *steer* the model by turning one up.

## Read

- **The ancestor, and it is not an AI paper:** Olshausen & Field, *Emergence of simple-cell
  receptive field properties by learning a sparse code for natural images* (Nature, 1996).
  Impose a sparsity penalty on a code for natural images and you recover oriented, localised,
  bandpass receptive fields — the simple cells Hubel and Wiesel found in visual cortex. Sparse
  coding is a **neuroscience** idea; the SAE literature reimported it twenty-five years later,
  and mostly does not cite it. Note the loop: **rung 1 measures those very receptive fields in a
  vision model.** The method you use here was invented to explain the thing you measured there.
- **Primary:** Bricken et al., *Towards Monosemanticity* (Anthropic, 2023).
- **Then:** Templeton et al., *Scaling Monosemanticity* (2024) — millions of features, feature
  steering ("Golden Gate Claude").

## Build

`starter.py` runs it end-to-end on **GPT-2 small**, which has *non-gated* pretrained SAEs
(Joseph Bloom's `gpt2-small-res-jb` release), so it needs no HF token and runs on CPU — no
SAE training required. It downloads one layer's SAE tensors directly (~150 MB) and uses plain
`transformers`:

1. **Interpret a feature by its decoder direction.** Logit-lens each feature's decoder row
   (`W_dec[f] @ W_U`) to see which tokens it writes toward, and auto-pick a sharp,
   word-initial content feature — no activation dataset needed to read what a feature *means*.
2. **Steer with it.** Add the feature's decoder direction to the residual stream at its own
   hook layer, injected at the feature's *natural activation scale* (calibrated, not past
   saturation). The top-5 next tokens flip from the sensible continuation to the feature's own
   tokens.

**Scale-up (canonical target):** Google's **Gemma Scope** SAEs for Gemma-2, via
[SAELens](https://github.com/decoderesearch/SAELens) / Neuronpedia — the same code, a bigger model
and millions of features. (Gemma-2 is gated, so it needs a HF token; GPT-2 keeps the starter
frictionless.)

## The control

Steering "works" trivially if *any* push changes the output — so the readout is a
**specificity** score: the mean logit of the feature's *own* tokens minus that of a fixed
control token set, which cancels the generic logit shift any large perturbation causes. Then
the null: steer with a **random direction of the same norm**, and with the **negation** of the
feature. A real feature raises its specificity far above the random baseline (which nets ~zero)
and its negation lowers it. In the runnable demo: feature **+6.5**, random null **+1.0**,
negation **−13.5**. This is the exact discipline the introspection paper failed (its negation
was "comparably effective") — you learn it here on open weights.

### Why this rung exists

We are not the first to notice this gap — we are the first to make it a runnable exercise.
[ARENA chapter 1, "Interpretability with SAEs"](https://www.arena.education/chapter1), the
best SAE course material available, states the problem exactly, in a **bonus** bullet:

> they found that autointerp on randomized SAE latents performed better than one might expect,
> because even random latents will display patterns when you take the top-k over a very large
> dataset

That is the whole thesis of this curriculum, correctly identified by someone else — and left
as an optional aside with no starter code, no test, and no solution. In that notebook,
"random direction" appears zero times and "null hypothesis" zero times.

The same gap runs through the wider ecosystem. Across Neuronpedia's educational surface,
controls appear only three times: as a **warning** (Gemma Scope's tooltips note that feature
labels are "sometimes misleading, inaccurate, or just plain wrong", without saying how to
check), as an **open problem** (its Open Problems tab asks for "comparing to fair
baselines!"), and as ARENA's bonus bullet above. Neuronpedia's own docs for
[features](https://docs.neuronpedia.org/features) and
[steering](https://docs.neuronpedia.org/steering) describe autointerp scores and steering with
no baseline or validation criterion at all. The honourable exception is the
[SAEBench writeup](https://www.neuronpedia.org/sae-bench/info), which does carry
randomly-initialized-model and PCA baselines — but as a results dashboard reproducing a paper,
not as something that teaches you to run your own.

So the default path through the ecosystem is *browse features → read the AI-written label →
steer → be impressed*. This rung is the missing step: **check**.

**We failed this ourselves, and an independent check caught it.**
[`tools/gate_autointerp.py`](../tools/gate_autointerp.py) asks whether this feature's
decoder-derived label describes where it *naturally fires* in real text — no steering at all,
which is the same question Delphi's `detection` scorer asks, computed exactly instead of by a
70B judge. The label survives (specificity **+3.24** at its top firing positions vs **+0.71** at
random and **+1.37** at another feature's top positions; correlation **+0.459** among active
positions). Building that gate is what exposed the calibration bug above. It runs in
`verify_all.sh`.

**And there is now a tool for the label half of it.** [Delphi](https://github.com/EleutherAI/delphi)
(EleutherAI) generates feature explanations *and scores them* — **detection** (given the
explanation, does a model correctly predict whether a whole sentence activates the feature?) and
**fuzzing** (the same at the level of individual highlighted tokens). Both are cheap. If you are
going to trust a label — and [Transluce Monitor](https://monitor.transluce.org) now ships
**~917,000** of them for Llama-3.1-8B — this is how you check one. See
[ECOSYSTEM.md](../ECOSYSTEM.md).

## Are learned features necessary? (a live question, 2026)

This rung's premise — superposed features can't be read off neurons, so learn a dictionary — is
**contested for circuit tracing**. Transluce's
[*Circuits Are Sparse in the Neuron Basis*](https://arxiv.org/abs/2601.22594) shows that with a
privileged basis (MLP *activations*, not outputs) and a stronger attribution method (RelP), raw
neurons give circuits as sparse and faithful as transcoder-based ones — reproducing three case
studies that had only been shown with learned features. See [rung 7](../rung7_attribution_graphs/)
for what that does and does not overturn.

**Independent convergence, worth weighing.** The
[Mechanistic Interpretability Benchmark](https://arxiv.org/abs/2504.13151) (MIB, ICML 2025)
reached the same verdict a year earlier by a completely different route — a held-out private test
set with public leaderboards — reporting that for causal variable localization, **"SAE features
are not better than neurons."** Two independent methods, same negative result. That is about as
strong as evidence gets here.

It does not touch what you did *here*: steering along a single interpretable direction, with a
null. But it should change what you require of the next paper claiming SAEs were **necessary** —
ask what neuron baseline it ran, and with which attribution method.

## Toward the recent papers

SAEs are the feature-finder behind the society-of-thought steering claim (rung 8) and a cousin
of the Jacobian lens (rung 6). Learning to steer-with-a-null here is exactly what lets you
adjudicate those claims.
