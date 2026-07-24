# Rung 5 — Sparse autoencoders

## The idea

If features are superposed (rung 3), you cannot read them off neurons — but you can *learn a
dictionary* that unfolds them. A **sparse autoencoder (SAE)** is trained to reconstruct a
model's activations from a large, sparse set of latent features; those latents turn out to be
far more **monosemantic** than neurons, and you can *steer* the model by turning one up.

## Read

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
[SAELens](https://github.com/jbloomAus/SAELens) / Neuronpedia — the same code, a bigger model
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

## Toward the recent papers

SAEs are the feature-finder behind the society-of-thought steering claim (rung 8) and a cousin
of the Jacobian lens (rung 6). Learning to steer-with-a-null here is exactly what lets you
adjudicate those claims.
