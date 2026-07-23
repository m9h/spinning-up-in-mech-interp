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

Open target: **Gemma Scope** — Google's released SAEs for Gemma-2 (via
[SAELens](https://github.com/jbloomAus/SAELens) / Neuronpedia). No SAE training required.

1. Load a Gemma-2-2B residual-stream SAE; encode activations on a prompt set.
2. Interpret a handful of live features (top-activating tokens) and confirm they are more
   monosemantic than the nearest raw neurons.
3. **Steer**: add a chosen feature's decoder direction to the residual stream and watch the
   generation bend toward that concept.

## The control

Steering "works" trivially if *any* push changes the output. The null: steer with a
**random direction** of the same norm, and with the **negation** of the feature. A real
feature's steering effect must exceed the random-direction baseline and be specific (the
concept appears, its negation doesn't). This is the same discipline the introspection paper
failed (its negation was "comparably effective") — you learn it here on open weights.

## Toward the recent papers

SAEs are the feature-finder behind the society-of-thought steering claim (rung 8) and a cousin
of the Jacobian lens (rung 6). Learning to steer-with-a-null here is exactly what lets you
adjudicate those claims.
