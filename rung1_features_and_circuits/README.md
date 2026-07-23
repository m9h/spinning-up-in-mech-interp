# Rung 1 — Features & circuits

## The idea

A neural network is not an inscrutable blob. It is built from **features** — directions in
activation space that detect something (a curve, an edge, a dog's snout) — wired together by
weights into **circuits** that compute. This is the founding claim of mechanistic
interpretability, and it is cleanest to *see* for the first time in a vision model, where a
feature is literally a picture. You will find one, and trace one small circuit end to end.

## Read

- **Primary:** Olah, Cammarata, Schubert et al., *Zoom In: An Introduction to Circuits*
  (Distill, 2020) — features, circuits, universality, polysemanticity.
- **Then:** *Curve Circuits* (2021) — a full circuit reverse-engineered from weights.
- Optional method background: *Feature Visualization* (2017).

## Build

Open model: **InceptionV1** (GoogLeNet), ImageNet-trained — weights are public (via `lucent`
/ OpenAI Microscope), and it is the model the source articles use.

1. Load InceptionV1 and pick a mid-level neuron (e.g. a curve detector in `mixed3b`).
2. Characterize it three ways: **feature visualization** (optimize an input that maximizes
   it), **dataset examples** (top-activating ImageNet crops), and a **synthetic sweep**
   (rotate a curve stimulus → a tuning curve).
3. Trace *one* weight connection: show that the curve detector reads from earlier
   edge/orientation neurons with the sign structure you'd predict.

`starter.py` scaffolds the load + a feature-visualization stub with `lucent`.

## The control

A pretty picture is not proof. Run the **model-randomization sanity check** (Adebayo et al.,
*Sanity Checks for Saliency Maps*, 2018): re-initialize the network's weights and repeat the
feature visualization. A genuine feature depends on the *trained* weights — its visualization
should collapse to noise. If your "curve detector" looks curve-like in a randomized network,
you were reading the architecture/optimization prior, not a learned feature. **Passing this
is the difference between a feature and a Rorschach blot.**

## Toward the recent papers

"Feature," "circuit," and "this direction *means* something" are the primitives every later
paper inherits. The Jacobian lens (rung 6) is a feature detector for *reportable* directions;
sparse autoencoders (rung 5) are how you find features at scale. It all starts here.
