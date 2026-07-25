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

`starter.py` runs it on **InceptionV1** — the actual Distill model, available as torchvision's
`googlenet` (ImageNet weights download once, ~50 MB). Rather than optimization-based feature
visualization (powerful but heavy, via `lucent` / OpenAI Microscope), it characterizes a
feature the tractable way — a **synthetic tuning curve**:

1. Load InceptionV1 and probe its first conv layer with **oriented gratings** across 12
   orientations × 7 spatial frequencies × 4 phases.
2. Score each of the 64 channels by **orientation selectivity** (1 − circular variance),
   measured **at that channel's preferred spatial frequency**. The sharpest is a near-perfect
   edge detector — in the demo, channel #48 fires only for ~0° edges (selectivity 1.000), a
   single spike in the tuning curve.
3. That the feature is *legible in the weights* is the whole point: this unit is an oriented
   Gabor filter the network learned.

> **Measure at the preferred frequency.** Averaging responses *across* frequencies understates
> tuning. conv1 is a 7×7 stride-2 filter that prefers **fine** gratings; with too narrow or too
> coarse a frequency range it can look *less* selective than a random network. We got exactly
> that wrong result while building the atlas — the fix is step 2 above.

*(Feature visualization and top-activating dataset crops — the other two ways to characterize a
feature — are the natural next exercise; the tuning curve is the one that runs in seconds and
carries a clean null.)*

## The control

A sharp tuning curve is not proof on its own — so the starter runs **two** nulls, each as a
*distribution* over seeds rather than a single draw:

- **Random-init** (Adebayo et al., *Sanity Checks for Saliency Maps*, 2018): rebuild the same
  architecture with random weights, repeat the identical probe.
- **Weight-shuffle** (the stronger null): permute each trained kernel's weights within-channel,
  preserving the weight *distribution* while destroying its structure.

Result: trained top **1.000**, random-init null tops out at **0.528**, weight-shuffle at
**0.980**; 34/64 units beat every random-init seed, 40/64 beat every shuffle seed. The sharpest
detectors are real. **Passing this is the difference between a feature and a Rorschach blot.**

Two things the original papers' pictures could not tell you, and which the numbers do:

1. **The median unit is not the top unit.** Median trained selectivity is only ~0.25 — "conv1
   is all Gabor filters" is too strong a reading.
2. **Which null you choose changes the answer.** Random-init is a *weak* control, because a
   randomly-initialized network is nearly dead deeper in the net — in our full-network atlas,
   100% of `inception5b` units beat the random-init null but only 46% beat the weight-shuffle
   null. Always report which control you ran.

## Toward the recent papers

"Feature," "circuit," and "this direction *means* something" are the primitives every later
paper inherits. The Jacobian lens (rung 6) is a feature detector for *reportable* directions;
sparse autoencoders (rung 5) are how you find features at scale. It all starts here.
