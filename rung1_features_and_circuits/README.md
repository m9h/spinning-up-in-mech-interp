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

1. Load InceptionV1 and probe its first conv layer with **oriented gratings** at 12
   orientations.
2. Score each of the 64 channels by **orientation selectivity** (1 − circular variance of its
   tuning). The sharpest is a near-perfect edge detector — in the demo, channel #31 fires only
   for ~0° edges (selectivity 0.998), a single spike in the tuning curve.
3. That the feature is *legible in the weights* is the whole point: this unit is an oriented
   Gabor filter the network learned.

*(Feature visualization and top-activating dataset crops — the other two ways to characterize a
feature — are the natural next exercise; the tuning curve is the one that runs in seconds and
carries a clean null.)*

## The control

A sharp tuning curve is not proof on its own. Run the **model-randomization sanity check**
(Adebayo et al., *Sanity Checks for Saliency Maps*, 2018): rebuild the *same architecture* with
**random weights** and repeat the identical probe. A genuine learned feature must vanish. It
does — trained top-5 selectivity ≈ **[1.0, 0.92, 0.84, …]** versus random weights capping at
**~0.45**: sharp orientation tuning exists only in the trained network, so it is a learned
feature, not an artifact of the architecture or the probe. **Passing this is the difference
between a feature and a Rorschach blot** — and the same randomization null guards every rung
above.

## Toward the recent papers

"Feature," "circuit," and "this direction *means* something" are the primitives every later
paper inherits. The Jacobian lens (rung 6) is a feature detector for *reportable* directions;
sparse autoencoders (rung 5) are how you find features at scale. It all starts here.
