# Rung 3 — Superposition

## The idea

A network often needs to represent **more features than it has dimensions**. It does this by
storing features as *non-orthogonal* directions that partly overlap — **superposition** —
accepting a little interference in exchange for capacity. This is *the* reason
interpretability is hard: a single neuron is **polysemantic** (it participates in many
features), so you cannot read features off neurons directly. Toy models make superposition
visible and tunable, and they run on a laptop in seconds.

## Read

- **Primary:** Elhage, Hume, Olsson et al., *Toy Models of Superposition* (Anthropic, 2022) —
  the phase transition into superposition, feature geometry (antipodal pairs, pentagons), and
  the role of feature importance and sparsity.
- Optional: *Superposition, Memorization, and Double Descent* (2023).

## Build

No pretrained model needed — you *train* the toy model (`starter.py`, fully runnable):

1. Train the toy ReLU autoencoder `x → W → ReLU(Wᵀh + b) → x̂` that compresses `n` sparse
   features into `m < n` dimensions.
2. Sweep **sparsity**: at low sparsity the model represents only the top-`m` features
   (orthogonal, no superposition); as sparsity rises it packs *all* `n` features into `m`
   dims — superposition. Watch it happen.
3. Read the geometry: `WᵀW` reveals antipodal pairs / regular polygons.

## The control

The claim is that the model *recovers* the true features, not that it hallucinates structure.
The **null**: compare the learned dictionary `W` against a **random dictionary** of the same
shape — feature-recovery quality (how well `WᵀW` aligns to the identity on represented
features) must beat the random baseline. `starter.py` computes both and prints the gap. If
your "recovered features" score no better than random directions, you are seeing geometry in
noise.

## Toward the recent papers

Superposition is *why* you need sparse autoencoders (rung 5) to find features, and why the
J-space (rung 6) is only ~≤10% of activation variance yet does real work: the important,
reportable directions are a sparse code superposed in the residual stream.
