# Rung 2 — The residual stream

## The idea

Transformers look complicated; their skeleton is simple. Every layer *reads from* and *writes
to* one shared **residual stream** — a running sum that each attention head and MLP edits. An
attention head is two circuits: **QK** (where to attend) and **OV** (what to move). Once you
see the residual stream as a communication channel, the rest of the field is legible.

## Read

- **Primary:** Elhage, Nanda, Olsson et al., *A Mathematical Framework for Transformer
  Circuits* (Anthropic, 2021) — QK/OV decomposition, the residual stream, and (a preview of)
  induction heads in a 2-layer attention-only model.

## Build

`starter.py` runs it on **GPT-2 small** with plain `transformers` (no TransformerLens needed —
though that is the canonical tool for this, and cleaner for per-head work). It measures the two
circuits of an attention head separately, each against its null:

1. **OV — what a head writes (weights only).** For every head, extract its OV matrix
   `W_V W_O` (residual → residual), send each token's embedding through it, unembed, and score
   how far the token promotes *itself* (a per-row z-score). This needs no forward pass at all —
   the copying behavior is in the weights. GPT-2's layer-11 heads score high; the top head lands
   at z ≈ +5.8.
2. **QK — where a head looks (one forward pass).** For a real sentence, measure each head's
   attention from position *i* to *i−1*. GPT-2's **L4H11** — the textbook previous-token head —
   comes out at ≈ 1.00.

Together, a QK that finds a position and an OV that copies from it are exactly the two pieces
rung 4 assembles into an induction head.

## The control

Attention patterns are famously *not* explanations (Jain & Wallace, *Attention is not
Explanation*, 2019), so neither score is trusted on its own — each is read against a null:

- **OV:** a **random matrix of the same norm** scores ≈ 0 (in the demo: top head +5.8, median
  head +1.0, random null +0.01) — copying is specific to particular heads, not an artifact of
  the measurement.
- **QK:** the **uniform-attention baseline** (~0.16 here) is what a head with no positional
  preference would score; a real previous-token head must beat it (L4H11 hits 1.00).

The deeper habit — carried to rung 4 — is causal: **ablate** a head you call important and
measure the behavior change against ablating a **random** head. If it changes nothing, the
attention-pattern story was decoration.

## Toward the recent papers

The residual stream is the object every lens reads (rung 6) and every attribution graph traces
(rung 7). "This layer writes direction *d* into the stream" is the sentence the workspace and
introspection papers are built out of.
