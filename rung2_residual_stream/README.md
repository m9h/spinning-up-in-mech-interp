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

Open model: **GPT-2 small** via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

1. Cache activations for a prompt; confirm the residual stream identity (each layer's output
   = input + head writes + MLP write).
2. For one head, compute its **QK attention pattern** and its **OV** effect on the logits
   (its "direct logit attribution").
3. Find a head whose OV consistently copies/moves a token — the seed of an induction head.

`starter.py` scaffolds a TransformerLens hook + direct-logit-attribution stub.

## The control

Attention patterns are famously *not* explanations (Jain & Wallace, *Attention is not
Explanation*, 2019). So don't trust the pattern — trust the **causal** effect: **ablate** the
head (zero its OV write, or mean-patch it) and measure the change in the model's output on the
behavior you attributed to it. If ablating a head you called "important" changes nothing,
your attention-pattern story was decoration. Compare against ablating a **random** head as a
baseline.

## Toward the recent papers

The residual stream is the object every lens reads (rung 6) and every attribution graph traces
(rung 7). "This layer writes direction *d* into the stream" is the sentence the workspace and
introspection papers are built out of.
