# Rung 7 — Attribution graphs

## The idea

Beyond single circuits: **attribution graphs** trace a whole computation across layers by
replacing the model with an interpretable **transcoder** ("replacement model") and following
which features caused which. This is how you get case studies like "the model plans the last
word of a rhyming line before writing the line."

## Read

- **Primary:** Ameisen et al., *Circuit Tracing: Revealing Computational Graphs in Language
  Models* (2025) — the method.
- **Then:** Lindsey et al., *On the Biology of a Large Language Model* (2025) — the case
  studies (multi-step reasoning, planning, multilingual, refusals, hallucination).

## Build

`starter.py` runs the method *underneath* attribution graphs — **activation patching** (causal
tracing) — on **GPT-2 small** with plain `transformers`, so the idea is concrete before you
scale to the full tool. On the classic IOI task ("When John and Mary went to the store, John
gave a drink to ___" → " Mary"):

1. Corrupt the subject (John→Mary) so the answer flips to " John" — a clean/corrupted pair that
   differ at exactly one token.
2. Copy the **clean** residual stream into the **corrupted** run, one (layer, position) at a
   time, and measure how much of the answer each cell restores.
3. Read the resulting layer × position map: name-identity sits at the **subject token in early
   layers**, then hands off to the **final token in late layers** (where the name-mover heads
   write the answer). That sparse, moving path *is* a minimal attribution graph — you can watch
   the computation cross layers.

**Scale-up (two, and they disagree — see the control below):**
**[`circuit-tracer`](https://github.com/safety-research/circuit-tracer)**
(released) builds transcoder-based attribution graphs on Gemma-2-2B / Llama, resolving the path
down to individual *features* rather than residual positions — the same causal logic, at feature
resolution.

## The control

The control is built into patching and is the whole point: **the median (layer, position) cell
restores ~0%** — only a handful of cells carry the computation. A component matters only if
patching it moves the output. This is also the guard against the failure mode of the scaled-up
method: attribution graphs can be *unfaithful* — clean, causal-looking, and wrong — by exploiting
dormant pathways (Makelov, Lange & Nanda, 2023; Anthropic's own *Toy Model of Mechanistic
Unfaithfulness*, 2025). The check is the same reflex: **validate the intervention**. If forcing
a feature the graph names does not move the output, the graph is a story, not a mechanism.
Always report the faithfulness of the edges you rely on.

## ★ The control that narrowed the claim (2026)

Everything above assumes the replacement model must be built from **learned features** —
transcoders or SAEs — because neurons are polysemantic (rung 3). Transluce ran the baseline the
literature never ran, and it ties:
[*Language Model Circuits Are Sparse in the Neuron Basis*](https://arxiv.org/abs/2601.22594)
(2026), code at [`TransluceAI/circuits`](https://github.com/TransluceAI/circuits) (ADAG).

Two changes, neither of them a learned dictionary:

1. **Attribute over MLP *activations*, not MLP *outputs*.** Post-nonlinearity activations are a
   *privileged basis* — aligned with the model's own element-wise nonlinearity. Worth **~100×
   smaller circuits at equal faithfulness**, by itself.
2. **RelP instead of Integrated Gradients** — linearize the nonlinearities via straight-through
   estimation, so attribution costs **one backward pass instead of ten**. (Concurrently proposed
   by [Jafari et al.](https://arxiv.org/abs/2508.21258); Transluce apply it neuron-to-neuron.)

Result on Llama-3.1-8B-Instruct: ~80% faithfulness at ~100k edges (10% of candidates), and **three
of the case studies above — multi-hop Dallas→Texas→Austin, addition, multilingual antonyms —
reproduced with raw neurons alone**, having originally been demonstrated with cross-layer
transcoders.

**Why this is the lesson and not just a result.** The original comparison varied **two things at
once**: the *basis* (learned features vs neurons) and the *attribution machinery*. Credit went
entirely to the first. Separate them and most of it belongs to the second. That is
[PITFALLS](../PITFALLS.md) **#14 — too-narrow comparison set** — operating at the scale of a
research program rather than a script, and it is exactly what a well-chosen baseline is for.

**Do not overclaim it, either.** Superposition is not refuted — rung 3's toy model is a
mathematical fact. What is refuted is the *inference from* superposition: that you must undo it
to trace a circuit. And Transluce state the remaining case for learned features themselves —
there are "still too many neurons in a reasonably comprehensive circuit for a human being to
easily interpret," with automatic descriptions of limited quality. **Neurons win on sparsity and
faithfulness; learned features may still win on interpretability per unit.** Those are different
claims, and keeping them apart is the whole skill.

## Toward the recent papers

Attribution graphs are the causal backbone under several 2025–26 results. Knowing their
failure mode (unfaithfulness) is precisely the skepticism rung 8 asks you to bring to the
grandest claims.
