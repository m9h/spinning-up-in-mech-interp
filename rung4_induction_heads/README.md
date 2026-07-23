# Rung 4 — Induction heads

## The idea

**Induction heads** are the first non-trivial circuit discovered in language models, and the
mechanism behind much of **in-context learning**: having seen `…A B … A`, an induction head
attends back to the earlier `B` and predicts `B` again. They appear abruptly during training
(a "phase change") right as in-context learning switches on. This is a concrete, causal,
reproducible circuit — the proof that transformer internals can be reverse-engineered.

## Read

- **Primary:** Olsson, Elhage, Nanda et al., *In-Context Learning and Induction Heads*
  (Anthropic, 2022).

## Build

Open model: **GPT-2 small** or **Pythia** (TransformerLens).

1. Build the induction test: random repeated token sequences `…[rand tokens]…[same tokens]…`
   and measure loss on the second copy.
2. Find the induction heads via the **prev-token → match** signature (an attention head that,
   at position *i*, attends to the token after the previous occurrence of the current token).
3. Confirm with **direct logit attribution**: the head writes the repeated token into the
   logits.

`starter.py` scaffolds the repeated-sequence eval + per-head induction score.

## The control

Correlation isn't mechanism. **Ablate** the candidate induction heads and show that
in-context (second-copy) loss rises sharply while ordinary next-token loss barely moves — and
that ablating an equal number of **random** heads does not. The causal, selective ablation is
the claim; the attention pattern is only the hint.

## Toward the recent papers

Induction is the template for "a named circuit does a nameable thing, provably." The society-
of-thought and workspace papers make far grander claims of the same shape — which is exactly
why rung 8 insists each one carry an ablation/null like this one.
