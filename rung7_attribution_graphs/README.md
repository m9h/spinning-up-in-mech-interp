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

Open tooling: **[`circuit-tracer`](https://github.com/safety-research/circuit-tracer)**
(released), on **Gemma-2-2B** or **Llama**.

1. Pick a small, sharp behavior (e.g. a two-step factual lookup, "the capital of the country
   whose capital is …").
2. Build the attribution graph; identify the intermediate feature that carries the bridging
   fact.
3. Intervene on that feature and confirm the downstream answer changes as the graph predicts.

## The control

Attribution graphs can be *unfaithful* — clean, causal-looking, and wrong — by exploiting
dormant pathways (Makelov, Lange & Nanda, 2023; Anthropic's own *Toy Model of Mechanistic
Unfaithfulness*, 2025). The check: **validate the intervention**. If forcing the intermediate
feature does not move the output as the graph claims, the graph is a story, not a mechanism.
Always report the faithfulness of the edges you rely on.

## Toward the recent papers

Attribution graphs are the causal backbone under several 2025–26 results. Knowing their
failure mode (unfaithfulness) is precisely the skepticism rung 8 asks you to bring to the
grandest claims.
