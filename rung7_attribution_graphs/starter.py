"""Rung 7 starter — attribution graph + faithfulness check (Gemma-2 / Llama).
Needs: circuit-tracer (github.com/safety-research/circuit-tracer)."""
# TODO: pick a 2-step behavior; build the attribution graph with circuit-tracer
# TODO: identify the intermediate feature carrying the bridging fact
# CONTROL: intervene on that feature; if the output does not move as the graph predicts, the
#          graph is UNFAITHFUL (Makelov/Lange/Nanda 2023). Report edge faithfulness.
print("See README.md — build the graph, then validate the intervention (faithfulness control).")
