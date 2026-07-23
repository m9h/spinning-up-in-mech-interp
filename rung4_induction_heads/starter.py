"""Rung 4 starter — find induction heads + causal ablation (GPT-2 / Pythia).
Needs: pip install transformer_lens."""
# TODO: build repeated-random-token sequences; measure loss on the 2nd copy
# TODO: per-head induction score (attention from token i to the token after prev occurrence)
# CONTROL: ablate the top induction heads -> in-context (2nd-copy) loss must rise sharply while
#          ordinary loss barely moves, and ablating random heads must not reproduce the effect.
print("See README.md — compute induction scores, then run the selective-ablation control.")
