"""Rung 2 starter — residual stream + direct logit attribution (GPT-2 small).
Needs: pip install transformer_lens."""
# TODO: model = HookedTransformer.from_pretrained("gpt2"); cache = model.run_with_cache(prompt)
# TODO: verify residual identity; compute one head's QK pattern and OV direct-logit-attribution
# CONTROL: ablate the head (hook that zeros its output) and measure the behavior change vs
#          ablating a RANDOM head. Trust the causal effect, not the attention pattern.
print("See README.md — build the DLA, then run the ablation-vs-random-head control.")
