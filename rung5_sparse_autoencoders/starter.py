"""Rung 5 starter — load a Gemma Scope SAE, interpret + steer a feature (Gemma-2-2B).
Needs: pip install sae_lens transformers."""
# TODO: load a Gemma-2-2B residual SAE via SAELens; encode activations; interpret a feature
# TODO: steer by adding the feature's decoder direction to the residual during generation
# CONTROL: repeat steering with a RANDOM direction (same norm) and with the feature's NEGATION.
#          The real feature must beat the random baseline and be specific (concept appears; not
#          its negation) -- the null the introspection paper skipped.
print("See README.md — load an SAE, steer a feature, then run the random-direction null.")
