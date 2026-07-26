# Glossary

The vocabulary you need to read the papers behind these rungs. Terms are grouped by where they
first matter, and each says which rung makes it concrete.

---

## The basic objects

**Feature** — a direction in activation space that represents something (a curve, a language, a
sense of "uncertainty"). The founding claim of the field is that networks are built from
features, not from inscrutable numbers. *Rung 1.*

**Circuit** — features connected by weights, computing something. "Curve detectors read from
oriented-edge detectors" is a circuit claim. *Rung 1.*

**Residual stream** — the running sum that every layer of a transformer reads from and writes
to. Not a "layer output" but a shared channel; attention heads and MLPs *add* to it. Almost
every technique above rung 2 is a way of reading or editing this stream. *Rung 2.*

**Polysemantic** — a neuron that responds to several unrelated things. The obstacle to reading
networks neuron-by-neuron. **Monosemantic** is the opposite, and the goal.

**Superposition** — the reason polysemanticity happens: when features are *sparse*, a network
can pack more of them than it has dimensions, at the cost of interference. *Rung 3.*

## Attention mechanics

**QK circuit** — the query–key product; decides *where* a head attends. **OV circuit** — the
value–output product; decides *what* it writes into the residual stream from there. A head
factors into these two independent pieces. *Rung 2.*

**Previous-token head** — attends from position *i* to *i−1*. Unglamorous, but it is the
prerequisite that makes induction possible. *Rung 2.*

**Induction head** — implements "…A B … A → B": finds an earlier occurrence of the current
token and copies what followed it. The circuit behind much in-context learning. Needs a QK that
finds the match and an OV that copies. *Rung 4.*

**Name-mover head** — in the IOI task, a late head that writes the answer name into the final
position. *Rung 7.*

**Copying score** — how strongly a head's OV circuit promotes the token it attends to,
measurable from weights alone. *Rung 2.*

## Measurement and intervention

**Ablation** — deleting a component (zeroing it, or replacing it with a mean) and measuring the
behavioural damage. The basic causal test. Always run it against **randomly chosen components**
as a baseline. *Rungs 2, 4.*

**Activation patching / causal tracing** — run a clean and a corrupted input, then copy clean
activations into the corrupted run one location at a time to find which locations restore the
answer. The method underneath attribution graphs. *Rung 7.*

**Direct logit attribution (DLA)** — projecting a component's write to the residual stream
through the unembedding to see which tokens it promotes.

**Steering** — adding a direction to the residual stream to push behaviour. Cheap to do, easy
to over-interpret: see *specificity*. *Rungs 5, 8.*

**Specificity** — whether an effect is particular to the thing you intervened on. Established
by two controls: a **random direction of matched norm** should do nothing, and the
**negation** should do the opposite. *Rung 5.*

**Null / control** — what your measurement returns on a system with no learned structure
(random weights, shuffled weights, a random direction, an untrained checkpoint). A result means
nothing without one — and the null must be *capable of scoring*. *Every rung.*

**Randomization sanity check** — re-run the method on a randomly-initialized model; genuine
findings must vanish. Introduced for saliency maps by Adebayo et al. (2018). *Rung 1.*

**Faithfulness** — whether an explanation actually describes the computation. Attribution
graphs can be clean, causal-looking, and unfaithful. *Rung 7.*

## Reading the residual stream

**Logit lens** — project a mid-layer residual through the unembedding to see what the model
"would say" there. Crude but free.

**Tuned lens** — a trained per-layer translator that corrects the logit lens's bias.

**Jacobian lens** — reads the residual stream through the model's averaged forward Jacobian.
The instrument behind the 2026 global-workspace claims. *Rung 6.*

## Sparse dictionaries

**Sparse autoencoder (SAE)** — trained to reconstruct activations from a large, sparse set of
latents, which turn out far more monosemantic than raw neurons. The standard tool for pulling
features out of superposition. *Rung 5.*

**Decoder direction** (`W_dec[f]`) — the direction feature *f* writes into the residual stream.
Logit-lensing it tells you what the feature *means* without needing any activation data.

**Transcoder** — a sparse replacement for an MLP; **cross-layer transcoders** are what
attribution graphs are built from. *Rung 7.*

**Autointerp** — automatically generating and scoring natural-language explanations of
features.

## Training dynamics

**Phase change** — an abrupt transition during training where a capability appears. Induction
heads form in one; in Pythia-160m it lands between step 512 and step 1000, in the same interval
for 10 random seeds and 4 model sizes. *Rung 4.*

**In-context learning (ICL) score** — how much better the model predicts later tokens than
earlier ones; the macroscopic signature that tracks induction-head formation.

**Checkpoint sequence** — intermediate saves during training. Pythia (154 per model) and OLMo
publish them, which is what makes "when does this property emerge?" answerable at all.

## The recent claims (rung 8)

**Global workspace** — a broadcast subspace posited to underlie conscious access in humans;
claimed to have an analogue in language-model representations.

**Introspection** — a model reporting on its own internal states. Distinguish genuine
introspection from an **injected** state being read back out.

**Society of thought** — the claim that a model contains multiple interacting internal
"voices", argued from SAE-feature steering.

**Metacognition** — a model's sense of what it knows and whether it is right; split into a
**covert** internal signal and **reportable** self-assessment, which can emerge at different
times in training.
