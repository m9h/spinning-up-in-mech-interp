# The dead salmon — the demonstration this curriculum rests on

**Not a rung. The premise. Run it before rung 1.** ~30 s, CPU, no token.

```bash
cd salmon && python starter.py
```

## The idea

In 2009 a dead Atlantic salmon was placed in an fMRI scanner and shown photographs of humans in
social situations. Standard analysis found brain regions significantly predictive of social
emotion. The fish was dead. The pipeline was the one everyone used.

[*The Dead Salmons of AI Interpretability*](https://arxiv.org/abs/2512.18792) (Méloux, Dirupo,
Portet & Peyrard, 2025) shows the AI version is not hypothetical: **feature attribution, probing,
sparse autoencoding and even causal analyses produce plausible explanations of randomly
initialized networks.** This script is their Figure 1, plus the three controls that turn a
demonstration into a lesson.

## What it does

Take **BERT with random weights** — it has learned nothing — embed 500 IMDb reviews, mean-pool,
train a cross-validated linear probe on sentiment.

## What you should see

| | accuracy | |
|---|---|---|
| **1.** random net, real labels | **60.6%** | the artifact |
| **2.** random net, shuffled labels | **47.6%** (5 shuffles, 44.0–50.6) | null (a) — CV integrity |
| **3.** bag-of-words → random projection | **57.2%** | null (b) — no transformer at all |
| **4.** pretrained net, real labels | **63.2%** | positive control |

Read together:

- **A probe read sentiment out of a network that never learned anything** — +10.6 points over chance.
- The **shuffled-label null is at chance**, so the cross-validation is sound and (1) is real.
- **Bag-of-words with a random projection gets 57.2%.** So of the 10.6 points, the transformer
  architecture contributed **+3.4** and the rest was *the input*.
- **Pretraining adds +2.6** — *less than random initialization did.*

That last line is the one to sit with. On this probe, **all of BERT's pretraining is worth less
than the architecture's random projection.** The probe was reading the data through the model, not
out of it.

## Why a probe on a dead network works

A randomly initialized network is a **random projection**, and Johnson–Lindenstrauss says random
projections approximately preserve distances. Whatever separates positive from negative reviews in
the *input* survives into the *output*. Nothing was learned; the geometry was already there.

**"I trained a probe and it worked" is a fact about your dataset until you show it is a fact about
your model.**

## Two honest caveats

**Mean-pooled BERT is a weak sentence representation.** That is precisely why Sentence-BERT
exists. So (4) understates what pretraining is worth *in general* — it shows what it is worth
*under the standard recipe someone would reach for*, which is the situation you are actually in
when reading a paper.

**The bag-of-words null is the one that bites, and it is the one usually missing.** Reporting (1)
and (2) alone — "the probe works, and shuffled labels don't" — looks like a rigorous result and
is nearly uninformative. Null (a) validates your pipeline. Only null (b) tells you whether the
*model* had anything to do with it.

## The trap this closes

Every rung in this curriculum ships a null because of the effect above. But note what building
this exposed: **our first version had the shuffled-label null sitting at 43% instead of 50%**,
which we nearly read as a curiosity. It was a bug — the probe's intercept biased predictions
toward the training majority, which is anti-correlated with a complementary test fold. See
[PITFALLS #25](../PITFALLS.md). A null that is *off* in the wrong direction is not a small
problem; it means you do not know where your floor is.

## Next

The salmon supplies the **null**: a good method finds nothing here. Its other half is a **positive
control** — a network whose contents are known by construction, where a good method must find
exactly the planted program. That is [Tracr](https://arxiv.org/abs/2301.05062), and it is the
companion to this script. See [ECOSYSTEM.md §4](../ECOSYSTEM.md).

Then [rung 1](../rung1_features_and_circuits/), where the same discipline is applied to a real
result: a vision feature, measured against a random-init net *and* a weight-shuffled one.
