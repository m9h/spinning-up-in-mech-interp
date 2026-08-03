# The ecosystem — what is live, what died, and what is new

Curricula rot at the links. This file is the map: every external tool, venue and community this
curriculum depends on or points at, with its **status as of August 2026**, plus the directions
that opened up after the rungs were written.

Checked by fetching each URL. If you find one broken, open an issue — that is a contribution.

---

## 1. Dead, dying, and what replaced it

| what | status | use instead |
|---|---|---|
| **OpenAI Microscope** (`microscope.openai.com`) | **503 since early 2025.** Took the visual layer of *Zoom In!* with it | **partially replaced — see §2.1.** Monitor covers the language case; **nothing covers vision** |
| **`circuits.pub`** | does not resolve | [`transformer-circuits.pub`](https://transformer-circuits.pub/) (**live**, Anthropic's venue) and [Distill *Circuits*](https://distill.pub/2020/circuits/) (**live**, but Distill itself is on indefinite hiatus since 2021 — an archive, not a venue) |
| **ARENA deep links** (`learn.arena.education/.../13_sae_intro/`) | **404** — ARENA restructured; SAE material is now §1.3.1/1.3.2 | link the chapter, not the section: [arena.education/chapter1](https://www.arena.education/chapter1) |
| **Distill** as a place to publish | on hiatus | there is no successor venue. Circuits work now appears on arXiv, `transformer-circuits.pub`, lab blogs, and **BlackboxNLP** (the ACL workshop that ran the MIB shared task) |

**Lesson worth internalising:** the *Zoom In!* atlas was an argument made in pictures hosted on a
service that went offline. That is why [rung 1](rung1_features_and_circuits/) recomputes the
numbers rather than linking the visuals, and why our artifacts are on the Hub.

---

## 2. Browse-and-steer tools

### 2.1 What "successor" can and cannot mean here

Microscope did **three separable things**. Grading a replacement means grading each:

| function | replaced? |
|---|---|
| browse precomputed feature visualizations for **vision** nets (InceptionV1, AlexNet, ResNet…) | **No. Nothing does this.** Monitor is text descriptions of a language model — different modality, different models, different method. `lucent` is the only route left and it was **last pushed 2025-03-21** |
| a **canonical shared reference**, so a paper can cite "unit 4b:373" and you can go look | language models only |
| an **on-ramp** — click around, build intuition | yes, and improved: search, activation linting, steering |

**So the vision gap is open, not filled.** That is an argument for
[rung 1](rung1_features_and_circuits/) mattering more, not less: it recomputes the numbers from
`torchvision` weights that are not going anywhere, rather than depending on a hosted service.

### 2.2 The tools

| tool | scope | what it gives you |
|---|---|---|
| **[Transluce Monitor](https://monitor.transluce.org)** — code [`TransluceAI/observatory`](https://github.com/TransluceAI/observatory) | **~917,000 MLP neurons of Llama-3.1-8B**, all described | live activations, an "AI linter" that clusters unexpected activations as candidate spurious cues, and **natural-language steering** of neuron clusters |
| **[Neuronpedia](https://neuronpedia.org)** | Gemma Scope, GPT-2, OLMo-3 and more; SAE features | feature browsing, dashboards, steering, an [API](https://www.neuronpedia.org/api-doc), and [SAEBench](https://www.neuronpedia.org/sae-bench/info) |

⚠️ **Both invite the failure mode [rung 5](rung5_sparse_autoencoders/) is about.** An auto-generated
description over 917k neurons is a *hypothesis per neuron*, produced at a rate no human verified.
Browse → read the label → steer → be impressed is not evidence. Which is why:

**[Delphi](https://github.com/EleutherAI/delphi)** (EleutherAI, formerly `sae-auto-interp`) is the
missing tool this curriculum kept gesturing at. It generates explanations of SAE/transcoder
features **and then scores them**, two ways:

- **detection** — given the explanation, can a model predict whether a *whole sentence* activates the feature?
- **fuzzing** — same question at the level of *individual highlighted tokens*.

Both are cheap and scalable. If you are going to trust a feature label, this is how you check it.
See also *Evaluating SAE interpretability without explanations* ([2507.08473](https://arxiv.org/abs/2507.08473)).

---

### 2.3 How these recommendations were checked — and how they were not

Every tool below was checked for **liveness and maintenance** (URL fetch, last push, archive flag,
issue count) in August 2026. Signals as measured:

| repo | last push | stars |
|---|---|---|
| [`EleutherAI/delphi`](https://github.com/EleutherAI/delphi) | 2026-07-27 | 270 |
| [`timaeus-research/devinterp`](https://github.com/timaeus-research/devinterp) | 2026-04-23 | 147 |
| [`TransluceAI/circuits`](https://github.com/TransluceAI/circuits) | 2026-04-10 | 36 |
| [`TransluceAI/observatory`](https://github.com/TransluceAI/observatory) | 2026-03-16 | 251 |
| [`decoderesearch/SAELens`](https://github.com/decoderesearch/SAELens) | 2026-07-28 | 1492 |
| [`decoderesearch/circuit-tracer`](https://github.com/decoderesearch/circuit-tracer) | 2026-07-18 | 2882 |
| [`aaronmueller/MIB`](https://github.com/aaronmueller/MIB) | 2025-08-15 | 26 |
| [`greentfrapp/lucent`](https://github.com/greentfrapp/lucent) | **2025-03-21** | 664 |
| [`AlignmentResearch/tuned-lens`](https://github.com/AlignmentResearch/tuned-lens) | **2025-08-07** | 607 |

⚠️ **This is maintenance evidence, not fitness evidence.** None of these tools has been *run* as
part of this curriculum. The rungs are verified because `verify_all.sh` executes them and checks
the control; the external tools here have no such gate. Hold them to the standard this curriculum
holds papers to — **run it and see whether the control passes** — before trusting a result you get
out of one. Reporting back that one of them fails to reproduce is a contribution.

**Watch for relocations, too.** SAELens moved from `jbloomAus/` and circuit-tracer from
`safety-research/`, both to `decoderesearch/`; old links redirect but are stale. `tuned-lens` is
still listed by EleutherAI, while the maintained copy lives under `AlignmentResearch` and has not
been touched in a year.

## 3. ★ Benchmarks — adjudication you can enter rather than run yourself

This is the biggest thing that changed since the rungs were written. Interpretability now has
**standing benchmarks with held-out private test sets**, which is a different and stronger
epistemic object than a control you run on your own claim.

### MIB — Mechanistic Interpretability Benchmark
[arXiv 2504.13151](https://arxiv.org/abs/2504.13151) (ICML 2025) · [github.com/aaronmueller/MIB](https://github.com/aaronmueller/MIB)

Two tracks over 4 tasks and 5 models, with **HuggingFace leaderboards and a private test set** —
you upload a circuit or a featurizer and it is scored on data you cannot see.

- **Circuit localization** — methods that find the components and edges driving a behaviour (attribution patching, information-flow routes).
- **Causal variable localization** — methods that featurize a hidden vector (SAEs, DAS) and align features to a task-relevant causal variable.

**Its headline results are controls, and they sting:**

> attribution and mask optimization methods perform best on circuit localization … for causal
> variable localization, the supervised DAS method performs best, while **SAE features are not
> better than neurons**.

Note the convergence: MIB found *"SAE features are not better than neurons"* in 2025 by
benchmarking, and Transluce found the same thing in 2026 by building better neuron circuits
([2601.22594](https://arxiv.org/abs/2601.22594), see [rung 7](rung7_attribution_graphs/)) —
**independent methods, same conclusion.** Two independent routes to a negative result is about as
strong as evidence gets in this field.

Also ran as the [BlackboxNLP 2025 shared task](https://aclanthology.org/2025.blackboxnlp-1.32/).

### Others worth knowing
- **AObench** — evaluation suite for activation oracles (§4), open-sourced.
- **AuditBench** ([2602.22755](https://arxiv.org/abs/2602.22755)) — evaluates alignment *auditing* techniques on models with deliberately hidden behaviours. A positive control for auditing itself.
- **SAEBench** ([writeup](https://www.neuronpedia.org/sae-bench/info)) — carries randomly-initialized-model and PCA baselines, the honourable exception noted in rung 5.

**If you are new and want a real target:** pick a MIB track. Your method is scored against a
private set alongside everyone else's, which is the one thing a self-run control cannot give you.

---

## 4. New technique families the rungs do not cover

### Activation oracles
A finetuned LLM that takes the *subject model's residual-stream activations* as input and answers
natural-language questions about them. The lineage our companion `tri-lens` work sits in.

- *Building Better Activation Oracles* ([2606.02609](https://arxiv.org/abs/2606.02609)) — on-policy rollouts, multi-layer input, a fixed injection formula, plus **AObench**.
- ★ *When Activation Oracles Learn Not to Read: Concept-Specific Blind Spots in Fine-Tuned Oracles* ([2607.23379](https://arxiv.org/abs/2607.23379)) — **the control paper.** Readers can learn *their own reporting policy*, including concept-specific omissions induced by the training setup. The consequence is the sentence to remember: **evaluating an interpretability tool requires more than checking whether the information is represented in the subject model.** The tool can be silent about something it can see.
- ARENA now ships demo notebooks and model-diffing exercises for these.

This is the same disease as an unfaithful attribution graph ([rung 7](rung7_attribution_graphs/)),
in a new organ: a readout that is *accurate when it speaks* and *systematically quiet* elsewhere
looks excellent under every metric that only scores what it said.

### Model organisms — and their lottery
*The Model Organism Lottery* ([2607.01033](https://arxiv.org/abs/2607.01033)) finds that
conclusions drawn from model organisms **depend strongly on the training methodology used to
create them**. Anyone who builds a small model to exhibit a phenomenon and then interprets it is
exposed. Read it before you trust a result derived from a purpose-built organism — including
several in this curriculum's source papers.

### Developmental interpretability
Structure emerges in **phase transitions** over training, and you can detect them from the loss
geometry rather than from behaviour.

- **[Timaeus](https://timaeus.co/)** — singular learning theory, the **local learning coefficient (LLC)**, library [`devinterp`](https://github.com/timaeus-research/devinterp) (released April 2026), Discord, open seminars.
- *Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient* ([2410.02984](https://arxiv.org/abs/2410.02984)) — rLLC applied per-head on a two-layer attention-only transformer; induction heads and a previously unidentified multigram circuit.
- **EleutherAI's "Interpreting Across Time"** — the same axis from a different direction.

Relation to [rung 4](rung4_induction_heads/): we measure the induction phase change *behaviourally*
across Pythia checkpoints, seeds and sizes, with nulls. rLLC derives the transition from the
weights. **Neither subsumes the other, and nobody has run them side by side** — which makes that a
genuinely open, tractable project.

---

## 5. Communities that will take a contribution

The MedARC model — an open-science community with a reading group as the front door and real
collaborative research behind it — exists here, in pieces:

| community | shape | how to contribute |
|---|---|---|
| **[EleutherAI](https://www.eleuther.ai/about/)** | Discord → collective → nonprofit; built Pythia, the Pile, `lm-evaluation-harness`, `tuned-lens`, Delphi | Discord; runs a *Summer of Open AI Research* mentorship programme |
| **[Timaeus](https://timaeus.co/projects)** | nonprofit + Discord + open seminars | ★ a **public project board with unclaimed projects**, tagged by difficulty. Protocol: contact the lead on Discord before starting an in-progress one |
| **[Transluce](https://transluce.org/)** | SF nonprofit, open tooling (Monitor, Docent, ADAG) | open-source repos |
| **[Neuronpedia](https://neuronpedia.org)** | open platform | hosts external lenses and SAEs; has an open-problems list |
| **[Open-R1](https://huggingface.co/blog/open-r1)** (HF) | staged open reproduction of a reasoning model | the reproduction model, at scale |
| **MIB / BlackboxNLP** | benchmark + ACL workshop | submit to a leaderboard |

**Unclaimed Timaeus projects that overlap this curriculum directly** (difficulty as they list it):

- *Toy Models of Superposition* — classify additional transitions — **Easy**. This is [rung 3](rung3_superposition/), which already ships with a random-dictionary null.
- *LLCs and Ablations* — compare weight-based LLC estimation against activation ablations — **Medium**. A method-agreement study; the discipline of [rung 7](rung7_attribution_graphs/).
- *Review of Complexity Measures* — compare effective-dimensionality notions across models — **Hard**. Beware the trap we hit in our own J-space work: a *difference* matrix looked strikingly low-rank until we checked that the matrix it came from **was already low-rank**, so most of the effect was inherited. Any comparison of dimensionality measures needs that control.
- *Development of Vision Circuits* — using the Distill circuits thread as the source — **Hard**, in progress, no lead listed. This is [rung 1](rung1_features_and_circuits/) plus a training axis.

---

## 6. Still-live foundations

[ARENA](https://www.arena.education/chapter1) · [Neel Nanda's tutorials](https://www.neelnanda.io/mechanistic-interpretability) ·
[200 Concrete Open Problems](https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability) ·
[Karpathy, Zero to Hero](https://karpathy.ai/zero-to-hero.html) · [RLHF Book](https://rlhfbook.com) ·
[transformer-circuits.pub](https://transformer-circuits.pub/) · [Distill Circuits](https://distill.pub/2020/circuits/) (archive)
