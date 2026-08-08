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

✅ **One gate now exists.** [`tools/gate_autointerp.py`](tools/gate_autointerp.py) runs in
`verify_all.sh` and holds the *idea* behind Delphi's `detection` scorer to account — does a
feature's label predict where it actually fires? — computed exactly, on CPU, in 9 seconds, rather
than via the 70B vLLM explainer Delphi defaults to. **Building it immediately found a live bug in
rung 5** (PITFALLS #24): the steering coefficient had been calibrated on GPT-2's attention sink
and was ~28× too large. That is what a gate is for.

⚠️ **For everything else this is maintenance evidence, not fitness evidence.** No other tool here
has been *run* as part of this curriculum. The rungs are verified because `verify_all.sh` executes them and checks
the control; the external tools here have no such gate. Hold them to the standard this curriculum
holds papers to — **run it and see whether the control passes** — before trusting a result you get
out of one. Reporting back that one of them fails to reproduce is a contribution.

**Watch for relocations, too.** SAELens moved from `jbloomAus/` and circuit-tracer from
`safety-research/`, both to `decoderesearch/`; old links redirect but are stale. `tuned-lens` is
still listed by EleutherAI, while the maintained copy lives under `AlignmentResearch` and has not
been touched in a year.

## 3. ★ Adjudication — the field has asked for this in writing

### The call
***Make Mechanistic Interpretability Auditable: A Call to Develop Guidelines via Continuous
Collaborative Reviewing*** ([arXiv 2606.00033](https://arxiv.org/abs/2606.00033), Lan, Oozeer,
Bandi, Quirke, Meek, Barez & Abdullah — **ACL 2026**) argues that MI has **no standardised system
for auditing experiments**, so its findings go unused in safety-critical settings where nobody can
verify them. Its motivating example is worth sitting with: two papers reached conflicting
conclusions about the same behaviour, and a third found **both were partially correct but
incomparable, because their methods were inconsistent.**

It proposes three things:

1. a **continuous collaborative reviewing platform** for the meta-science that does not fit in
   papers — critiques, **negative findings**, reproductions, partial results;
2. **expert-verified guidelines** generalised from what accumulates there;
3. **source-based auditing** — dependency chains showing which claims hold up which other claims.

It is a **position paper**: none of it is built, and the authors explicitly invite debate on
implementation. Several are at **[Martian](https://withmartian.com/prize)**, which runs a **$1M
interpretability prize** awarding completed work.

**If you want to know why this curriculum ships a null with every technique, that paper is the
field's own answer, and it is asking for help.**

### The rarest artifact is a published ablation

Weights are common now. What makes a claim *checkable* rather than merely reported is the layer
underneath — and Ai2 publishes it as standard: *"training data, intermediate checkpoints, data
mixes, evaluations, **ablations**, and demos."* Almost nobody else ships ablations.

It matters because an ablation table is where a headline goes to be tested. See the COS-PLAY entry
in §4: the abstract says an 8B beats frontier models, and the ablation table says every partial
configuration lands at or below the base. **You cannot adjudicate what was not published**, and
most labs publish the headline without the table.

⚠️ Counter-caveat on the same announcement: "more new artifacts each year than any other
organization" is a **volume** claim, from Hugging Face's own heatmap, in a release co-authored by
both parties. Volume is not reproducibility. Converting the first into the second is exactly the
work this file argues is unowned.

### Benchmarks — adjudication you can enter rather than run yourself

Interpretability now has **standing benchmarks with held-out private test sets**, which is a
different and stronger epistemic object than a control you run on your own claim.

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

### ★ The dead salmon, and interpretability-as-statistical-inference
[*The Dead Salmons of AI Interpretability*](https://arxiv.org/abs/2512.18792) (Méloux et al., 2025)
imports the field's most famous cautionary tale — a **dead fish** showing significant fMRI
activation to social stimuli under standard analysis — and shows the AI analogue is everywhere:
feature attribution, probing, sparse autoencoding **and causal analyses** all yield plausible
explanations of **randomly initialized networks**.

The reframing is the useful part: an explanation is a **parameter of a statistical model inferred
from computational traces**, so it needs alternative hypotheses, identifiability, and quantified
uncertainty — not a picture and a name. This is the closest thing the field has to a statement of
why [READING_A_PAPER.md](READING_A_PAPER.md) exists. A reply,
[*Resurrecting the Salmon*](https://arxiv.org/abs/2508.09363), argues the case is overstated for
domain-specific SAEs; the dispute is live and is itself a good adjudication target.

### ★ Agents — the gap in this curriculum, and in the field

**Every method named in this file assumes a single forward pass.** Attribution graphs, SAEs,
lenses, probes, activation oracles, circuit tracing — all of it takes one prompt and looks inside
one pass. So do all eight rungs here.

Deployment moved. Coding agents run for hours across hundreds of tool calls, and the toolkit did
not follow.

**And this is measured, not predicted.** ARC-AGI has already run the experiment three times:

| | what moved | what it measures |
|---|---|---|
| **ARC-AGI-1** | saturated at **98%** for $0.52/task, against a **$17** human baseline | harness on a *frozen* 8B model: **+53.3 pts** ([2505.07859](https://arxiv.org/abs/2505.07859)). Base-model swap 3B→8B: **+3.4** |
| **ARC-AGI-2** | API **92.5%** vs compute-capped Kaggle **~24%** | **68 points on identical tasks**, from budget and scaffold alone |
| **ARC-AGI-3** | **<1% (Mar 2026) → ~30% (Jul)** | the benchmark went **interactive and agentic** |

The tell is in the name: ARC-AGI-3's first milestone was won by Tufa Labs' **"Duck Harness"** — a
small open LLM writing Python in a live REPL. The winning entry is a harness, and is called one.
ARC Prize's own 2024 report: *"there does not exist any static inference-style transduction
solution that scores above 10%."*

⚠️ **Do not over-read this.** It is a claim about *where score movement comes from*, strongest for
small models under a compute cap. The 2026 frontier also delivered very large genuine model-axis
gains. But it does license one uncomfortable observation:

> **Every technique in this curriculum studies the model. On the benchmark where this has been
> measured most carefully, the model is the minority term.**

The honest question, which nobody has a good answer to:

> **What does a control even look like for a claim about an agent's reasoning over a hundred
> steps?**

A steering result you can check with a random direction of matched norm has no obvious analogue
when the "behaviour" is a trajectory. Neither does a mismatch null, or a median-cell baseline.

**The gap is not unoccupied — it is occupied by psychologists.** *AI Agent Behavioral Science*
([Nature HSSC, 2026](https://www.nature.com/articles/s41599-026-07316-7)) proposes systematic
behavioural observation, interventions to test hypotheses, and theory-guided interpretation of how
agents act and adapt **over time**. *Machine Psychology* describes itself as "moving beyond
architectural interpretability by treating models as experimental subjects." They have the
longitudinal methodology this curriculum's techniques lack. They also inherit a set of hard-won
cautions the AI literature is mostly skipping.

**★ The one that matters most.** The classical method for studying long-horizon reasoning is
**Ericsson & Simon's protocol analysis** — *Verbal Reports as Data*. Agent tooling is reinventing
it now: [AgentTrace](https://arxiv.org/html/2602.10133v1) is "the first open standard for
structured agent logging," spanning cognitive, operational and contextual traces. What is not being
reimported is the thirty years of work on **when a verbal report is admissible evidence at all**:

> **Chain-of-thought is a concurrent verbal protocol, and "unfaithful chain-of-thought" is
> [Nisbett & Wilson (1977)](https://doi.org/10.1037/0033-295X.84.3.231) rediscovered** — *Telling
> More Than We Can Know*, in which people confidently report reasons that had no causal role in
> their behaviour.

Protocol analysis distinguishes **concurrent from retrospective** report, quantifies **reactivity**
(verbalizing changes the process being reported), and specifies the conditions under which a report
*tracks* a process rather than *narrating* one. That is the missing validity theory for every
interpretability claim built on an agent's stated reasoning — and it is a solved problem in another
discipline. Same lesson as the dead salmon, different method: **the field this one is borrowing
constructs from already paid for the controls.**

**★ A worked case, and a correction we are leaving visible.** Meta's CICERO reached human-level
play at *Diplomacy* — a game whose entire premise is negotiation
([Science, 2022](https://www.science.org/doi/10.1126/science.ade9097)). Then
[*More Victories, Less Cooperation*](https://aclanthology.org/2024.acl-long.672/) (Wongkamjan et
al., ACL 2024; UMD / Princeton / Sydney / USC ISI) assessed the communication itself: 24 games,
200 human-player hours, ~27,000 messages, **Abstract Meaning Representation** used to map stated
intentions onto actual moves, plus human annotation of perceived deception (**318** messages
annotated as lies, **1,167** perceived as lies). Their conclusions:

> Cicero's communication is *"more transactional, relying on its optimal strategy rather than the
> alliance building which is the hallmark of top human players"* — humans reliably identify it, and
> it is **less** deceptive and persuasive than they are.

And they state the attribution question openly: *"it is unclear if Cicero's success is due to its
use of natural language or its strategic model."*

That is a **verbal-report validity study with an independent behavioural record** — the moves —
which single-prompt chain-of-thought work does not have. Of the 13 papers citing it (Semantic
Scholar, Aug 2026) **zero** mention faithfulness, chain-of-thought, verbal reports or
interpretability. The two literatures run in parallel. Connecting them is a good first project.

⚠️ **What this file said two days ago, and why it was wrong.** We described the paper as running an
**ablation** — communication switched off, score barely moves. **There is no such experiment.** That
came from a university press page's paraphrase, and reading it as an ablation was our inference on
top of it. Worse: when we finally fetched the PDF, an automated summariser told us the *opposite* —
that removing communication sharply *reduced* the win rate. Also absent.

**Two secondary sources, two contradictory wrong answers, one primary text.** A summary that
agrees with your prior is not evidence, and neither is one that contradicts it. This is
[PITFALLS](PITFALLS.md) territory and it happened here, in the file that tells you to check things.

Where to start if you want to work on it:

- **[ARES](https://github.com/withmartian/ares)** (Martian, open source, active) — RL-first
  infrastructure for training coding agents, built partly to support interpretability of
  *sequential decision-making*. Its central design choice — *"the LLM itself is the agent, not
  the scaffolding"* — is worth arguing with: it draws the model/harness boundary in a specific
  place, and where you draw it determines what your interpretability claim is even about.
- **[COS-PLAY](https://github.com/wuxiyang1996/COS-PLAY)** ([arXiv 2604.20987](https://arxiv.org/abs/2604.20987),
  **MIT**) — six game environments, an LLM decision agent plus a skill-bank agent that turns
  rollouts into reusable "skills." **The clearest harness-over-model table currently available:**
  an 8B base scores 379.6 average reward alone, **924.4** with the harness, against GPT-5.4's
  717.4. A harness moving an 8B from bottom to top of a frontier leaderboard.

  ⚠️ **Read its ablation table before believing it.** Every partial configuration lands at or
  *below* the base model — SFT+final-skill **359.5**, GRPO+1st-skill **305.2**, base **379.6** —
  while the full system doubles the best of them. Their own words: "no single component is
  sufficient." An interaction effect with **no main effects** is either genuine synergy or
  something the ablations do not isolate, and at 16 rollouts per condition with public MIT code it
  is cheap to check. **This is a good adjudication project.**

  ⚠️ **And the half the write-ups lead less with:** on multi-player social reasoning the same
  harness *loses* — Avalon win rate 39.0 vs GPT-5.4's 65.0; Diplomacy 2.96 supply centres vs 4.70.
  So harness gains are large **and domain-bounded**: they transfer where skills are reusable
  procedures and fail where the task is social and adversarial. Where that boundary sits is
  unmeasured.
- **[Terminal-Bench / Harbor](https://github.com/harbor-framework/terminal-bench)** — the task
  format ARES consumes, and the place to contribute an environment.
- **[AstaBench](https://arxiv.org/abs/2510.21652)** (Ai2, ICLR 2026) — **2,400+ scientific research
  problems** across literature understanding, code execution, dataset analysis and end-to-end
  discovery workflows, with the eval toolkit **and a collection of baseline agents released open
  source**. Framed explicitly as *rigorous* benchmarking of agents, which makes it the natural
  companion to [ABC](https://github.com/uiuc-kang-lab/agentic-benchmarks) (§3) — two independent
  efforts at agent-benchmark rigour, and **nobody has applied ABC's failure taxonomy to AstaBench.**
  That is a cheap, concrete adjudication project. *(Evidence tier: search summaries, paper unread —
  see the caveat in §2.3.)*
- ⚠️ **Check licences before building.** Not everything in this space is open: the
  `GoodStartLabs/AI_Diplomacy` harness ships under a non-commercial licence that also forbids
  "extracting moves, outcomes, user interactions, or simulation data" — because for that company
  the extracted data *is* the product. Its vendored game engine, however, is
  [`diplomacy`](https://github.com/diplomacy/diplomacy) (Paquette et al., 2019, **AGPL-3.0**),
  which is separately and genuinely open. Knowing which layer you are standing on matters.
- **SWE-Bench Verified**, evaluable in ~20 minutes with remote sandboxing, which makes the
  iteration loop cheap enough to actually study.

This is the largest uncovered surface in the field and the most honest thing this curriculum can
say about it is that it does not cover it either.

### ★ Program synthesis — the other half of the salmon

The dead salmon gives you a **null**: a randomized network, where a good method must find nothing.
It does not give you the other side. **[Tracr](https://arxiv.org/abs/2301.05062)** (DeepMind, 2023)
does: it *compiles* a human-written **RASP** program into transformer weights, so the mapping from
weights to function is known by construction.
**[TracrBench](https://arxiv.org/abs/2409.13714)** is a dataset of them.

| substrate | what a sound method must do |
|---|---|
| randomized network | find **nothing** |
| **Tracr-compiled model** | find **exactly the compiled program** |

Together those bracket a method. This is [question zero](READING_A_PAPER.md) — *do you have a
positive control?* — and the field has had the answer since 2023 and uses it thinly. (Timaeus lists
*LLCs of Compiled Neural Networks (TRACR)* as an **unclaimed** project.)

**And program synthesis is where capability actually moved.** On ARC the winning family is not
"models that output answers" but **models that output programs, filtered by execution**:

- Greenblatt sampled ~8,000 programs at t=1.0 and selected by whether they **reproduce the training
  pairs** → 50% on the public eval (ARC Prize verified 42–43%).
- BARC runs the *same model* two ways — programs at **t=0.8** with execution filtering, grids at
  **t=0** with beam search — and ensembling adds **+13.75**, because the two families solve
  *different* tasks.
- SOAR weights execution accuracy over grid vote-count by a factor of **1000**.
- ARC-AGI-3's first milestone went to a **small open LLM writing Python in a live REPL**.

The common element is not "programs." It is **verifiable selection** — keeping what *executes
correctly* rather than what is popular. Contrast [rung 5](rung5_sparse_autoencoders/), where the
only thing standing between a feature label and a story is a null you remembered to run.

**Why this matters for interpretability specifically.** A program is an interpretable artifact by
construction: read it, run it, test it — no reverse-engineering required. And execution filtering
comes with something no interpretability method currently has — **a measured false-discovery
rate**. Greenblatt found ~**9%** of programs that pass the training pairs are nonetheless wrong,
and voting removes about half of those. Ask what the equivalent number is for an attribution graph
or an SAE feature label. Nobody can tell you.

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
- **OLMo-3 (Ai2)** is the substrate that makes this practical on real models rather than toys: base
  *and* post-trained variants, the public training data, and **~1,486 intermediate checkpoints**. As
  of August 2026 Ai2's Hugging Face storage went to ~2 PB with **standard Hub rate limits removed**,
  so multi-checkpoint downloads run at full throughput — which was the actual tax on doing
  trajectory work, not compute.

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
| **[Cohere Labs Open Science Community](https://labscommunity.cohere.com/)** | ★ the closest thing to a **MedARC for ML**: application-gated Discord, a **Summer School** (speakers from INRIA, Meta FAIR, DeepMind), a community HF org, and a paid full-time **Scholars Program** | *"led by the interests of its members, with active community members **proposing and leading programs**"* — contributing a module is a defined path, not a cold pitch |
| **MIB / BlackboxNLP** | benchmark + ACL workshop | submit to a leaderboard |
| **[Martian](https://withmartian.com/prize)** | company (LLM routing) with a serious interp arm; co-authors of the ACL 2026 auditability call | runs a **$1M interpretability prize** — grants for directions *and* awards for completed work, currently between rounds — and hackathons with **Apart Research** |
| **[NDIF / `nnsight`](https://github.com/ndif-team)** | **$9M NSF** national infrastructure (Bau Lab); remote execution on open-weight model internals | 110+ papers; free academic access |
| **[Decode Research](https://www.decoderesearch.org/)** | the nonprofit that now maintains **Neuronpedia + SAELens + circuit-tracer + SAEDashboard** | "always looking for new partners" |

**★ A curriculum-shaped hole.** The Cohere Labs Summer School's 21 sessions span ML math,
transformers, computer vision, embeddings, world models, RL, distributed training, agentic AI and
evaluations — and contain **no mechanistic interpretability at all**, and no controls thread. Three
sessions are adjacent enough that the audience is primed: *Test-Time Scaling Small LMs to o1 level*
(harness-over-model, in their own curriculum), *A Temperature Check on Web Agents*, and
*Understanding Transformers via N-gram Statistics*. Their format is talks; this curriculum is eight
runnable rungs that need no GPU, no HF token and under 25 seconds each. **Those are complementary
gaps in both directions.** Rung 3 is the right entry point for a general-ML audience — no
downloads, no prior interpretability.

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
