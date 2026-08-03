# The Cognitive Hexagon Reading Group

**Without a reproduction you cannot have the discussion. You can only discuss the paper.**

That is the whole rationale. A paper is a *report about* a system. Until someone rebuilds it on
weights you can open, the system itself is not in the room — and you cannot ask it anything.
You can only ask what the authors chose to tell you, and evaluate the argument on style.

The clearest illustration is the one this project began with. When Anthropic reported that its
model contains a global workspace, it invited commentary from Dehaene and Naccache — the
neuroscientists whose theory it borrows. They could not test it. They proposed six experiments
and observed that Anthropic could run them. **None were run, because the model is closed.** The
most qualified readers in the world were reduced to writing a response. That is what discussion
without reproduction looks like, even at the very top.

So: **we make sure exciting papers have public open-source implementations and training
documentation.** A session is finished when the paper has, in public, a **runnable
implementation** on open weights, **training documentation** someone else can follow, and — the
part usually missing — **the control that could falsify it.**

### What reproduction buys that reading does not

**You get to ask questions the paper did not.** This is the real return, and it is larger than
verification. Almost everything worth having in this project came from a question no paper
asked, which only became askable once something ran:

- *Do two different instruments agree about the same activation?* Neither paper could ask it;
  each has one instrument. They agree 42× above a mismatch null.
- *Does the covert error signal actually help pick better answers?* No. +0.008 over baseline —
  a bound on our own claim that reading could never have produced.
- *Does the reported effect survive a per-layer noise floor?* Half of one number was noise; the
  headline ratio doubled.
- *Does it hold across seeds?* The induction phase change lands in the same interval for ten
  seeds — but a **different head** does the work in every one.

None of those are in any paper. All of them are one afternoon away once the thing runs.

Every large interpretability claim of 2025–26 imports a construct from cognitive science — a
*global workspace*, *introspection*, *metacognition*, a *society of mind*. Most reading groups
read the AI paper and take the construct on trust. This one reads the construct first, from the
discipline that owns it, and only then asks whether the borrowing is faithful.

## The frame

The Sloan Foundation's 1978 report drew cognitive science as a hexagon of six disciplines —
**psychology, linguistics, philosophy, anthropology, neuroscience, artificial intelligence** —
with edges for the interdisciplinary areas between them. Some edges were drawn **solid** (a real,
working connection: psycholinguistics, neurolinguistics) and some **dashed** (a connection hoped
for but not yet made).

That distinction is the organising device. **The AI–neuroscience and AI–psychology edges are
being redrawn right now, at speed, largely by one side.** Each session takes one edge, reads the
foundational text and the 2026 claim together, and asks a single question:

> *Is the AI paper using the construct, or using its name?*

### The failure mode this group must avoid

Reproduction is necessary and it is not self-validating. **We produced a confident false
negative by reproducing.** We reported that a published method failed a mismatch null; it was
two bugs of ours, and the authors' repository had shipped reference numbers the whole time.

A null-only reproduction cannot distinguish *"the method fails"* from *"our rebuild is broken"* —
and the second is far likelier, because the authors got it working and you have been at it for
an afternoon. A group that reproduces without this discipline will generate wrong negatives at
scale, and wrong negatives about other people's work are worse than no reproduction at all.

**So every reproduction here opens with a green test:** something the original authors published
that our rebuild must match, plus deliberately broken variants that must *fail*. Only then does a
null mean anything. See [`READING_A_PAPER.md`](../READING_A_PAPER.md), question zero.

The second-order lesson from that episode is the one worth carrying: one of our two bugs came
from a **correct, separately verified premise** reasoned to a wrong conclusion. No amount of
further thinking would have caught it. Only the published anchor did.

## What a session ships

| output | why it is the point |
|---|---|
| **Open implementation** | on open weights, because the source model is usually closed |
| **Training/method documentation** | the reproduction recipe, including the parameters that are load-bearing and undocumented |
| **A control** | most papers ship the technique without the null; that gap is the contribution |
| **An honest status** | reproduced / refuted / inconclusive-under-control — negatives published |

### Already shipped, as the template

| paper | what was missing | what we published |
|---|---|---|
| Natural Language Autoencoders (2026) | no evaluation suite, no baselines | a red-green harness verified against the authors' own worked example (`fve` 0.839 vs 0.821) — after it caught **two bugs of ours** that had produced a false negative |
| Verbalizable Representations / J-space (2026) | closed model; reviewers could not test it | full replication on OLMo-3, plus the six tests Dehaene & Naccache proposed and nobody had run |
| In-Context Learning and Induction Heads (2022) | 34 **internal** models; per-head scores never released | the first public per-head scores across a full checkpoint sequence, with the causal ablation |
| Distill *Circuits* (2020) | argued in pictures; no numbers, no null, Microscope offline since 2025 | 5,808 units quantified, with two null distributions |
| Societies of Thought (2026) | no code, no data | an adversarial replication — the accuracy gain **reverses** on a second benchmark |

## Session 0 — the spine: at what timescale are you explaining this?

Read before everything else, and return to it constantly.

**Sapolsky, *Behave: The Biology of Humans at Our Best and Worst* (2017).** Its structure is the
method, not the content: take one behaviour and explain it at every timescale in turn — what
happened a second before (neurons), minutes to hours before (hormones), days to months
(plasticity), years (development), centuries (culture), millennia (evolution). No level is *the*
explanation; the behaviour is the intersection.

Interpretability has exactly this problem and rarely names it. When a model emits a token, the
candidate explanations sit at wildly different timescales:

| timescale | the model's analogue | how we study it |
|---|---|---|
| ~now | the forward pass — attention, the residual stream at this token | lenses, patching, ablation |
| this conversation | the context window, in-context learning | induction heads |
| weeks | fine-tuning, RLHF, preference data | the post-training ladder |
| months | pretraining | checkpoint sweeps |
| decades | who wrote the internet, in what language | data provenance — barely studied |
| centuries | the culture encoded in that text | session 7, and almost nobody |

**Why this belongs at the front.** Most disputes in these papers are timescale confusions
wearing other clothes. "Does the model introspect?" mixes a forward-pass question with a
training-history one. Our own results are strewn across the table — an induction phase change at
one timescale, a post-training viewpoint shift at another, and a covert error signal that turns
out to be present in pretraining and made *reportable* weeks later by fine-tuning. That result
is unreadable without the ladder of levels.

**Companion:** Sapolsky, *Determined* (2023), for session 2. If human authorship of our own
reasons is substantially constructed after the fact — which is Nisbett & Wilson's finding with
the philosophy attached — then "the model confabulates its reasoning" stops being a
disqualifying objection and becomes a *comparison*.

*Local note: this syllabus is unusually local. **Stanford** — McClelland (Center for Mind,
Brain, Computation and Technology) and Sapolsky. **Berkeley** — Gopnik (cultural technologies),
Piantadosi (Computation and Language Lab), Olshausen (Redwood Center for Theoretical
Neuroscience), and Steinhardt, whose function-vector-head result is the live dispute cited at
rung 4. The Simons Institute is running a **Special Year on Large Language Models and
Transformers** that several of them are part of. These are conversations available in person
rather than only through papers.*

## Sessions

Eight edges, plus a synthesis. Each session is one foundational reading, one recent AI paper,
and one control — a result that
constrains how much weight the analogy can bear.

### 1. Neuroscience ⇄ AI — the global workspace
- **Source:** Baars, *A Cognitive Theory of Consciousness* (1988); Dehaene & Naccache, *Towards a
  cognitive neuroscience of consciousness* (2001) — ignition, broadcast, the bottleneck.
- **Claim:** *Verbalizable Representations Form a Global Workspace in Language Models* (2026).
- **Control:** the **COGITATE** adversarial collaboration, whose *preregistered* ignition
  prediction was **not confirmed in humans**. The analogy is being borrowed from a theory under
  active challenge in its home discipline.
- **Ours:** the OLMo replication — what survives when you run it on open weights.

### 2. Philosophy ⇄ Psychology — introspection, and why psychology abandoned it
- **Source:** Nisbett & Wilson, *Telling More Than We Can Know* (1977) — people confidently
  report reasons for their own behaviour that demonstrably are not the causes. Plus the collapse
  of introspectionism (Wundt, Titchener) that made behaviourism attractive.
- **Claim:** *Emergent Introspective Awareness in Large Language Models* (2025).
- **Control:** *Can LLMs Introspect? A Reality Check*.
- **The question this session exists for:** psychology spent a century learning that verbal
  self-report is not privileged access. What licenses treating a model's self-report as evidence
  about its internals?

### 3. Psychology ⇄ AI — metacognition, and how it is actually measured
- **Source:** Flavell (1979) on metacognition; Nelson & Narens' framework; **meta-d′ and the
  M-ratio** (Maniscalco & Lau) — the type-2 signal-detection machinery psychology built
  precisely because "confidence tracks accuracy" is too loose to test.
- **Claim:** the 2026 LLM-metacognition survey literature.
- **Ours:** a covert error signal present in the base model, made *reportable* by supervised
  fine-tuning — and the negative that bounds it: it does **not** improve answer selection.
- **Takeaway:** psychology already has the measurement theory. Most AI metacognition papers
  reinvent a weaker version of it.

### 4. AI ⇄ Philosophy — societies of mind
- **Source:** Minsky, *The Society of Mind* (1986); Fodor, *The Modularity of Mind* (1983) — what
  it takes for a "module" claim to have content.
- **Claim:** *Reasoning Models Generate Societies of Thought* (2026).
- **Control:** the replication in which the accuracy gain **reverses** on a second benchmark, and
  the authors' own retraction of the redundancy finding.

### 5. Philosophy ⇄ everything — indicator frameworks and moral status
- **Source:** Butlin, Long et al., *Consciousness in Artificial Intelligence: Insights from the
  Science of Consciousness* (2023) — the indicator-property method, which is explicitly a way of
  making progress *without* settling the metaphysics.
- **Session task:** take three indicators and ask what measurement would actually implement each.
  Most have never been operationalised, which is the gap this group's host project is trying to
  fill.

### 6. Linguistics ⇄ AI — what "verbalizable" means
- The weakest edge, and worth saying so. The 2026 workspace paper turns on *verbalizability* —
  what a representation is "disposed to say" — yet the linguistics of production (Levelt's
  speaking model; the formulation/articulation split) is almost never cited.
- **Reading:** **Piantadosi** (Berkeley, Computation and Language Lab) on what language models
  do and do not settle about human language learning — including the deliberately provocative
  case that they refute the poverty-of-the-stimulus argument. This is the edge where the AI side
  has been *least* willing to engage the discipline it is making claims about.
- **Session task:** does "verbalizable" in the AI sense name anything a linguist would recognise?

### 7. Anthropology ⇄ AI — the dashed edge, and why it should not be
Anthropology was the least-connected vertex in 1978 and is nearly absent from interpretability
now. It should not be, because **training encodes human behavioural preferences and culture, in
two distinct ways**:

- **Pretraining** passively absorbs the distribution of written human text — filtered by who
  writes, in which languages, and what got digitised.
- **RLHF and preference tuning** actively impose a *curated* preference structure, produced by a
  specific annotator pool under specific instructions in a specific institutional setting. That
  is not "human values"; it is *somebody's* values, sampled.

**Reading:** ★ Farrell, **Gopnik** et al., *Large AI models are cultural and social
technologies* ([Science, 2025](https://www.science.org/doi/10.1126/science.adt9819)) — the
argument that these systems are best understood as **cultural technologies** in the lineage of
writing, print and libraries: techniques for transmitting information between people, not
agents. Then Yiu, Kosoy & Gopnik, *Transmission versus truth, imitation versus innovation*,
which sharpens it empirically — children **innovate**, models **transmit**. Then Lambert,
[*RLHF Book*](https://rlhfbook.com) — **Lecture 8, "On 'Preferences' and Preference Data"**, and
the chapter on character training. Then RewardBench 2's decision to
**commission new human prompts** rather than reuse existing ones — a contamination fix that is
also, unavoidably, a choice about whose prompts. Pair with Sapolsky on culture as transmitted
rather than innate (session 0).

**★ Ours, and an interpretation worth arguing about.** The stages differ sharply in how far they
move the model's internal viewpoint, while task capability stays flat:

| stage | what its reward encodes | viewpoint movement |
|---|---|---|
| SFT + DPO | **human preference judgements** | **~29%** beyond noise |
| RLVR (RL-Zero) | **verifiable correctness** (maths, code) | **~3%** |

The stages trained on *human preference* move the model's internal stance an order of magnitude
more than the stage trained on *checkable correctness*. That is at least consistent with the
reading that what post-training installs is largely **cultural rather than epistemic** — and it
is measurable, which is unusual for a claim in this area.

**Caveat, stated with it:** the arms differ in algorithm, data volume and pipeline position as
well as in reward type, so this is a suggestive alignment, not an isolated variable. Designing
the contrast that *would* isolate it is a genuine open problem and a good session output.

**Session task:** what would an anthropology of model post-training measure? Candidate: the same
probe across models tuned by different annotator populations — the cross-cultural design, applied
to models rather than to people.

### 7b. Neuroscience ⇄ AI — sparse coding, the ancestor nobody cites
Sparse autoencoders are treated in the AI literature as a 2023 invention. They are a
**neuroscience idea, reimported after twenty-five years.**

- **Source:** **Olshausen** & Field, *Emergence of simple-cell receptive field properties by
  learning a sparse code for natural images* (1996). Impose sparsity on a code for natural
  images and you recover oriented, localised, bandpass receptive fields — the simple cells Hubel
  and Wiesel found in cortex. Olshausen directs Berkeley's **Redwood Center for Theoretical
  Neuroscience**, which exists for exactly this question.
- **Claim:** *Towards Monosemanticity* (2023) and the SAE literature — impose sparsity on a code
  for *activations* and recover interpretable features.
- **The loop worth noticing:** rung 1 of the technique ladder measures orientation selectivity in
  a vision model — i.e. looks for simple cells. Rung 5 uses sparse coding to find features. The
  method used at rung 5 was invented to explain the thing measured at rung 1. Neither rung says
  so, which is the sort of gap this reading group exists to close.
- **Session task:** what did sparse coding assume about the data that SAEs on activations may not
  inherit?

### 8. Psychology ⇄ AI, done properly — run the paradigm on both
The syllabus above is adversarial by construction: every session pairs a claim with a control
that constrains it. That is a distortion if it is all we read. **There is an existing literature
that does this well, and it should be the standard the others are measured against.**

- **Source + claim in one:** Dasgupta, Lampinen, Chan, Creswell, Kumaran, **McClelland** & Hill,
  *Language models show human-like content effects on reasoning tasks*
  ([arXiv 2207.07051](https://arxiv.org/abs/2207.07051)). Content effects — that believable
  conclusions are endorsed more readily than logically equivalent unbelievable ones — are a
  classic finding from Wason, Evans and Johnson-Laird. The paper takes the *human paradigm*, runs
  it on models, and compares against *human data*.
- **Companion:** *A Systematic Comparison of Syllogistic Reasoning in Humans and Language Models*
  ([arXiv 2311.00445](https://arxiv.org/abs/2311.00445)).
- **Background:** McClelland & Rumelhart, *Parallel Distributed Processing* (1986) — the origin
  of the connectionist program, itself built at the psychology/AI/neuroscience intersection this
  hexagon describes.

**Why this is the methodological anchor.** Most papers that invoke a cognitive construct never
run the human side at all — they borrow the *name* and test only the model. Comparing to human
subjects requires reproducing the paradigm on both, which is the reproduction argument extended:
you need the model runnable *and* the human data available, or the comparison is rhetorical.

**Session task:** take one claim from sessions 1–3 and design its human-subject counterpart. What
would the human version of "the workspace covertly encodes its own errors" actually be? (It
exists: it is meta-d′, and psychology has been measuring it since the 1970s.)

*Local note: McClelland's Center for Mind, Brain, Computation and Technology is at Stanford —
this is a Bay Area conversation that can be had in person rather than only through papers.*

### 9. Methods — telling a faithful borrowing from a name
Synthesis. Working from [`READING_A_PAPER.md`](../READING_A_PAPER.md), build a checklist for
imported constructs: Is the source theory contested in its own field? Is the operationalisation
the source's, or a new one wearing the same word? Would the source discipline's own control
apply — and does the paper run it?

## How to run it

- **Every session names its artifact before it starts.** "What will exist publicly that did not
  exist last week?" If the answer is "a better understanding", the session is under-specified.
- **Find the gap first.** Before implementing, check what is genuinely missing: the substrate
  (weights, SAEs, checkpoints) is usually public, and the *control layer* usually is not. Do not
  re-release what exists; publish what does not.
- **Two texts per session, source first.** The order matters; reading the AI paper first frames
  the construct in its terms.
- **Someone must argue for the borrowing.** These claims are not silly, and a session that only
  debunks has learned nothing.
- **End each session with a measurement, not a verdict:** *what experiment would settle this?*
  Several such experiments in this project began as that question.
- Pairs naturally with the [technique ladder](../README.md) — rung 8 adjudicates the same papers
  empirically, this reads what they are claiming.

*An [Orthogonal Research and Education Lab (OREL)](https://orthogonal-research.github.io/)
activity, hosted alongside the curriculum.*
