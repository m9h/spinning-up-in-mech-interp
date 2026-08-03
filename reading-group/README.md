# The Cognitive Hexagon Reading Group

**Read the source before the borrowing.**

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

## Sessions

Each is one foundational reading, one recent AI paper, and one control — a result that
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
- **Session task:** does "verbalizable" in the AI sense name anything a linguist would recognise?

### 7. Anthropology ⇄ AI — the dashed edge that stayed dashed
- Anthropology was the least-connected vertex in 1978 and is nearly absent from interpretability
  now. Yet post-training is *enculturation*: a model acquires a register, a set of refusals, a
  persona.
- **Ours:** post-training moves the model's internal viewpoint ~10× more than reinforcement
  learning does, while task capability stays flat — a change in stance, not competence.
- **Session task:** what would an anthropology of model post-training even measure?

### 8. Methods — telling a faithful borrowing from a name
Synthesis. Working from [`READING_A_PAPER.md`](../READING_A_PAPER.md), build a checklist for
imported constructs: Is the source theory contested in its own field? Is the operationalisation
the source's, or a new one wearing the same word? Would the source discipline's own control
apply — and does the paper run it?

## How to run it

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
