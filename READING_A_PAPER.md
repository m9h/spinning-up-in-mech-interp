# How to read an interpretability paper

Interpretability results are unusually easy to believe. They arrive with a picture, a
compelling name, and a mechanism story that fits. This is a checklist for reading one the way
you would read a drug trial — not cynically, but with the questions that separate a finding
from a decoration.

Use it on the papers behind each rung. Then use it on ours.

---

## Question zero: do you have a *positive* control?

Everything below is about finding the null that could kill a claim. That is necessary and it is
not sufficient, and getting this backwards is the most expensive mistake in this document.

**A null-only design cannot distinguish "the method fails" from "my reproduction is broken."**
Both produce the same output: no effect. And the second is far more likely, because the method's
authors got it working and you have been at it for an afternoon.

So before running any null, establish a **green test**: something your pipeline must reproduce
that the original authors published — a released number, a worked example, a reference output.
Then break your pipeline deliberately and confirm the test *fails*. A green test whose red cases
also pass is measuring nothing.

We learned this by getting it wrong. Reproducing a 2026 method, we ran a mismatch null, found
zero effect, and wrote it up as a finding about the method. It was two bugs in our own code. The
authors' repository shipped a worked example with reference numbers the whole time; running
against it turned the "finding" into a pass. Worse, one of the bugs came from a *correct,
separately verified* premise that we reasoned from to a wrong conclusion — no amount of further
thinking would have caught it, only the anchor did.

Practical form:

- **Green:** reproduce a published number to a stated tolerance.
- **Red:** wrong layer, wrong scale, wrong position, shuffled inputs — each must degrade.
- Only then: run your null on the claim you actually care about.

If no reference number exists, say so explicitly and treat every negative result as provisional.
An unanchored negative about someone else's method is a hypothesis about your own code.

## The seven questions

### 1. What is the claim, as a falsifiable sentence?

Rewrite the paper's headline in a form that could be wrong. "The model has a global workspace"
is not yet a claim. "A subspace exists whose contents predict the model's next output better
than the residual stream at that layer does" is.

If you cannot write the falsifiable version, the paper may not have made one — and that is
worth noticing before you accept it.

### 2. What *kind* of evidence is it?

There is a hierarchy, and papers do not always signal where they sit:

| kind | what it shows | what it does not |
|---|---|---|
| **Visualization** | this component is describable | that the description is right, or that it matters |
| **Correlation** | a score tracks a behaviour | that the component *causes* the behaviour |
| **Intervention** | changing it changes the output | that the effect is *specific* to it |
| **Specific intervention** | and a matched control does nothing | — this is the standard |

An induction *score* is correlational. In our Pythia sweep the highest-scoring heads had
**zero** causal effect at one checkpoint (−0.003) and a decisive one at the next (+9.48). The
score alone would have told you the mechanism existed before it did.

### 3. What is the null — and is it alive?

Ask what the paper's own measurement would return on a system with no learned structure:
random weights, shuffled weights, a random direction of matched norm, an untrained checkpoint,
a permuted label.

Then ask the harder question: **is that null capable of scoring at all?** A null that returns
zero because the comparison system is inert is not evidence. In our vision atlas, 100% of the
last layer's units beat a randomly-initialized network and only 46% beat a weight-shuffled one.
Same units, same probe — the weaker null doubled the apparent success rate.

If a paper reports no null, you can often estimate one yourself in an afternoon. That is what
rungs 1–7 train you to do.

### 4. Is the intervention *specific*?

Two questions retire most steering results:

- Does a **random direction of the same norm** produce the same effect?
- Does the **negation** produce the opposite effect, or the same one?

A real feature raises its own tokens far above the random baseline and its negation lowers
them. "The negation was comparably effective" is a refutation, not a footnote.

### 5. Is it measured where the phenomenon lives, and is the metric well-behaved?

- **Right place**: orientation tuning at the unit's preferred spatial frequency; a signal read
  at the layer it exists in; a token position that carries the information.
- **Right scale**: an intervention within the regime where the model still functions, not one
  that saturates it.
- **Bounded metric**: ratios explode near zero. Prefer normalized indices.
- **Multiple comparisons**: "the best of 144 heads" is a maximum over 144 draws. Compare it to
  the maximum of the null, not the null's mean.

### 6. Is the comparison *within-item* or *between-item*?

When a paper compares two conditions, ask whether the same items appear in both — and whether
the statistic is computed per item and then averaged, or averaged first and then compared.

The two can differ dramatically. One analysis found correct reasoning traces more diverse than
**length-matched** incorrect ones (+0.0110, 1,003 pairs, surviving Bonferroni across four
domains) — apparently solid. Re-run holding the *problem itself* fixed, comparing each problem's
correct traces against its own incorrect ones, it fell to **+0.0023**, with the between-problem
estimate sitting **3.1 SE outside** that interval. The effect was between-problem structure, not
a property of correctness.

This applies to our own work: our ladder comparisons average the Jacobians *before* comparing
them, which weights by magnitude, where the within-item version would compare per prompt and
then average. We have not measured the gap, and say so.

**Ask: does the design hold the item fixed, or only the condition?**

### 7. Would it survive being run again — by someone else, on something else?

- **n = 1?** One model, one seed, one prompt set is a hypothesis.
- **Are the artifacts public?** Not the code — the *outputs*: weights, checkpoints, scores.
  Code that could compute a result is not the result.
- **Does it hold across seeds and scales?** We found the induction phase change lands in the
  same interval for 10 seeds and 4 model sizes — but the *identity* of the head doing the work
  differed in every single seed. A paper naming "the induction head at layer 5" would have
  described one seed's accident as a fact about transformers.

---

## Reading the specific literature this curriculum covers

**Vision circuits (2020).** Argued largely in pictures, with no randomization control anywhere
in the thread. The claims may well be right — but the evidence type is "visualization", and
the quantitative layer was never published. Rung 1 supplies it.

**Induction heads (2022).** Unusually strong: six independent lines of argument including
ablation and an architectural perturbation. Its weakness is availability — the 34 models were
internal, so nobody outside could check it for a decade. Note also the live dispute over
whether induction heads or function-vector heads drive few-shot in-context learning.

**Sparse autoencoders (2023–24).** Strong interpretability evidence; the steering evidence is
where the controls thin out. No major SAE release publishes per-feature steering effects
against a random-direction null — Gemma Scope runs no steering experiments at all and lists
comparing SAE steering to steering vectors as an *open problem* in its own paper; SAEBench has
no steering eval; Neuronpedia's per-feature exports contain no steering measurement. The gap is
acknowledged by the field and unfilled.

**Attribution graphs (2025).** The authors themselves document unfaithfulness — graphs that are
clean, causal-looking and wrong. Read the faithfulness validation before the case studies.

**Emergence claims in RL'd reasoning models** (e.g. *Ring-Zero*, 2026, reporting emergent
self-verification and parallel reasoning at 1T parameters). Ask question 2 first: is the
evidence *behavioural* (the model writes text that looks like self-checking) or
*representational* (a measured internal mechanism)? Those are routinely conflated, and the
second does not follow from the first. Rung 4 shows why it matters — an induction *score* is
correlational, and the top-scoring heads had zero causal role one checkpoint before they had a
decisive one.

**The 2025–26 property papers** (global workspace, introspection, societies of thought). These
make the largest claims on the least accessible systems. Apply questions 3 and 4 hardest here,
and notice when a striking result is an *injection* being read as a discovery.

---

## The discipline in one line

**Verify against a published number first, then run the control that could kill your own result
— and report it when it does.**

Everything in this repo that looks like a finding was, at some point, a number we had to go
back and re-measure. Rung 1's headline inverted once because of a stimulus-frequency choice.
Rung 5's null "worked" until the metric was fixed. That is the normal condition of this work,
not a sign of doing it badly.
