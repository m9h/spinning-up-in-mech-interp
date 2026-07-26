# How to read an interpretability paper

Interpretability results are unusually easy to believe. They arrive with a picture, a
compelling name, and a mechanism story that fits. This is a checklist for reading one the way
you would read a drug trial — not cynically, but with the questions that separate a finding
from a decoration.

Use it on the papers behind each rung. Then use it on ours.

---

## The six questions

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

### 6. Would it survive being run again — by someone else, on something else?

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
against a random-direction null.

**Attribution graphs (2025).** The authors themselves document unfaithfulness — graphs that are
clean, causal-looking and wrong. Read the faithfulness validation before the case studies.

**The 2025–26 property papers** (global workspace, introspection, societies of thought). These
make the largest claims on the least accessible systems. Apply questions 3 and 4 hardest here,
and notice when a striking result is an *injection* being read as a discovery.

---

## The discipline in one line

**Run the control that could kill your own result — and report it when it does.**

Everything in this repo that looks like a finding was, at some point, a number we had to go
back and re-measure. Rung 1's headline inverted once because of a stimulus-frequency choice.
Rung 5's null "worked" until the metric was fixed. That is the normal condition of this work,
not a sign of doing it badly.
