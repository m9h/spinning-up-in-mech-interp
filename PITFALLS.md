# Pitfalls

Every one of these cost us a wrong result or a wasted hour while building this repo. They are
not hypothetical. If a rung gives you a number that seems too good, too bad, or too boring,
look here first.

The pattern worth internalizing: **most interpretability bugs do not crash. They return a
plausible number.** That is why every rung ends in a control — a crash you notice, but a
flattering artifact you publish.

---

## 1. Attention weights come back empty (`IndexError` on `out.attentions[l]`)

Since `transformers` 5.x the default attention implementation is SDPA, which does not
materialize attention probabilities, so `output_attentions=True` silently gives you nothing.

```python
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
```

*Hit in rung 2 and rung 4.*

## 2. A forward hook that works on one version breaks on the next

`GPT2Block.forward` returns a bare tensor in recent `transformers` and a tuple in older ones.
A hook that assumes `out[0]` will index into the tensor's first row and produce a
shape error deep inside LayerNorm.

```python
is_tuple = isinstance(out, tuple)
hs = out[0] if is_tuple else out
...
return ((hs,) + tuple(out[1:])) if is_tuple else hs
```

*Hit in rung 5.*

## 3. Steering at an arbitrary magnitude just breaks the model

Adding a direction scaled to some multiple of the residual norm saturates everything
downstream; greedy decoding then emits the same token forever (`ness ness ness…`) and you
conclude the feature is "very strong". It isn't — you left the regime where the model works.

**Calibrate to the feature's own activation scale**: encode real text, take a high percentile
of that feature's non-zero activations, and inject at a small multiple of *that*.

*Hit in rung 5.*

## 4. Your steering metric moves for *any* perturbation

If you score steering by the raw logit of the target tokens, a random direction of matched
norm also "works", and the feature's negation can move the *same* way — because a large
perturbation shifts all logits together.

**Use a specificity score**: target-token logits *minus* a fixed control-token set. The common
shift cancels, the random null goes to ~0, and the negation correctly reverses.

> Rung 5's verified numbers: feature **+6.5**, random null **+1.0**, negation **−13.5**.
> Before the fix, the random null looked like a real effect.

## 5. Measuring away the thing you are looking for (preferred spatial frequency)

Orientation tuning must be measured **at each unit's preferred spatial frequency**. Averaging
responses across frequencies blurs units tuned to one of them. With a frequency range that
didn't reach conv1's preference, InceptionV1's first layer measured as *less* orientation-
selective than a random network — the exact opposite of the truth.

The general form: **probe where the phenomenon lives.** Wrong layer, wrong position, wrong
stimulus scale — all produce confident nulls.

*Hit in rung 1. It briefly inverted our headline result.*

## 6. A dead null is not a null

A randomly-initialized deep network is nearly *inactive* several blocks in — in our
InceptionV1 atlas, 0% of units respond below `inception3a`. "Beats the random-init null" is
then trivially true and means nothing.

Use a null that stays alive. **Shuffling each trained kernel's weights** preserves the weight
distribution and keeps activations in range:

| layer | beats random-init null | beats weight-shuffle null |
|---|---|---|
| `inception5b` | **100%** | **46%** |

Same units, same probe — the weaker control doubles your success rate.

*Hit in rung 1.*

## 7. Ratios explode when the denominator is near zero

"Curved response ÷ straight response" produced selectivity values of **2,000,000** for units
that simply didn't respond to the straight stimulus. Clamping the denominator hides it rather
than fixing it.

Use a **bounded index**: `(a − b) / (a + b)`, which lives in [−1, 1] and degrades gracefully.

*Hit in rung 1.*

## 8. Lower precision can silently change your scores

Running in `bfloat16` to fit a bigger model is normal — but verify it doesn't move the
measurement. On Pythia-160m, bf16 and fp32 agree to ±0.001 through the induction phase change,
yet at the **final** checkpoint 22 of 144 heads shifted by >0.05, the top head fell from 0.945
to 0.337, and even the argmax head changed. On Pythia-1.4b at the same checkpoint, bf16 was
fine (0.940 vs 0.941).

So the sensitivity is **model- and training-stage-specific**; you cannot reason it away. Run
the same checkpoint in both dtypes and compare before trusting a mixed-precision series.

## 9. An attention pattern is not a mechanism

A head can attend exactly where your story predicts and contribute nothing. Always ablate and
measure the behavioural change — **and ablate an equal number of random heads as the baseline.**

> Rung 4: ablating induction heads costs **+6.3** in-context loss; five random heads, **+0.23**.

The stronger version: an induction score is *correlational*. In our Pythia sweep, ablating the
top-scoring heads changed nothing at step512 (−0.003) and cost +9.48 one checkpoint later. The
score existed before the mechanism did.

## 10. Disk fills up during checkpoint sweeps

Every model revision is a fresh download. 154 Pythia-160m checkpoints is ~58 GB if you keep
them. Purge the repo's cache directory after each checkpoint (the sweep script does this).

## 11. Batch size changes your error bars, not your answer

Shrinking the batch to fit memory is fine — the estimate stays unbiased, the variance grows.
Compensate with more stimulus seeds and **record batch size and seed count in your output**,
so a later comparison isn't confounded by measurement noise you forgot about.

## 12. Check that your script actually ran

While timing these rungs we recorded suspiciously fast runtimes — the environment was missing
`transformers`, so three rungs crashed and we timed the crash. Redirecting stdout hid it.
Check exit codes, and sanity-check that a runtime is *physically plausible* for the work done.

## 13. A null with no positive control

The headline pitfall, and the one that cost us most. We ran a mismatch null on a published
method, got zero excess, and wrote it up as a result about the method. Two bugs in our own
pipeline: we sampled token positions outside the regime the model was trained on, and we applied
a scaling correction in the wrong direction.

The scaling bug is the instructive one. We had *empirically verified* that the framework applies
a √d normaliser internally — a correct premise, separately confirmed — and reasoned from it that
we should divide the injected vector by √d. Dividing scored −0.737; not dividing scored 0.839
against a published 0.821. **A verified premise led to a wrong conclusion, and only the
published anchor caught it.**

Always build the green test first. See [READING_A_PAPER.md](READING_A_PAPER.md), question zero.

## 14. Concluding from too narrow a comparison set

We measured how much per-prompt Jacobians vary in a sparse mixture-of-experts model and found it
4–6× higher than in dense models — a clean "sparsity breaks this method" story. Then we widened
the comparison and found a **dense** model in the same table at nearly the same value. Dense
models alone spanned a 3.3× range; the effect we attributed to sparsity was mostly *model
family*.

The tell was available before the conclusion: we had six models and compared against three of
them. Before attributing a difference to the variable you care about, check how much it varies
among things that *don't* differ on that variable.

The fix that settled it was a **within-family control** — a sparse and a dense model from the
same family, same recipe, similar size. It reversed the conclusion.

## 15. Aliased submodules stop a model being freed

```python
stack = model.model.language_model      # alias!
W_U   = model.get_output_embeddings().weight
del model                               # frees nothing
```

`del model` drops one reference while `stack` and `W_U` still pin the parameters. The next large
model load then OOMs with the previous one still resident. Delete every alias and call
`gc.collect()` — and print `torch.cuda.memory_allocated()` to prove it actually went.

## 16. float32 accumulation makes cosine exceed 1

Comparing two 4096×4096 matrices by flattened cosine, float32 returned **1.0057**. Over 16.7M
elements, accumulation error swamps the last digits. Compute similarity of large tensors in
float64. If a bounded metric leaves its bounds, the arithmetic is wrong, not the finding.

## 17. Chat templates: `tokenize=True` may not return tokens

`tokenizer.apply_chat_template(..., tokenize=True)` can return a `BatchEncoding`, so
`len(ids)` gives **2** — the number of its keys — and iterating yields key names. Use
`tokenize=False` to get the string, then tokenize with `add_special_tokens=False`, because the
rendered template usually already contains `<bos>`. Adding a second one shifts every position
index by one.

## 18. Vendor formats drift between versions

Gemma Scope 2 stores SAE weights under `w_enc`/`w_dec` (lowercase); the previous release used
`W_enc`/`W_dec`. `jlens` requires `source_layers` strictly below `target_layer`, so "all layers"
means `range(n_layers - 1)`. Neither is documented where you would look. Print the actual keys
and read the actual error rather than assuming continuity with the version you learned on.

## 19. An exception at the end discards everything before it

A multi-stage run computed its two expensive stages — two 27B models, most of an hour — and then
raised on a trivial mistake in the cheap scoring stage at the end. Nothing had been written to
disk, so all of it was lost and had to be recomputed.

Persist after each expensive stage, not at the end: write a checkpoint, and **commit it** if you
are on a remote store that requires an explicit commit (Modal Volumes do). Then make the script
resume from that checkpoint, so the retry costs the cheap stage only.

The general rule: **the cost of a failure should be proportional to the cost of the thing that
failed**, not to everything that ran before it. Logging metrics as you go (see
[GETTING_STARTED.md](GETTING_STARTED.md)) gives you the same property for free — even a failed
run leaves its numbers behind.

## 20. Asymmetric filtering: a filter that bites one condition harder than the other

Any filter that drops low-signal items will drop *most* from whichever arm has the least
signal. That is not neutral data cleaning — it deletes one condition's lower tail and
manufactures an effect.

A concrete case: an analysis excluded traces with too few segments to score. The intervention
being studied *created* segments, so the drop rate collapsed with dose — **95 of 200 baseline
traces discarded (47.5%) versus 8 at the top dose (4%)**. Nearly half the baseline was deleted,
and precisely its least diverse half. Reported diversity fell with dose. The truth was that it
**rises**.

**Always report the drop rate per arm.** If it differs, the filter is part of your result. Where
the measure allows it, score the excluded items at their floor value instead of dropping them.

*Found by the [societies-of-thought](https://github.com/m9h/societies-of-thought) agent, whose
retraction of that result is the reason this entry exists.*

## 21. Dividing by a covariate is not controlling for it

Normalising a metric by something that varies between your conditions feels like a control. It
usually is not.

A diversity score was normalised by `log2(n_segments)` on the assumption that this removed the
length dependence. The effect measured **−0.0186**, with a confidence interval excluding zero,
and it was stable across resampling, across sample size, and across **two different embedders**.
Under **1:1 length matching** it fell to **−0.0003** — a 99% shrinkage.

Division rescales; it does not equalise the distribution. If two arms differ in length, norm, or
depth profile, **match them** (pair items of equal length) or model the covariate explicitly.
Cosine-type geometry measures are exposed to the same class of problem whenever arms differ in
magnitude or trace length.

## 22. Robustness checks that cannot see the confound you have

The sharpest methodological lesson here, and both projects hit it independently.

You suspect an effect might be spurious, so you re-run it: different seeds, different sample
sizes, different embedder. It holds every time. You write "SIGN STABLE across 3 runs" and move
on. **But if none of those variations touches the suspected confound, three consistent runs of a
confounded estimator are still confounded.** Consistency is not validity.

Our own instance, from the other direction: we reported a 4–6× MoE-sparsity penalty and checked
it across models and across two different measurement methods — but never across model *family*,
which is where the confound lived. Widening to a within-family control reversed the conclusion
(see #14).

**Before running a robustness check, write down which confound it could move.** If the answer is
"none", it is reassurance, not evidence.

## 23. Per-item accuracy is often bimodal, not binomial

Sampling a model N times per problem and treating correctness as a binomial draw is a common
default and frequently wrong. On one GPQA run, only **168 of 767 problems** produced both a
correct and an incorrect answer — 388 were always right, 211 always wrong. Under independence at
that accuracy, ~94% of problems should have been mixed; 22% were.

The model largely *knows* a problem or does not, and resampling rarely changes it. Consequences:
a probe trained on pooled samples mostly learns **which problems are easy**, not which *answers*
are right — between-item structure wearing the costume of a within-item signal. That is a
candidate explanation for why our own error-monitoring signal works across questions and fails
at selecting among samples of the same question.

*Both from the societies-of-thought agent's GPQA analysis.*

## 24. The attention sink will calibrate your intervention for you

**This one was live in this repository, in the rung that teaches controls.**

GPT-2's **first position is an attention sink**: many SAE features sit there at a huge, near-constant
activation that has nothing to do with the token. For rung 5's feature #3552 the activation at
position 0 is **~123 on every prompt**, while its largest *content*-driven activation across 20,000
tokens of wikitext is **16.6**, and its mean is **0.029**.

Rung 5 calibrated its steering coefficient as "the 90th percentile of the feature's nonzero
activation" — over text *including* position 0. On all four calibration prompts the feature was
**exactly 0.0 at every content position**, so the calibration was not slightly contaminated by the
sink; it was **entirely the sink**. We steered at `4 × 123 = 492` while printing the words *"at the
feature's natural activation scale"*. The honest figure is `4 × 4.32 ≈ 17`. **We over-steered ~28×
and said the opposite.**

Two things it cost:

- **A false qualitative claim.** At the honest scale the feature does not displace the top-5 next
  tokens at all. The "the feature's own tokens take over" demo needed the 28× overshoot. The ladder
  is now printed: 1× natural → 0/5 tokens change, 4× → 0/5, 32× → 5/5.
- **An inverted statistic.** In `tools/gate_autointerp.py`, leaving position 0 in flips the headline
  correlation from **+0.459 to −0.345** — it changes the sign of the conclusion. Pearson and
  Spearman disagreeing in sign (−0.345 vs +0.125) was the tell; 40 of 40 extreme activations turned
  out to be position 0.

**The general shape:** a magnitude read off the model and fed back in as an intervention parameter is
a place where an artifact becomes invisible, because the number *looks* empirical — it was measured,
not chosen. Drop position 0. Then check that what remains is non-empty, because if the feature never
fires on your calibration text, your "natural scale" is defined entirely by whatever artifact is left.

**And the meta-lesson:** this had shipped, verified and green, through a `verify_all.sh` that only ever
checked our own scripts against our own expectations. It was caught the first time an *independent
line of evidence* was demanded — observational firing rather than intervention. Related: **#13** (a
null with no positive control), **#21** (dividing is not controlling), **#23** (bimodal, not binomial
— the same tell, a distribution with two modes masquerading as one).

## 25. An intercept can push your null *below* chance, and that looks like a finding

Building the [dead-salmon demo](salmon/), the shuffled-label null came out at **43%** rather than
50% — twice, stably. A null landing 7 points *below* chance is not noise at n=500, and the
temptation is to find it interesting.

It was a bug in the probe. The ridge fit added the **training-set mean back as an intercept**, so
predictions leaned toward whichever class was commoner in training. With complementary
cross-validation folds, the training majority is *anti*-correlated with the test fold, so the
bias pushed accuracy systematically below chance. Centring the target and dropping the intercept
moved the null to **47.6%** (5 shuffles, 44.0–50.6) — where a null belongs.

Two things worth taking:

- **Check your null is centred where it should be, not merely "not the signal."** A null at 43%
  and a null at 50% look equally "not 60%", and only one of them means your floor is measured.
  Had the signal been weaker, the same bias would have *manufactured* a gap.
- **One draw of a null is a sample, not a floor.** The fix is visible partly because we started
  averaging over five shuffles and could see the spread. A single shuffle at 44% is
  indistinguishable from a single shuffle at 50% plus bad luck.

Related: **#13** (a null with no positive control), **#20** (asymmetric filtering), **#24** (the
attention sink — the other case where a number that looked measured was an artifact).
