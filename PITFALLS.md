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
