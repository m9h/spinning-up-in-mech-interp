# Getting started

Every runnable rung is a **single self-contained script** that downloads a small open model,
finds a real result, and runs its own control. No GPU is required — the numbers below were all
produced on CPU.

---

## Install

```bash
git clone https://github.com/m9h/spinning-up-in-mech-interp
cd spinning-up-in-mech-interp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

That is the whole setup. `requirements.txt` is deliberately small — `torch`, `transformers`,
`torchvision`, `huggingface_hub`, `safetensors` — because the rungs deliberately avoid the
heavier canonical tools (TransformerLens, SAELens, lucent, `circuit-tracer`). Those are named
in each rung as the *scale-up*, not as a dependency, so nothing blocks you on day one.

No Hugging Face token is needed: every model used is ungated.

## Run

```bash
cd rung3_superposition && python starter.py
```

Or check the whole curriculum at once:

```bash
./verify_all.sh
```

## What to expect

Verified end-to-end from a clean virtualenv on CPU. Times exclude the one-off model download.

| rung | runtime | download | what you should see |
|---|---|---|---|
| **1** features & circuits | ~23 s | ~50 MB | conv1 channel **#48**, selectivity **1.000**, single-spike tuning curve; trained median 0.248 |
| **2** residual stream | ~5 s | ~500 MB | OV copying top **L11H3 +5.84** (median head +0.96); QK previous-token top **L4H11 = 1.00** |
| **3** superposition | ~13 s | none | more features represented than dimensions as sparsity rises; at sparsity 0.99 recovery collapses to the null |
| **4** induction heads | ~18 s | (GPT-2, cached) | **L5H5 0.926**, L5H1, L6H9 — GPT-2's documented induction heads |
| **5** sparse autoencoders | ~7 s | ~150 MB | feature **#3552** → ` alike, respectively, who…`; top-5 next tokens flip from weekdays to the feature's tokens |
| **7** attribution graphs | ~10 s | (GPT-2, cached) | clean +2.53 / corrupted −2.98; the computation moving from the subject token (early layers) to the final token (late layers) |

Rungs 2, 4, 5 and 7 all use GPT-2 small, so the ~500 MB download happens once.

### The controls — the actual point

| rung | signal | its null |
|---|---|---|
| 1 | trained top **1.000** | random-init ~0.5, weight-shuffle ~0.98; **40/64** units beat every shuffle seed |
| 2 | copying z **+5.84** | random matrix ≈ **0**; QK uniform-attention baseline **0.16** |
| 3 | recovery beats null at moderate sparsity | random dictionary; at sparsity 0.99 they **converge** — the control catches the failure |
| 4 | ablate induction heads **+6.35** | ablate 5 random heads **+0.23** |
| 5 | steer with feature **+6.53** | random direction ≈ **0**; negation **−13.45** |
| 7 | top cells restore **~100%** | median (layer, position) cell restores **~0%** |

### Why your numbers may differ slightly

The *measured* quantities above are deterministic and should reproduce exactly. The *null*
values will vary a little with library version and RNG — we see the random-matrix null move
between +0.01 and +0.10, and the random-direction steer between −1.2 and +1.0. That is
correct behaviour: a null is a distribution scattered around zero, which is why the rungs
compare against it rather than quoting it as a constant. If a null ever approaches the signal,
that is a finding, not a bug — investigate it.

## Track your runs (five lines, and you stop losing results)

Every rung produces numbers you will want to compare — across library versions, across a fix you
made, across the null you added afterwards. Shell output does not survive that.

[**trackio**](https://huggingface.co/docs/trackio/index) is Hugging Face's free experiment
tracker: local-first, no account, a Gradio dashboard, and API-compatible with Weights & Biases,
so it is a drop-in:

```python
import trackio as wandb                      # yes, that import line is the whole migration

wandb.init(project="spinning-up", name="rung4-induction",
           config={"model": "gpt2", "batch": 32, "seq": 64, "dtype": "float32"})
wandb.log({"induction_top": 0.926, "ablate_induction": 6.35, "ablate_random": 0.23})
wandb.finish()
```

`pip install trackio`, then `trackio show` for the dashboard. Logs go to **SQLite** (freezable to
Parquet) and there is a **CLI that queries the SQL directly**, which is what makes it usable by
scripts and agents rather than only by a human staring at charts.

One operational gotcha: trackio's init state is **thread-affine**. Initialise on a worker
thread and log from the main thread and you get *"Call trackio.init() before trackio.log()"*.
If you wrap it in a watchdog thread, init and log on the same one.

**Log the config, not just the metric.** Almost every confusing result in
[PITFALLS.md](PITFALLS.md) — dtype, batch size, seed count, layer index, token position — is a
*config* difference that looked like a finding. If the config is in the row, the confusion lasts
a minute instead of an afternoon.

> We did not do this. This project ran 154 checkpoints × 10 seeds × 4 model sizes, dtype
> controls, and multi-stage cloud jobs entirely through JSONL files and shell logs — and lost a
> completed 27-billion-parameter stage to an exception that fired after it. Five lines would
> have kept it.

## Hardware

Everything above runs on a laptop CPU. A GPU makes the rungs faster but changes nothing. The
two rungs that need real compute are **6** and **8**, which use our companion research code
([`jacobian-lens`](https://github.com/m9h/jacobian-lens),
[`jlens-lab`](https://github.com/m9h/jlens-lab)) — and even there, the generated artifacts are
published openly, so you can inspect and verify them without reproducing them.

## Suggested order

- **New to interpretability**: read [`GLOSSARY.md`](GLOSSARY.md) first, then rungs 1 → 3 → 2 →
  4. Rung 3 needs no downloads and shows the core idea (superposition) fastest.
- **Comfortable with transformers, here for the recent papers**: 2 → 4 → 5 → 7, then rung 8.
- **Here to evaluate a specific claim**: read [`READING_A_PAPER.md`](READING_A_PAPER.md), then
  the rung matching the technique.

## When something breaks

Check [`PITFALLS.md`](PITFALLS.md) — it lists every trap that cost us a wrong result while
building this, including the two that briefly inverted our own published numbers.

The most common first stumble: attention weights come back empty because recent
`transformers` defaults to SDPA. The rungs already pass `attn_implementation="eager"`; if you
adapt the code, carry it over.
