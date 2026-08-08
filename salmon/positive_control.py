"""THE SALMON'S TWIN — a network whose contents are known by construction.

Run after `starter.py`. ~2 s, CPU, no downloads.

WHY
---
The dead salmon gives you a NULL: a probe finds sentiment in a network that learned nothing, so
"my probe worked" proves little. That is only half a test. The other half is a POSITIVE CONTROL:
a network where you know exactly what is inside, so you can check the probe finds *that* and does
not invent anything else.

Without both, a null result is uninterpretable (PITFALLS #13): "the probe found nothing" and "my
probe cannot find things" produce identical output.

WHAT THIS IS
------------
Instead of training a model and hoping to learn what it computes, we **write the weights by hand**
so the computation is known before the first forward pass. This is the idea behind
**Tracr** (Lindner et al., 2023, arXiv:2301.05062), which compiles RASP programs into transformer
weights; Tracr is Apache-2.0 but archived and JAX-based, so here the construction is done directly
in ~40 lines of torch and you can read the ground truth in the source.

THE COMPILED PROGRAM
--------------------
One attention-only layer computing `frac_prevs(tok == 'a')` -- at every position, the fraction of
tokens so far that are 'a'.

  * Q = K = 0, so all attention scores are equal; with a causal mask, softmax gives a UNIFORM
    average over positions 0..i. Averaging an indicator IS a fraction.
  * V copies the 'is a' indicator; the output projection writes it to residual dimension 5.

So after the layer, residual dim 5 holds `frac_a` exactly. Nothing computes `frac_b`.

THE TEST -- three targets, and the middle one is the point
----------------------------------------------------------
  frac_a     the model computes it, it is literally in dim 5    -> R^2 must be ~1
  frac_b     the model NEVER computes it ...                    -> R^2 lands ~0.45
  shuffled   frac_a values attached to the wrong rows           -> R^2 must be ~0

The middle row is why this file exists. `frac_b` is not in the model -- we wrote the weights and
it is not there. But the three fractions must sum to 1, so `frac_b` is *statistically entangled*
with `frac_a`, and a linear probe decodes it at R^2 ~ 0.45 anyway.

**"A probe decodes X" and "the model represents X" are different claims**, and here we can prove
they come apart, because we built the model. In a trained network you cannot prove it -- which is
exactly why a decoding result is not a representation result.

The shuffled row is the real floor: same values, wrong rows, so no relationship survives. That is
what "the probe found nothing" actually looks like.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

VOCAB, D, N_SEQ, SEQ_LEN, SEED = ["a", "b", "c"], 8, 400, 12, 0
D_TOK, D_IND, D_OUT = slice(0, 3), 3, 5          # residual layout, fixed by construction
torch.manual_seed(SEED); np.random.seed(SEED)


def compile_model():
    """Hand-written weights. No training. The ground truth is these four lines."""
    W_E = torch.zeros(len(VOCAB), D)
    for i in range(len(VOCAB)):
        W_E[i, i] = 1.0                                   # one-hot token identity -> dims 0..2
    W_E[VOCAB.index("a"), D_IND] = 1.0                    # 'is a' indicator      -> dim 3

    W_Q = torch.zeros(D, D)                               # Q = K = 0  =>  uniform attention
    W_K = torch.zeros(D, D)
    W_V = torch.zeros(D, D); W_V[D_IND, D_IND] = 1.0      # V carries the indicator
    W_O = torch.zeros(D, D); W_O[D_IND, D_OUT] = 1.0      # write the average into dim 5
    return W_E, W_Q, W_K, W_V, W_O


def forward(tokens, W):
    """One attention-only layer, written out so nothing is hidden in a library call."""
    W_E, W_Q, W_K, W_V, W_O = W
    x = W_E[tokens]                                       # [B, T, D]
    scores = (x @ W_Q) @ (x @ W_K).transpose(-1, -2)      # all zeros, by construction
    mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), 1)
    scores = scores.masked_fill(mask, float("-inf"))      # causal
    attn = F.softmax(scores, -1)                          # uniform over j <= i
    return x + ((attn @ (x @ W_V)) @ W_O)                 # residual after the layer


def probe_r2(X, y, folds=5):
    """Cross-validated R^2 of a ridge probe. Same estimator family as starter.py, but the target
    is continuous here, so R^2 rather than accuracy. 0 = no better than the mean."""
    idx = np.random.RandomState(SEED).permutation(len(y)); X, y = X[idx], y[idx]
    out = []
    for f in range(folds):
        te = np.zeros(len(y), bool); te[f::folds] = True
        Xtr, Xte, ytr, yte = X[~te], X[te], y[~te], y[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        c = ytr.mean()
        w = np.linalg.solve(Xtr.T @ Xtr + 1e-3 * len(Xtr) * np.eye(Xtr.shape[1]),
                            Xtr.T @ (ytr - c))
        pred = Xte @ w + c
        out.append(1 - ((yte - pred) ** 2).sum() / max(((yte - yte.mean()) ** 2).sum(), 1e-9))
    return float(np.mean(out))


if __name__ == "__main__":
    W = compile_model()
    rng = np.random.RandomState(SEED)
    toks = torch.tensor(rng.randint(0, len(VOCAB), (N_SEQ, SEQ_LEN)))
    resid = forward(toks, W).numpy()

    # ground truth, computed independently of the model
    ind_a = (toks.numpy() == VOCAB.index("a")).astype(float)
    ind_b = (toks.numpy() == VOCAB.index("b")).astype(float)
    denom = np.arange(1, SEQ_LEN + 1)[None, :]
    frac_a = np.cumsum(ind_a, 1) / denom
    frac_b = np.cumsum(ind_b, 1) / denom

    X = resid.reshape(-1, D)                                   # every position is a sample
    r2_a = probe_r2(X, frac_a.reshape(-1))
    r2_b = probe_r2(X, frac_b.reshape(-1))
    shuffled = rng.permutation(frac_a.reshape(-1))              # true floor: values, wrong rows
    r2_s = probe_r2(X, shuffled)
    exact = float(np.abs(resid[:, :, D_OUT] - frac_a).max())

    print("Compiled model: one attention-only layer computing frac_prevs(tok == 'a').")
    print(f"  weights written by hand; no training. {N_SEQ} sequences x {SEQ_LEN} positions.\n")
    print(f"  residual dim {D_OUT} vs true frac_a, max abs error : {exact:.2e}   <- the compile is exact")
    print(f"  probe -> frac_a    (IS computed, dim {D_OUT})        R^2 : {r2_a:7.3f}   <- must succeed")
    print(f"  probe -> frac_b    (NOT computed, but entangled)  R^2 : {r2_b:7.3f}   <- the trap")
    print(f"  probe -> shuffled  (no relationship at all)       R^2 : {r2_s:7.3f}   <- the real floor")

    ok = exact < 1e-5 and r2_a > 0.99 and abs(r2_s) < 0.05 and r2_b > 0.15
    print("\n" + (f"  The probe recovers what was planted (R^2 {r2_a:.3f}) and is flat where nothing\n"
                  f"  relates to anything (R^2 {r2_s:.3f}). So it works.\n\n"
                  f"  And it still decodes frac_b at R^2 {r2_b:.3f} -- a quantity these weights DO NOT\n"
                  f"  COMPUTE. We know because we wrote them. The three fractions sum to 1, so frac_b\n"
                  f"  is entangled with frac_a, and the probe reads the correlation.\n\n"
                  "  **A probe decoding X is not evidence the model represents X.** Here we can prove\n"
                  "  the two come apart because the ground truth is in the source. In a trained\n"
                  "  network you cannot -- so treat every decoding result as an upper bound.\n\n"
                  "  Pair with the salmon: that shows a probe finds structure in a network with none.\n"
                  "  This shows a WORKING probe still attributes to the model what belongs to the data."
                  if ok else
                  "  UNEXPECTED -- read the three rows individually. frac_a low means the probe is\n"
                  "  broken; shuffled high means leakage in the CV; frac_b near zero would mean the\n"
                  "  entanglement this file teaches is absent in this sample."))
    sys.exit(0 if ok else 1)
