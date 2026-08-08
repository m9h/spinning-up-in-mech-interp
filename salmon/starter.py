"""THE DEAD SALMON — the demonstration this whole curriculum rests on.

Not a rung. The premise. Run it before rung 1.

WHAT IT SHOWS
-------------
Take a transformer with RANDOMLY INITIALIZED WEIGHTS. It has learned nothing; it cannot
contain a representation of sentiment any more than a dead fish can reason about social
situations. Extract its token representations for a few hundred labelled sentences, average
over the sequence, train a linear probe.

The probe works. Well above chance, cross-validated.

This is Figure 1 of *The Dead Salmons of AI Interpretability* (Méloux, Dirupo, Portet &
Peyrard, 2025, arXiv:2512.18792), which imports the 2009 fMRI study in which a dead Atlantic
salmon showed "significant" brain activation to photographs of humans in social situations.
The fish was dead. The analysis was standard.

WHY A PROBE ON A DEAD NETWORK STILL WORKS
-----------------------------------------
A randomly initialized network is a **random projection**. Johnson-Lindenstrauss says random
projections approximately preserve distances -- so whatever separates positive from negative
sentences in the *input* is still linearly separable in the *output*. The probe is reading the
data through the network, not out of it.

That is the trap in one sentence: **"I trained a probe and it worked" is a fact about your
dataset until you show it is a fact about your model.**

THE FOUR NUMBERS
----------------
  1. RANDOM net + real labels     -- the artifact. Well above chance.
  2. SHUFFLED labels              -- null (a). Must collapse to chance, or the CV leaks and
                                     every other number here is meaningless.
  3. BAG-OF-WORDS, random proj.   -- null (b). THE ONE THAT BITES. Same dimensionality, no
                                     transformer at all. If the random net does not beat this,
                                     the architecture contributed nothing and the "representation"
                                     is just the input.
  4. PRETRAINED net + real labels -- positive control. Must beat the random net, or the probe
                                     cannot detect learning and finding nothing would prove
                                     nothing (PITFALLS #13).

Read them together. Any one alone tells you the wrong thing.
"""

import sys
import numpy as np
import torch

N_SENT, MAXLEN, SEED = 500, 64, 0
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_sentences(n=N_SENT):
    """IMDb, as the paper used. Truncated to keep this a ~20 s demo on a laptop CPU."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ! needs `pip install datasets` (also used by tools/gate_autointerp.py)")
        sys.exit(2)
    ds = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=SEED).select(range(n))
    return [t[:800] for t in ds["text"]], np.array(ds["label"])


@torch.no_grad()
def embed(model, tok, texts, bs=32):
    """Mean-pooled last hidden state -- the standard 'sentence representation' recipe."""
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAXLEN)
        h = model(**enc).last_hidden_state                       # [B, T, d]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).numpy())    # mean over real tokens
    return np.concatenate(out)


def cv_accuracy(X, y, folds=5):
    """Cross-validated logistic regression, standardised. Plain numpy so the rung keeps its
    tiny dependency list: closed-form ridge on centred features, thresholded."""
    idx = np.random.RandomState(SEED).permutation(len(y))
    X, y = X[idx], y[idx]
    accs = []
    for f in range(folds):
        te = np.zeros(len(y), bool); te[f::folds] = True
        Xtr, Xte, ytr, yte = X[~te], X[te], y[~te], y[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        t = ytr * 2.0 - 1.0                                       # {0,1} -> {-1,+1}
        t = t - t.mean()                                          # centre: no majority-class bias
        lam = 0.1 * len(Xtr)
        w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ t)
        accs.append((((Xte @ w) > 0).astype(int) == yte).mean())
        # NOTE: adding the training intercept back would bias predictions toward the training
        # majority, which is ANTI-correlated with a complementary test fold -- it drove the
        # shuffled-label null to 43% instead of 50% and looked like a real effect. PITFALLS #25.
    return float(np.mean(accs))


if __name__ == "__main__":
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    NAME = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(NAME)
    texts, labels = load_sentences()
    print(f"Dead salmon: {len(texts)} IMDb sentences, {labels.mean():.0%} positive.\n")

    # --- 1. the artifact: a probe on a network that has learned nothing ---
    torch.manual_seed(SEED)
    rand_net = AutoModel.from_config(AutoConfig.from_pretrained(NAME)).eval()
    X_rand = embed(rand_net, tok, texts)
    a_rand = cv_accuracy(X_rand, labels)

    # --- 2. null (a): shuffled labels. Catches leakage in the CV itself.
    #        Averaged over several shuffles -- one draw of a null is a sample, not a floor. ---
    shufs = [cv_accuracy(X_rand, np.random.RandomState(k).permutation(labels)) for k in range(5)]
    a_shuf = float(np.mean(shufs))

    # --- 3. null (b): no transformer at all. Bag-of-words -> random projection, same width. ---
    vocab = tok.vocab_size
    counts = np.zeros((len(texts), 2048), np.float32)
    for i, t in enumerate(texts):                                 # hashed BoW, keeps it small
        for tid in tok(t, truncation=True, max_length=MAXLEN)["input_ids"]:
            counts[i, tid % 2048] += 1.0
    R = np.random.RandomState(SEED).randn(2048, X_rand.shape[1]).astype(np.float32)
    a_bow = cv_accuracy(counts @ R / np.sqrt(2048), labels)

    # --- 4. positive control: the same probe on a network that HAS learned ---
    trained = AutoModel.from_pretrained(NAME).eval()
    a_train = cv_accuracy(embed(trained, tok, texts), labels)

    print(f"  1. RANDOM net,  real labels     : {a_rand:6.1%}   <- the artifact")
    print(f"  2. RANDOM net,  SHUFFLED labels : {a_shuf:6.1%}   <- null (a): CV integrity "
          f"(5 shuffles, {min(shufs):.1%}-{max(shufs):.1%})")
    print(f"  3. bag-of-words, random proj.   : {a_bow:6.1%}   <- null (b): no transformer")
    print(f"  4. PRETRAINED net, real labels  : {a_train:6.1%}   <- positive control")

    print(f"\n  random net beats chance by      : {a_rand - 0.5:+.1%}")
    print(f"  random net beats bag-of-words by: {a_rand - a_bow:+.1%}   <- what the architecture added")
    print(f"  pretraining adds                : {a_train - a_rand:+.1%}")

    ok = (a_rand > 0.55) and (abs(a_shuf - 0.5) < 0.05) and (a_train > a_rand)
    print("\n" + ("  A linear probe read sentiment out of a network that never learned anything.\n"
                  "  The shuffled-label null is at chance, so the pipeline is sound. The\n"
                  "  bag-of-words null tells you how much of it was ever about the model.\n"
                  "  Pretraining still helps -- so the probe CAN see learning, which is the only\n"
                  "  reason its verdict on the random net means anything.\n\n"
                  "  Before you believe any interpretability result: run it on a dead network."
                  if ok else
                  "  UNEXPECTED. One of the four is out of range -- read them individually before\n"
                  "  drawing any conclusion. That is the point of having four."))
    sys.exit(0 if ok else 1)
