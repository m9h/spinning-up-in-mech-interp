"""Rung 3 — Toy Models of Superposition, runnable on a laptop (CPU, seconds).

Trains the toy ReLU autoencoder from Elhage et al. (2022): n sparse features compressed into
m < n dimensions. Sweeps sparsity to show the transition into superposition, and runs THE
CONTROL — the learned dictionary must recover features better than a random dictionary.

    python starter.py

No GPU, no downloads. Dependencies: torch.
"""
import torch


def make_batch(n_features, batch, sparsity, importance, g):
    """Sparse feature vectors: each feature present with prob (1 - sparsity), value ~U(0,1)."""
    val = torch.rand(batch, n_features, generator=g)
    mask = (torch.rand(batch, n_features, generator=g) > sparsity).float()
    return val * mask, importance


def train_toy(n_features=20, m_dims=5, sparsity=0.7, steps=4000, seed=0):
    """x -> W (m x n) -> h -> ReLU(W^T h + b) -> x_hat, minimizing importance-weighted MSE."""
    g = torch.Generator().manual_seed(seed)
    importance = 0.9 ** torch.arange(n_features)                 # earlier features matter more
    W = torch.randn(m_dims, n_features, generator=g, requires_grad=True)
    b = torch.zeros(n_features, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(steps):
        x, imp = make_batch(n_features, 512, sparsity, importance, g)
        h = x @ W.T                                             # encode  [batch, m]
        x_hat = torch.relu(h @ W + b)                           # decode  [batch, n]
        loss = (imp * (x - x_hat) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), importance


def feature_recovery(W):
    """How orthonormal are the represented feature directions? cols of W, normalized -> W^T W.
    A represented feature has |w_i| ~ 1 and small overlap with others. Score = mean over the
    'represented' features (top-|w| ones) of how close their Gram row is to a one-hot."""
    norms = W.norm(dim=0)                                        # [n]
    Wn = W / norms.clamp_min(1e-6)
    gram = Wn.T @ Wn                                             # [n, n], ~identity if orthogonal
    represented = norms > 0.5                                    # features the model actually stores
    if represented.sum() < 2:
        return 0.0, int(represented.sum())
    idx = represented.nonzero().squeeze(1)
    sub = gram[idx][:, idx].abs()
    off_diag = (sub.sum(1) - sub.diag()) / (len(idx) - 1)        # mean off-diagonal per feature
    return float((1 - off_diag).mean()), int(represented.sum())  # 1 = perfectly orthogonal


def random_dictionary_null(n_features, m_dims, seed=0):
    g = torch.Generator().manual_seed(seed + 999)
    W = torch.randn(m_dims, n_features, generator=g)
    return feature_recovery(W)[0]


if __name__ == "__main__":
    n, m = 20, 5
    print(f"Toy model: {n} features -> {m} dims  (superposition needs n > m)\n")
    print(f"{'sparsity':>9} {'#represented':>13} {'recovery':>9} {'random null':>12} {'beats null?':>12}")
    for sparsity in (0.0, 0.5, 0.7, 0.9, 0.99):
        W, _ = train_toy(n, m, sparsity=sparsity)
        rec, nrep = feature_recovery(W)
        null = random_dictionary_null(n, m)
        print(f"{sparsity:9.2f} {nrep:13d} {rec:9.3f} {null:12.3f} "
              f"{'YES' if rec > null + 0.05 else 'no':>12}")
    print("\nRead: at sparsity 0 the model stores only the top-m features orthogonally "
          "(#represented ~ m).")
    print("As sparsity rises it packs MORE than m features into m dims -- superposition -- and")
    print("recovery must still beat the random-dictionary null to count as real feature learning.")
