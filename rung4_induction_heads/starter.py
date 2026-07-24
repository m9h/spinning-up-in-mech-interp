"""Rung 4 — Induction heads on GPT-2, runnable (HuggingFace transformers + hooks).

Finds the induction heads behind in-context learning, then runs THE CONTROL: ablating them
must selectively destroy in-context (2nd-copy) prediction while barely touching ordinary loss,
and ablating an equal number of RANDOM heads must not.

    python starter.py                 # downloads GPT-2 (~500MB) once

Uses plain `transformers` (no TransformerLens) so it runs in any HF environment. The canonical
tooling for this (TransformerLens, and ARENA's induction exercises) is cleaner for per-head
work — see README.md — but doing it by hand here shows exactly what an "induction head" is.
Dependencies: torch, transformers.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(dev).eval()
tok = GPT2TokenizerFast.from_pretrained("gpt2")
NL, NH = model.config.n_layer, model.config.n_head          # 12 layers, 12 heads


def repeated_batch(batch=32, seq=64):
    """[ rand tokens (len seq) ][ same tokens again ] — the induction test stimulus."""
    half = torch.randint(0, tok.vocab_size, (batch, seq))
    return torch.cat([half, half], dim=1).to(dev)           # [batch, 2*seq]


@torch.no_grad()
def second_copy_loss(ids, ablate=None):
    """Cross-entropy on the SECOND copy only (where induction should help)."""
    hooks = _install_ablation(ablate) if ablate else []
    out = model(ids)
    for h in hooks:
        h.remove()
    logits = out.logits[:, :-1]; targets = ids[:, 1:]
    L = ids.shape[1] // 2
    lp = torch.log_softmax(logits.float(), -1)
    tok_lp = lp.gather(2, targets.unsqueeze(2)).squeeze(2)  # [batch, 2L-1]
    return -tok_lp[:, L:].mean().item()                     # second-copy positions only


@torch.no_grad()
def induction_scores(ids):
    """Per (layer,head): attention from position i (2nd copy) to i-L+1 (the token after the
    previous occurrence) — the induction pattern. Returns [NL, NH]."""
    out = model(ids, output_attentions=True)
    L = ids.shape[1] // 2
    scores = torch.zeros(NL, NH)
    dest = torch.arange(L, 2 * L - 1)                       # 2nd-copy positions
    src = dest - L + 1                                      # induction target
    for l in range(NL):
        a = out.attentions[l]                              # [batch, NH, seq, seq]
        scores[l] = a[:, :, dest, src].mean(dim=(0, 2))
    return scores


class _Restore:
    def __init__(self, m): self.m = m
    def remove(self):
        self.m.forward = self.m._orig_forward; del self.m._orig_forward


def _install_ablation(head_list):
    """Zero the OV contribution of the given (layer,head) pairs by masking the per-head slice
    of the attention output before the output projection (c_proj)."""
    dh = model.config.n_embd // NH
    by_layer = {}
    for (l, h) in head_list:
        by_layer.setdefault(l, []).append(h)
    handles = []
    for l, heads in by_layer.items():
        def mk(heads):
            def hook(self, x):                              # c_proj input = concat heads [.., n_embd]
                x = x.clone()
                for h in heads:
                    x[..., h * dh:(h + 1) * dh] = 0
                return self._orig_forward(x)
            return hook
        cproj = model.transformer.h[l].attn.c_proj
        cproj._orig_forward = cproj.forward
        cproj.forward = mk(heads).__get__(cproj)            # ablate the head slice, then project
        handles.append(_Restore(cproj))
    return handles


if __name__ == "__main__":
    ids = repeated_batch()
    base = second_copy_loss(ids)
    scores = induction_scores(ids)
    flat = sorted(((float(scores[l, h]), l, h) for l in range(NL) for h in range(NH)),
                  reverse=True)
    top = [(l, h) for _, l, h in flat[:5]]
    print("Top induction heads (layer, head, score):")
    for s, l, h in flat[:5]:
        print(f"  L{l:>2} H{h:>2}   induction score {s:.3f}")

    rng = torch.Generator().manual_seed(1)
    rand = [(int(torch.randint(0, NL, (1,), generator=rng)),
             int(torch.randint(0, NH, (1,), generator=rng))) for _ in range(5)]

    abl_ind = second_copy_loss(ids, ablate=top)
    abl_rnd = second_copy_loss(ids, ablate=rand)
    print(f"\n2nd-copy loss   baseline: {base:.3f}")
    print(f"  ablate top-5 INDUCTION heads: {abl_ind:.3f}   (delta {abl_ind - base:+.3f})")
    print(f"  ablate 5 RANDOM heads       : {abl_rnd:.3f}   (delta {abl_rnd - base:+.3f})")
    print("\nCONTROL: ablating induction heads should raise 2nd-copy loss far more than "
          "ablating random heads.")
    print("That selective causal effect -- not the attention picture -- is the claim.")
