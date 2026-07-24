"""Rung 2 — The residual stream is a channel; attention heads are QK + OV (runnable).

The Mathematical Framework (Elhage et al., 2021) says an attention head factors into two
independent circuits: **QK** decides *where* to attend, **OV** decides *what* to write into the
residual stream from there. This script measures both on GPT-2 small, each against its null:

  * OV copying score (weights only): when a head attends to a token, does its OV circuit push
    that same token's logit up? A "copying head" has a large positive score; a RANDOM matrix of
    the same size does not.
  * QK previous-token score (one forward pass): does the head attend from position i to i-1?
    A "previous-token head" beats the uniform-attention baseline; most heads don't.

Together these are the two ingredients of an induction head (rung 4): a QK that finds the
match and an OV that copies. No training, no downloads beyond GPT-2.

    python starter.py

Dependencies: torch, transformers.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").to(dev).eval()
tok = GPT2TokenizerFast.from_pretrained("gpt2")
NL, NH = model.config.n_layer, model.config.n_head          # 12, 12
D = model.config.n_embd; DH = D // NH                        # 768, 64
wte = model.transformer.wte.weight.detach()                 # [vocab, D]; also the (tied) unembed


def head_WOV(l, h):
    """The OV matrix W_V W_O for head (l,h): residual -> residual, i.e. what this head copies
    into the stream from a token it attends to (with the attention weight factored out)."""
    Wc = model.transformer.h[l].attn.c_attn.weight.detach()  # [D, 3D] (q|k|v), x @ Wc
    Wv = Wc[:, 2 * D + h * DH: 2 * D + (h + 1) * DH]         # [D, DH]  residual -> value
    Wo = model.transformer.h[l].attn.c_proj.weight.detach()[h * DH:(h + 1) * DH, :]  # [DH, D]
    return Wv @ Wo                                           # [D, D]


@torch.no_grad()
def copying_score(WOV, sample_ids):
    """For each sampled token i: send its embedding through the OV circuit, unembed, and see how
    far the token's OWN logit sits above the average logit across the sample (a per-row z-score).
    Copying head -> large positive; random matrix -> ~0."""
    moved = wte[sample_ids] @ WOV                            # [S, D]  what OV writes for token i
    L = moved @ wte[sample_ids].T                            # [S, S]  logit of token j via token i
    z = (L.diag() - L.mean(1)) / L.std(1).clamp_min(1e-6)   # self-logit, in row std units
    return z.mean().item()


@torch.no_grad()
def prev_token_scores(text):
    """Per (layer,head): mean attention from position i to i-1 -- the 'previous-token head'
    pattern. Returns [NL, NH]. Uniform attention would give ~mean(1/i)."""
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    att = model(ids, output_attentions=True).attentions      # NL x [1, NH, seq, seq]
    seq = ids.shape[1]
    i = torch.arange(1, seq)
    scores = torch.zeros(NL, NH)
    for l in range(NL):
        scores[l] = att[l][0, :, i, i - 1].mean(dim=1)       # attention i -> i-1, averaged over i
    uniform = (1.0 / torch.arange(1, seq).float()).mean().item()
    return scores, uniform


if __name__ == "__main__":
    sample = torch.randperm(wte.shape[0])[:512].to(dev)      # 512 random tokens

    # --- OV: which heads copy? (weights only) ---
    ov = torch.tensor([[copying_score(head_WOV(l, h), sample) for h in range(NH)]
                       for l in range(NL)])
    rng = torch.Generator(device=dev).manual_seed(0)
    typ = head_WOV(0, 0).norm()
    rnd = torch.randn(D, D, generator=rng, device=dev); rnd = rnd / rnd.norm() * typ
    null_ov = copying_score(rnd, sample)

    flat = sorted(((float(ov[l, h]), l, h) for l in range(NL) for h in range(NH)), reverse=True)
    print("OV copying score -- top heads (does the OV circuit promote the attended token?):")
    for s, l, h in flat[:5]:
        print(f"  L{l:>2} H{h:>2}   copying z = {s:+.2f}")
    print(f"  median head          : {flat[len(flat)//2][0]:+.2f}")
    print(f"  RANDOM matrix (null) : {null_ov:+.2f}   <- control: copying is specific to some heads")

    # --- QK: which heads look back one token? (one forward pass) ---
    text = ("The mechanistic interpretability of transformers begins with the residual stream, "
            "a channel that every layer reads from and writes to.")
    qk, uniform = prev_token_scores(text)
    flatq = sorted(((float(qk[l, h]), l, h) for l in range(NL) for h in range(NH)), reverse=True)
    print("\nQK previous-token score -- top heads (does the head attend from i to i-1?):")
    for s, l, h in flatq[:5]:
        print(f"  L{l:>2} H{h:>2}   attn(i -> i-1) = {s:.2f}")
    print(f"  uniform-attention null : {uniform:.2f}   <- control: prev-token heads beat chance")

    print("\nAn attention head = QK (where) + OV (what). The heads that score high here are the")
    print("raw material of the induction circuit in rung 4: a QK that finds a match, an OV that")
    print("copies. Trust the scores against their nulls -- not the story -- exactly as there.")
