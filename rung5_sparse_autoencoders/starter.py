"""Rung 5 — Sparse autoencoders: interpret a feature and steer it, with the null (runnable).

Loads a *pretrained* residual-stream SAE for GPT-2 small (Joseph Bloom's, the SAELens
`gpt2-small-res-jb` release), reads what one feature *means* (logit-lens of its decoder
direction), steers the model with it, and runs THE CONTROL: a real feature must raise its own
tokens far more than a random direction of the same norm, and its negation must lower them.
Steering that isn't specific to the feature is just noise.

    python starter.py                 # downloads GPT-2 + one SAE layer (~150MB) once

Uses plain `transformers` + a direct tensor download (no SAELens / TransformerLens), so it runs
in any HF environment and on CPU. GPT-2 SAEs are non-gated (unlike Gemma-2), so no HF token is
needed. The README's canonical target is Google's Gemma Scope — same idea, bigger model; with
SAELens it is a one-line release swap.
Dependencies: torch, transformers, huggingface_hub, safetensors.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
tok = GPT2TokenizerFast.from_pretrained("gpt2")

LAYER = 7                                                    # SAE trained on resid_pre of block 7
REPO, HOOK = "jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{LAYER}.hook_resid_pre"
sd = load_file(hf_hub_download(REPO, f"{HOOK}/sae_weights.safetensors"))
W_enc = sd["W_enc"].to(dev)                                 # [d_model, d_sae]  (768, 24576)
b_enc = sd["b_enc"].to(dev)                                 # [d_sae]
b_dec = sd["b_dec"].to(dev)                                 # [d_model]
W_dec = sd["W_dec"].to(dev)                                 # [d_sae, d_model]  -- each row is the
#                                                             direction feature f writes to resid
W_U = model.transformer.wte.weight.detach().T.to(dev)      # [d_model, vocab]  logit-lens unembed


def logit_lens_top(direction, k=8):
    """Which tokens does this residual direction promote? -> the feature's 'meaning'."""
    return (direction @ W_U).topk(k).indices.tolist()


def pick_word_feature(scan=8000, chunk=1000):
    """Pick the feature whose decoder direction most sharply promotes a single *whole word*
    (peaked logit-lens, top token word-initial and alphabetic) -- an automatically-
    interpretable content feature. Word-initial ('Ġ...') tokens avoid suffix features like
    -ness/-ly, whose steering just spams a fragment."""
    best_val, best_f = -1e9, 0
    for i in range(0, scan, chunk):
        peak, arg = (W_dec[i:i + chunk] @ W_U).max(dim=1)  # top-token logit per feature
        order = peak.argsort(descending=True)
        for j in order.tolist():
            piece = tok.convert_ids_to_tokens([int(arg[j])])[0]   # e.g. 'Ġdog' (Ġ = space)
            if piece.startswith("Ġ") and piece[1:].isalpha() and len(piece) >= 5:
                if peak[j].item() > best_val:
                    best_val, best_f = peak[j].item(), i + j
                break
    return best_f


def encode_one(resid, f):
    """Activation of a single SAE feature f on residual acts [.., d_model] -> [..]."""
    return torch.relu((resid - b_dec) @ W_enc[:, f] + b_enc[f])


@torch.no_grad()
def feature_scale(f, texts):
    """A realistic 'feature strongly on' magnitude: 90th percentile of its nonzero activation
    over some text. Steering at this scale injects the feature at its own natural strength
    instead of blowing past saturation."""
    vals = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states[LAYER][0]  # resid_pre.LAYER
        vals.append(encode_one(hs, f))
    a = torch.cat(vals)
    nz = a[a > 0]
    return float(nz.quantile(0.9)) if nz.numel() else 1.0


# --- steering: add a raw vector at the SAE's hook layer (resid_pre.7 == output of block 6) ---
_STEER = {"vec": None}


def _hook(module, inp, out):
    if _STEER["vec"] is None:
        return out
    is_tuple = isinstance(out, tuple)                       # transformers 5.x: bare tensor
    hs = out[0] if is_tuple else out
    hs = hs + _STEER["vec"].to(hs.dtype)
    return ((hs,) + tuple(out[1:])) if is_tuple else hs


model.transformer.h[LAYER - 1].register_forward_hook(_hook)  # output of block 6 == resid_pre.7


@torch.no_grad()
def specificity(prompt, vec, target_ids, control_ids):
    """Steer by vec, then score how much the feature's OWN tokens stand out: mean logit of
    target_ids minus mean logit of a fixed control token set. Subtracting the control set
    cancels the generic logit shift any large perturbation causes -- so a random direction
    nulls out, and only a direction *specific* to these tokens scores."""
    _STEER["vec"] = vec
    last = model(tok(prompt, return_tensors="pt").input_ids.to(dev)).logits[0, -1].float()
    _STEER["vec"] = None
    return (last[torch.tensor(target_ids)].mean() - last[torch.tensor(control_ids)].mean()).item()


@torch.no_grad()
def top_next(prompt, vec, k=5):
    """The model's top-k next-token predictions with the steering vector applied."""
    _STEER["vec"] = vec
    lg = model(tok(prompt, return_tensors="pt").input_ids.to(dev)).logits[0, -1]
    _STEER["vec"] = None
    return [tok.decode([t]) for t in lg.topk(k).indices.tolist()]


if __name__ == "__main__":
    f = pick_word_feature()
    feat = W_dec[f]
    top = logit_lens_top(feat)
    print(f"Picked SAE feature #{f} of {W_dec.shape[0]} (GPT-2 layer {LAYER} resid_pre).")
    print("  it promotes tokens:", [tok.decode([t]) for t in top])

    calib = ["The weather in London is", "She studied biology at the university and",
             "The government announced a new policy on", "He picked up the phone and said"]
    scale = feature_scale(f, calib)
    coef = 4.0 * scale                                     # inject the feature ~4x its typical strength
    print(f"  natural activation scale ~{scale:.1f}; steering at coef {coef:.1f}")

    feat_vec = coef * feat                                 # steer WITH the feature
    neg_vec = -coef * feat                                 # its negation
    rng = torch.Generator(device=dev).manual_seed(0)
    rand = torch.randn_like(feat, dtype=torch.float32)     # random dir, matched to feat_vec's norm
    rand_vec = rand / rand.norm() * feat_vec.norm()
    ctrl = list(range(1000, 1000 + 256))                   # fixed control token set (scale baseline)

    prompt = "The city council met on"
    clean = specificity(prompt, None, top, ctrl)
    pos = specificity(prompt, feat_vec, top, ctrl)
    neg = specificity(prompt, neg_vec, top, ctrl)
    rnd = specificity(prompt, rand_vec, top, ctrl)

    print(f"\nSpecificity (mean logit of the feature's tokens minus a control set) at "
          f"'{prompt} …':")
    print(f"  clean (no steer)         : {clean:6.2f}")
    print(f"  + feature   (steer on)   : {pos:6.2f}   (delta {pos - clean:+.2f})")
    print(f"  + RANDOM dir (same norm) : {rnd:6.2f}   (delta {rnd - clean:+.2f})   <- control")
    print(f"  - feature   (negation)   : {neg:6.2f}   (delta {neg - clean:+.2f})   <- control")

    gen_vec = scale * feat                                 # 1x natural strength, for a legible effect
    print(f"\nTop-5 next tokens after '{prompt}':")
    print("  clean  :", top_next(prompt, None))
    print("  steered:", top_next(prompt, gen_vec), " <- the feature's own tokens take over")

    print("\nCONTROL: steering WITH the feature must raise its own tokens' specificity far more")
    print("than a random direction of the same norm (which shifts all logits together and nets")
    print("~zero here), and its NEGATION must lower it. Interpretable + steerable + null-does-")
    print("nothing = a real monosemantic feature, not a coincidence of the readout.")
