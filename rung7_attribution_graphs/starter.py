"""Rung 7 — Trace a computation across layers with activation patching (runnable).

Attribution graphs answer "which internal components carry this computation, and how does it
flow across layers?" The method underneath them is **activation patching** (causal tracing;
Meng et al. 2022, Wang et al. IOI 2022): run a clean and a corrupted prompt, then copy the
clean residual stream into the corrupted run one (layer, position) at a time and see how much
of the answer it restores. The cells that restore it ARE the computation's path.

Here, on GPT-2 small's classic IOI task: "When John and Mary went to the store, John gave a
drink to ___" -> " Mary". Corrupting the subject (John->Mary) flips the answer to " John".
Patching sweeps every (layer, position) and maps where the name-identity information lives and
how it moves to the final token -- a minimal attribution graph. THE CONTROL is built in: only
a few cells restore the answer; the rest (the null) do nothing.

    python starter.py                 # downloads GPT-2 (~500MB) once

Dependencies: torch, transformers.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
tok = GPT2TokenizerFast.from_pretrained("gpt2")
NL = model.config.n_layer

CLEAN = "When John and Mary went to the store, John gave a drink to"      # -> " Mary"
CORR = "When John and Mary went to the store, Mary gave a drink to"       # -> " John"
IO, S = tok(" Mary").input_ids[0], tok(" John").input_ids[0]


@torch.no_grad()
def logit_diff(ids, patch=None):
    """logit(' Mary') - logit(' John') at the last position. If patch=(layer, pos, vec), overwrite
    that residual position at that block's output during the forward pass."""
    handle = None
    if patch is not None:
        L, P, vec = patch

        def hook(module, inp, out):
            is_tuple = isinstance(out, tuple)
            hs = out[0] if is_tuple else out
            hs = hs.clone()
            hs[:, P, :] = vec.to(hs.dtype)
            return ((hs,) + tuple(out[1:])) if is_tuple else hs

        handle = model.transformer.h[L].register_forward_hook(hook)
    lg = model(ids).logits[0, -1]
    if handle:
        handle.remove()
    return (lg[IO] - lg[S]).item()


if __name__ == "__main__":
    clean_ids = tok(CLEAN, return_tensors="pt").input_ids.to(dev)
    corr_ids = tok(CORR, return_tensors="pt").input_ids.to(dev)
    toks = [tok.decode([t]) for t in clean_ids[0].tolist()]
    ld_clean = logit_diff(clean_ids)
    ld_corr = logit_diff(corr_ids)
    print(f"logit-diff (Mary - John):  clean {ld_clean:+.2f} (says Mary),  "
          f"corrupted {ld_corr:+.2f} (says John)")

    # clean residual at each block output, to patch into the corrupted run
    clean_hs = model(clean_ids, output_hidden_states=True).hidden_states  # len NL+1

    seq = clean_ids.shape[1]
    grid = torch.zeros(NL, seq)
    for L in range(NL):
        for P in range(seq):
            ld = logit_diff(corr_ids, patch=(L, P, clean_hs[L + 1][0, P]))
            grid[L, P] = (ld - ld_corr) / (ld_clean - ld_corr)        # fraction of answer restored

    ramp = " .:-=+*#%@"
    print("\nRecovery of the answer when the CLEAN residual is patched in at (layer, position).")
    print("1.0 = this cell alone restores ' Mary'.  Columns = token positions:\n")
    print("        " + "".join(f"{t.strip()[:4]:>5}" for t in toks))
    for L in range(NL):
        cells = "".join(ramp[min(len(ramp) - 1, max(0, int(grid[L, P] * 9)))] * 2 + " "
                        for P in range(seq))
        print(f"  L{L:>2} | {cells}")

    flat = sorted(((float(grid[L, P]), L, P) for L in range(NL) for P in range(seq)), reverse=True)
    print("\nTop cells (the attribution path):")
    for r, L, P in flat[:5]:
        print(f"  L{L:>2}  pos {P:>2} ({toks[P].strip()!r:8})  restores {r:.0%}")
    med = sorted(float(grid[L, P]) for L in range(NL) for P in range(seq))[NL * seq // 2]
    print(f"\nCONTROL: the median (layer, position) cell restores {med:.0%} -- almost nothing. Only a")
    print("few cells carry the computation: the subject name early, then the FINAL token in late")
    print("layers, where name-mover heads write the answer. That sparse path IS the attribution")
    print("graph; a graph that names cells the patch shows are inert would be unfaithful.")
