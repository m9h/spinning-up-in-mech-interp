"""TOOL GATE — does an independent line of evidence recover the signal rung 5 measured?

WHY THIS EXISTS
---------------
ECOSYSTEM.md recommends external tools (Delphi, Monitor, Neuronpedia, ADAG). Every one of
those recommendations rests on *maintenance* evidence -- the repo is alive, it has stars, it
was pushed last month. That is not evidence the tool works. This curriculum's whole argument
is that a plausible method needs a control before you believe it, and we were exempting our
own tool recommendations from the rule.

So: the standard is that a tool must recover a signal we independently measured.

WHAT IS UNDER TEST
------------------
Rung 5 reads a feature's "meaning" off pure DECODER GEOMETRY -- the tokens its decoder row
promotes through the unembed, `W_dec[f] @ W_U` -- and then validates it INTERVENTIONALLY, by
steering with that direction and watching those tokens take over (+6.5 specificity, random
-2.6, negation -13.5).

Both halves of that are *our* construction. Nothing so far shows the feature behaves this way
when nobody is pushing on it. If the label only describes the feature under a 4x-scale
injection, it is a fact about our intervention, not about the model.

THE INDEPENDENT EVIDENCE
------------------------
Where the feature FIRES ON ITS OWN, in real text, with no steering whatsoever. We stream
wikitext, record the feature's activation at every position, and ask whether the decoder-
derived token set is elevated exactly where the feature happens to be active.

This is precisely what Delphi's `detection` scorer asks -- "given the explanation, can you
tell where the feature is active?" -- except computed exactly rather than by asking a 70B
model to guess, so it runs on CPU in seconds instead of needing vLLM and 40GB of weights.
See ECOSYSTEM.md for the heavyweight path.

RED / GREEN
-----------
GREEN  observational specificity at the feature's own top-firing positions beats BOTH nulls.
RED    it does not -- in which case rung 5's label describes our steering, not the feature,
       and the rung needs rewriting. A red result here is a finding, not a bug.

Nulls:
  (1) random positions           -- controls for "text just looks like this"
  (2) ANOTHER feature's top positions -- controls for "any strongly-firing position looks
      like this", which is the null that actually bites
"""

import sys
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

N_CHUNKS, CHUNK, TOP_K = 40, 512, 200          # ~20k tokens of real text
dev = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
tok = GPT2TokenizerFast.from_pretrained("gpt2")

LAYER = 7
REPO, HOOK = "jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{LAYER}.hook_resid_pre"
sd = load_file(hf_hub_download(REPO, f"{HOOK}/sae_weights.safetensors"))
W_enc, b_enc = sd["W_enc"].to(dev), sd["b_enc"].to(dev)
b_dec, W_dec = sd["b_dec"].to(dev), sd["W_dec"].to(dev)
W_U = model.transformer.wte.weight.detach().T.to(dev)

CTRL = torch.tensor(list(range(1000, 1000 + 256)))   # same fixed control set rung 5 uses


def logit_lens_top(direction, k=8):
    return (direction @ W_U).topk(k).indices.tolist()


def pick_word_feature(scan=8000, chunk=1000):
    """Byte-for-byte the selection rung 5 makes, so we are testing *its* claim."""
    best_val, best_f = -1e9, 0
    for i in range(0, scan, chunk):
        peak, arg = (W_dec[i:i + chunk] @ W_U).max(dim=1)
        for j in peak.argsort(descending=True).tolist():
            piece = tok.convert_ids_to_tokens([int(arg[j])])[0]
            if piece.startswith("Ġ") and piece[1:].isalpha() and len(piece) >= 5:
                if peak[j].item() > best_val:
                    best_val, best_f = peak[j].item(), i + j
                break
    return best_f


def encode(resid, f):
    return torch.relu((resid - b_dec) @ W_enc[:, f] + b_enc[f])


def load_text(n_chunks=N_CHUNKS, chunk=CHUNK):
    """Real text. wikitext if available (it is, cached); otherwise bail loudly rather than
    quietly substituting synthetic text, which would make the whole test meaningless."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:4000]")
        blob = "\n".join(t for t in ds["text"] if len(t.strip()) > 200)
    except Exception as e:                                        # noqa: BLE001
        print(f"  ! could not load wikitext ({type(e).__name__}). "
              f"Install `datasets`; this gate needs REAL text, not a synthetic stand-in.")
        sys.exit(2)
    ids = tok(blob, return_tensors="pt").input_ids[0]
    need = n_chunks * chunk
    if ids.numel() < need:
        print(f"  ! only {ids.numel()} tokens available, need {need}")
        sys.exit(2)
    return ids[:need].view(n_chunks, chunk)


@torch.no_grad()
def sweep(f_main, f_other, batches):
    """One forward pass per chunk yields BOTH the feature activations (from hidden_states) and
    the specificity of the next-token distribution (from logits) at every position.

    POSITION 0 OF EVERY CHUNK IS DROPPED. GPT-2's first position is an attention sink where
    many SAE features sit at a huge constant activation independent of the token. Leaving it in
    inverts this test's headline statistic (corr goes -0.345 -> +0.459) and, in rung 5, silently
    replaced the steering calibration with an artifact. See PITFALLS.md #24."""
    acts_main, acts_other, specs = [], [], []
    tgt = torch.tensor(TARGET)
    for row in batches:
        out = model(row.unsqueeze(0).to(dev), output_hidden_states=True)
        resid = out.hidden_states[LAYER][0]                       # resid_pre of block LAYER
        lg = out.logits[0].float()                                # next-token logits, UNSTEERED
        acts_main.append(encode(resid, f_main)[1:].cpu())         # [1:] drops the sink
        acts_other.append(encode(resid, f_other)[1:].cpu())
        specs.append((lg[:, tgt].mean(-1) - lg[:, CTRL].mean(-1))[1:].cpu())
    return torch.cat(acts_main), torch.cat(acts_other), torch.cat(specs)


if __name__ == "__main__":
    f = pick_word_feature()
    TARGET = logit_lens_top(W_dec[f])
    print(f"Feature #{f} (GPT-2 L{LAYER} resid_pre), as picked by rung 5.")
    print("  decoder-derived label -- it promotes:", [tok.decode([t]) for t in TARGET])

    # a comparison feature: different, and also strongly word-promoting, so null (2) is fair
    f_other = pick_word_feature(scan=16000) if pick_word_feature(scan=16000) != f else f + 1
    print(f"  comparison feature for null (2): #{f_other}")

    print(f"\nStreaming {N_CHUNKS * CHUNK:,} tokens of wikitext "
          f"(no steering anywhere; position 0 of each chunk dropped -- attention sink)...")
    a_main, a_other, spec = sweep(f, f_other, load_text())

    top_main = a_main.topk(TOP_K).indices
    top_other = a_other.topk(TOP_K).indices
    g = torch.Generator().manual_seed(0)
    rand_idx = torch.randperm(spec.numel(), generator=g)[:TOP_K]

    s_signal = spec[top_main].mean().item()
    s_rand = spec[rand_idx].mean().item()
    s_other = spec[top_other].mean().item()
    nz = (a_main > 0).float().mean().item()

    print(f"\n  feature fires (act>0) at {nz:6.1%} of positions")
    print(f"  observational specificity of the feature's OWN tokens, no steering:")
    print(f"    at its top-{TOP_K} firing positions : {s_signal:7.3f}   <- signal")
    print(f"    at {TOP_K} random positions         : {s_rand:7.3f}   <- null (1)")
    print(f"    at feature #{f_other}'s top-{TOP_K}      : {s_other:7.3f}   <- null (2)")

    # Continuous version. NOTE: correlating over ALL positions is wrong here -- the feature
    # is zero at ~98% of them, so the statistic is dominated by a constant and lands at ~0.000
    # regardless of the truth. Among ACTIVE positions is the question that has content: when
    # the feature speaks louder, does its signature get stronger?
    act_f, spec_f = a_main.float(), spec.float()
    live = act_f > 0

    def corr(x, y):
        return float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std() + 1e-9))

    r_all = corr(act_f, spec_f)
    r_live = corr(act_f[live], spec_f[live]) if live.sum() > 30 else float("nan")
    s_live, s_dead = spec_f[live].mean().item(), spec_f[~live].mean().item()
    print(f"    corr over all {spec.numel():,} positions        : {r_all:+.3f}  "
          f"(uninformative -- feature is 0 at {(~live).float().mean():.1%} of them)")
    print(f"    corr among the {int(live.sum()):,} ACTIVE positions   : {r_live:+.3f}")
    print(f"    mean specificity  active {s_live:+.3f}  vs  inactive {s_dead:+.3f}")

    worst_null = max(s_rand, s_other)
    margin = s_signal - worst_null
    ok = margin > 1.0 and r_live > 0.10 and (s_live - s_dead) > 0.5
    print(f"\n  margin over the worse null: {margin:+.3f}")
    print("\n" + ("  GREEN -- the decoder-derived label describes where the feature actually\n"
                  "  fires, with no steering involved. Rung 5's interpretation survives an\n"
                  "  independent line of evidence."
                  if ok else
                  "  RED -- the label does NOT predict the feature's natural behaviour. Rung 5's\n"
                  "  reading may be an artifact of the intervention. Investigate before trusting\n"
                  "  any tool that reports this feature's meaning the same way."))
    sys.exit(0 if ok else 1)
