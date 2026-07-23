# Rung 6 — Lenses

## The idea

A **lens** maps a residual-stream vector into vocabulary space, so you can *read what a layer
is thinking in words*. The **logit lens** (unembed the residual directly) is the crude
version; the **tuned lens** learns an affine correction; the **Jacobian lens** (2026) reads
the *verbalizable* subspace — what the model is *poised to report* — and is the tool behind the
"global workspace" claim. This rung is where the ladder meets the frontier of 2026.

## Read

- **Primary:** Gurnee, Sofroniew, Pearce et al., *Verbalizable Representations Form a Global
  Workspace in Language Models* (Anthropic, 2026) — the Jacobian lens.
- **Lineage:** *logit lens* (nostalgebraist, 2020); Belrose et al., *Tuned Lens* (2023).

## Build

Open tooling — this rung runs on **our companion code**:
[`jacobian-lens`](https://github.com/m9h/jacobian-lens) +
[`jlens-lab`](https://github.com/m9h/jlens-lab), with fitted lenses on the
[Hub](https://huggingface.co/mhough/olmo3-jacobian-lenses). Open models: **OLMo-3 / Qwen**.

1. Apply the logit lens, the tuned lens, and the Jacobian lens to the same OLMo-3 prompt;
   compare what each reads at mid layers.
2. Read a *covert* concept: a passage that implies a concept without naming it, and watch it
   appear in the workspace band but not the output.

## The control

Three nulls ship with this one, all in `jlens-lab` (this is what the technique *is*):
1. **Randomization** — randomize the trained blocks; the lens must read out ~nothing (a real
   lens needs learned weights; ours passes, 0.0003 vs 0.34).
2. **Distance null** — for any geometry claim, compare to a matrix depending only on layer
   separation (it reproduces most apparent "structure").
3. **Logit-lens baseline** — always check whether the plain logit lens already does it (its
   `pass@k` metric can *reward noise*).

## Toward the recent papers

The Jacobian lens is the instrument for rung 8's workspace and metacognition results. Master
the three nulls here and you can read — and check — the flagship 2026 claim yourself.
