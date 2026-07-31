# Rung 0 — Build the model first (prerequisite track)

## The idea

You cannot reverse-engineer what you cannot build. Before interpreting a transformer, spend a
few evenings *constructing* one: the residual stream, attention, the MLP, the training loop,
and — for the reasoning-model claims at rung 8 — an RL fine-tuning loop. Everything the later
rungs "read from" (rung 6) or "trace through" (rung 7) is something you will have written by
hand here. This rung is optional if you already know it, and load-bearing if you don't.

## Read / build (pick your depth)

- **Fastest path — nanoGPT.** Andrej Karpathy, [`nanoGPT`](https://github.com/karpathy/nanoGPT)
  and the *Let's build GPT from scratch* video ([Neural Networks: Zero to
  Hero](https://karpathy.ai/zero-to-hero.html)). Train a small GPT end to end; you will meet
  the residual stream and attention as code, not diagrams.
- **The thorough path — Build an LLM from Scratch.** Sebastian Raschka, *Build a Large Language
  Model (From Scratch)* (Manning) +
  [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch). Tokenizer →
  attention → pretraining → instruction fine-tuning, all from first principles. This is the
  cleanest bridge into rungs 2–5.
- **For post-training — the RLHF Book (free) and its course.** Nathan Lambert,
  *Reinforcement Learning from Human Feedback and LLM Post-Training*
  ([rlhfbook.com](https://rlhfbook.com), [repo](https://github.com/natolambert/rlhf-book),
  [course](https://rlhfbook.com/course)). Instruction tuning → reward models → rejection
  sampling → RL → direct alignment (DPO) → synthetic data → evaluation: 11 lectures with free
  videos and slides. This is the canonical open reference for the *training* side of everything
  rung 8 measures, written by someone who builds OLMo's post-training. Two pieces are near-
  required reading for rung 8: **Lecture 6 (Direct Preference Optimization)** and
  **Conversation 2, a case study in DPO for OLMo 3**.

  It is lectures and theory, with no coding exercises — which is exactly why it pairs well with
  this repo rather than overlapping it.

- **For actually running post-training — TRL and the Hugging Face *Training Agents* series.**
  [TRL](https://github.com/huggingface/trl) v1.0 is a unified post-training stack covering
  **SFT, reward modelling, DPO and GRPO** — i.e. every stage rung 8 measures, in runnable code.
  The [*Training Agents*](https://www.youtube.com/live/ztdTed5egrM) live classes teach it
  hands-on (class 2 distillation, class 3 reinforcement learning / GRPO), and the LLM Course has
  a written [Implementing GRPO in TRL](https://huggingface.co/learn/llm-course/en/chapter12/4)
  chapter.

  GRPO matters specifically for rung 8: it is the RLVR family, and our result is that **RLVR
  barely moves the J-space** (~6% from base) compared with SFT+DPO (~31%). To have a view on
  why, you need to know what GRPO actually does to a model.

- **For the reasoning claims — Build a Reasoning Model from Scratch.** Sebastian Raschka,
  *Build a Reasoning Model (From Scratch)* +
  [`rasbt/reasoning-from-scratch`](https://github.com/rasbt/reasoning-from-scratch). Chain-of-
  thought, verifiers, and RL fine-tuning built by hand — the machinery behind the
  society-of-thought / RL claims you will adjudicate at rung 8.
- **For hands-on investigation — Mike X Cohen, *50 ML projects to understand LLMs*.**
  [`mikexcohen/ML4LLM_book`](https://github.com/mikexcohen/ML4LLM_book) — 50 Colab projects that
  investigate transformer *mechanisms* by treating internals as data: attention patterns, layer
  dynamics, statistical/causal analysis, and activation manipulation. It's less "build the
  model" than "measure the model," so it's really a **companion project track that pairs with
  rungs 2–7** — reach for it whenever you want another angle of attack on a rung's technique.

## Build (the checkpoint)

You are ready to leave rung 0 when you can, from your own code:

1. point to *where in the residual stream* a given token's information lives;
2. write an attention head's QK and OV as two separate matrices and say what each does;
3. run one step of RL / preference fine-tuning and watch a reward move.

## The control

Even here, the reflex: after training your from-scratch model, verify it actually *learned* —
compare its next-token loss to a **unigram / bigram baseline**. A model that doesn't beat
bigram statistics has nothing worth interpreting. (The same instinct — "beat the trivial
baseline" — becomes the null at every rung above.)

## Toward the rest of the ladder

Rung 0 makes rungs 2 (residual stream), 4 (induction heads), and 5 (SAEs) concrete rather than
metaphorical, and Raschka's reasoning book is the direct on-ramp to rung 8's reasoning-model
claims. Build it once; read the rest of the field with the confidence of someone who knows
what is inside.
