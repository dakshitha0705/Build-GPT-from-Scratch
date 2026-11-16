# Learning log — Building GPT from Scratch

One entry per lesson: what I built, the key numbers, and what I learned.

## 2026-11-16 — Setup
- Lessons 0–3 done by hand earlier (overview, LM loss, char tokenizer, embeddings).
- Added a progress checklist to `README.md`, this log, and ignored the course text, `shakespeare.txt` and `.pt` weights.

## 2026-11-16 — Lesson 4: Self-attention
- Built `04_self_attention.ipynb`: one attention head by hand (query/key/value, scaled scores, softmax, weighted sum of values).
- Numbers: q/k/v `(4, 8, 16)`, weights `(4, 8, 8)`, every row sums to 1 = `True`, output `(4, 8, 16)`; attention matrix identical to the course (max diff 0.0, torch 2.15.0).
- Learned: attention is a learned weighted average — and without a mask every token can peek at the future.

## 2026-11-16 — Lesson 5: Look back, never forward (causal mask)
- Built `05_causal_mask.ipynb`: masked the attention scores with `torch.tril` → `-inf` before softmax.
- Numbers: before/after matrices identical to the course (max diff 0.0); all future weights 0 = `True`; rows still sum to 1; row 0 = `[1.0, 0.0, …, 0.0]`.
- Learned: the mask must go before the softmax so the allowed weights re-normalise — that one line is what makes a GPT a decoder.
