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

## 2026-11-16 — Lesson 6: The transformer block
- Built `06_transformer_block.ipynb`: `Head`, `MultiHeadAttention`, `FeedForward`, `Block` (pre-norm + residuals).
- Numbers: input `(4, 8, 32)` → output `(4, 8, 32)`; **12608** parameters (matches course). By hand: heads 3072, proj 1056, feed-forward 8352, layernorms 128.
- Learned: a block is "communicate (attention), then think (feed-forward)", each added onto x — same shape in and out, so blocks stack.

## 2026-11-16 — Lesson 7: Assemble & train the GPT
- Built `07_assemble_and_train.ipynb`: full `GPT` (embeddings → 3 blocks → ln_f → lm_head), 400 AdamW steps, saved `gpt.pt` (not committed).
- Numbers: **159937** parameters and untrained loss 4.450 (both exact); val loss 4.409 → 2.177 at step 400 (course 2.177). Max gap vs course curve 0.002 (torch 2.15.0); sample matches the course for 27 chars then drifts. Training took ~6 s on CPU.
- Noted: the course's untrained-loss check draws `xb` and `yb` from two different random indices, so targets don't line up — kept as-is.
- Learned: a GPT is just the blocks stacked between an embedding and a vocab projection — 400 steps already teach it line breaks, capitals and word shapes.

## 2026-11-16 — Lesson 8: Make it talk (decoding)
- Built `08_make_it_talk.ipynb`: loaded `gpt.pt` and wrote one `sample()` with temperature, top-k and top-p (nucleus).
- Numbers (distinct chars / real-word share): T=0.5 → 20 / 59%, T=1.4 → 42 / 14%, top-k 10 → 27 / 35%, top-p 0.9 → 36 / 40%.
  Samples share 2–54 starting characters with the course, then drift (tiny weight differences from L7), and often re-sync later.
- Learned: the trained model is fixed — decoding settings alone move it from repetitive (low T) to gibberish (high T); top-k/top-p cut the nonsense tail.

## 2026-11-16 — Lesson 9: Ship your mini-GPT
- Built `09_ship_mini_gpt.ipynb` and `mini_gpt.py`: `MiniGPT` loads `gpt.pt` and has one method, `complete(prompt)` (temperature 0.8, top-k 20).
- Numbers: `ROMEO:` + 200 chars = 206; module has 159937 params, weights identical to the notebook class, same seed → same text (`True`). Runs from any folder (`python work/mini_gpt.py`).
- Sample shares 8 chars with the course, then drifts and re-syncs (`dove she ill,`) — same tiny L7 weight differences.
- Portfolio project: not done here — I'll build it separately.
- Learned: a whole GPT ships as ~100 lines of code + a 0.7 MB weights file behind a prompt-in/text-out interface — the same shape as any LLM API.
