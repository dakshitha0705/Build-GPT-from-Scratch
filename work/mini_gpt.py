"""mini_gpt.py — my character-level GPT from "Building GPT from Scratch", packaged for reuse.

Usage (from any folder):
    from mini_gpt import MiniGPT
    gpt = MiniGPT()                       # loads gpt.pt from this file's folder
    print(gpt.complete("ROMEO:", max_new_tokens=200))

Needs shakespeare.txt (to rebuild the 65-character vocabulary) and gpt.pt (trained in lesson 7),
both next to this file. Model code and config are exactly the course's lesson 9 code.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

# --- vocabulary: rebuilt from the training text, same as lessons 2-9 ---
text = open(HERE / "shakespeare.txt").read()
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

# --- model config (must match the config gpt.pt was trained with) ---
block_size = 32     # context length
n_embd     = 64     # embedding size
n_heads    = 4      # attention heads
n_layers   = 3      # transformer blocks


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj  = nn.Linear(n_heads * head_size, n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.sa  = MultiHeadAttention(n_heads, n_embd // n_heads)
        self.ff  = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)                       # (B, T, vocab)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


class MiniGPT:
    "Load a trained GPT and generate text from a prompt."
    def __init__(self, weights=HERE / "gpt.pt"):
        self.model = GPT(); self.model.load_state_dict(torch.load(weights)); self.model.eval()

    @torch.no_grad()
    def complete(self, prompt="\n", max_new_tokens=300, temperature=0.8, top_k=20):
        idx = torch.tensor([encode(prompt)], dtype=torch.long)
        for _ in range(max_new_tokens):
            logits, _ = self.model(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            kth = torch.topk(logits, top_k)[0][:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return decode(idx[0].tolist())


if __name__ == "__main__":
    torch.manual_seed(1337)
    print(MiniGPT().complete(prompt="ROMEO:", max_new_tokens=200))
