# CLAUDE.md — Learn Fast drill: "Building GPT from Scratch"

I'm Dakshitha. This repo is my template copy of `thili-ai/learnfast-workbench` and it is what I submit
(`E:\projects\Build-GPT-from-Scratch` → `github.com/dakshitha0705/Build-GPT-from-Scratch`).
The full text of lessons 4–9 is in `course/`. My commit history is assessed: one clean commit per lesson.

## The lessons, in order
| # | Title | Course text | Notebook | Who |
|---|-------|-------------|----------|-----|
| 0 | Overview | course/00_overview.txt | — | done |
| 1 | What a language model is | — | work/01_what_is_a_language_model.ipynb | **I did it by hand** |
| 2 | Text to tokens | — | work/02_text_to_tokens.ipynb (downloads work/shakespeare.txt) | **I did it by hand** |
| 3 | Tokens to vectors | — | work/03_tokens_to_vectors.ipynb | **I did it by hand** |
| 4 | Self-attention | course/04_self_attention.txt | work/04_self_attention.ipynb | you |
| 5 | Look back, never forward (causal mask) | course/05_causal_mask.txt | work/05_causal_mask.ipynb | you |
| 6 | The transformer block | course/06_transformer_block.txt | work/06_transformer_block.ipynb | you |
| 7 | Assemble & train the GPT | course/07_assemble_and_train.txt | work/07_assemble_and_train.ipynb (writes work/gpt.pt) | you |
| 8 | Make it talk (decoding) | course/08_make_it_talk.txt | work/08_make_it_talk.ipynb | you |
| 9 | Ship your mini-GPT | course/09_ship_mini_gpt.txt | work/09_ship_mini_gpt.ipynb + work/mini_gpt.py | you |

**Do not modify or recreate my lesson 1–3 notebooks.** If they're not committed yet, leave them alone (don't
`git add` them in your commits — use explicit file paths). If `work/shakespeare.txt` is missing, download it with
the lesson-2 code (url:
`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`).

## Environment
- PyTorch on CPU in the `lfllm` workbench: `docker compose -f lfllm/docker-compose.yml up` (I keep it running),
  container `lfllm-workbench`, Jupyter at http://127.0.0.1:8889/lab?token=learnfast, `work/` mounted at `/home/learnfast/work`.
- Git Bash on Windows: `export MSYS_NO_PATHCONV=1` before any `docker exec` with Linux paths.
- Run a notebook (keeps outputs, which I submit):
  `docker exec -w /home/learnfast/work lfllm-workbench jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <file>.ipynb`
  (working directory = work/, so `open("shakespeare.txt")` and `gpt.pt` resolve inside work/).
- Write notebook builder scripts with the Write tool, not long shell heredocs (quoting breaks).
- Sizing: the course already sizes everything for CPU (lesson 7 = ~160k params, 400 AdamW steps, should take
  1–3 min). Keep the course's config exactly — do NOT scale anything up or down.

## One-time setup (first commit)
Append to `.gitignore` (only lines not already there): `course/`, `work/shakespeare.txt`, `work/gpt.pt`, `*.pt`.
Add a "Progress" checklist section (lessons 0–9, 0–3 already ticked) to the end of `work/README.md` — keep its
existing content. Create `work/LOG.md`. Commit `.gitignore CLAUDE.md work/README.md work/LOG.md` as
"Setup: drill plan, progress checklist, ignore course text and data" and push.

## How to build each lesson (4 → 9)
1. Read its text in `course/`. Build the notebook: the course's code in small cells, short plain-English markdown
   notes between them in MY words (don't paste the lesson prose). Keep the course's names, seeds, configs.
2. Execute it in the container until it runs top to bottom.
3. **Reference check:** compare every printed line with the course output in the course text.
   Shapes, `True`, parameter counts (12608 in L6, 159937 in L7) must match exactly. Seeded numbers (attention
   matrices, losses, samples) may differ slightly across PyTorch versions — if they do, say so in a markdown cell,
   show course vs mine side by side, and confirm the behaviour is the same (rows sum to 1, mask is lower-triangular,
   loss starts ≈4.4 and falls to ≈2.2, samples have speaker tags/line breaks). Don't fake outputs.
   Note for L7: the course draws `xb` and `yb` with two different random indices in the untrained-loss check (so
   targets don't line up with inputs). Keep the course code as-is, and add one markdown line pointing this out.
4. End the notebook with `## Test yourself` (3 questions + 2–4 line answers) and `## My notes` (first person,
   1–2 lines: the key idea + anything surprising).
5. Update `work/LOG.md` (date, what was built, key numbers, `- Learned:` line in first person) and tick the lesson
   in `work/README.md`.
6. Commit only that lesson's files and push, e.g.
   `git add work/04_self_attention.ipynb work/README.md work/LOG.md && git commit -m "Lesson 4: Self-attention — <one-line result>" && git push`
7. Print, then go straight on to the next lesson WITHOUT waiting:
   ```
   ✅ LESSON N COMPLETE — <title>
   Mark it complete on askThili.
   Result: <check + key numbers>
   Summary: <4–6 plain lines>
   Test yourself: <3 questions>
   Answers: <short answers>
   Try it: <one change and what to expect>
   ```
Only stop and ask me if something is truly blocked.

## Lesson 9 module
Also put the model classes + `MiniGPT` into `work/mini_gpt.py` (loads `gpt.pt` from its own folder) and have the
lesson 9 notebook show it works by importing it. Commit `mini_gpt.py` with lesson 9.

## Portfolio project — NOT part of this run
The portfolio project at the end of lesson 9 is out of scope. Don't build it; I'll do it separately.

## Don'ts
- Don't commit `course/`, `shakespeare.txt` or any `.pt` file. Don't edit outside `work/`, `.gitignore`, `CLAUDE.md`.
- Never `git add .` or `git add work/` (my hand-made 1–3 notebooks might be uncommitted) — add files by name.
- Commit messages end with the usual Co-Authored-By trailer (keep it — honest about AI help).

## Finish
Print a table of lessons 4–9 with commit hashes, confirm everything is pushed, that `git ls-files` shows no
`course/`, `.pt` or `shakespeare.txt`, and remind me to mark lessons 4–9 complete on askThili.
