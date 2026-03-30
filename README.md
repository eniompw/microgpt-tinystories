# microgpt

A minimal, dependency-free GPT implementation in pure Python — train and run inference in a single file.

Based on [microgpt.py](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py) by Andrej Karpathy.

## Usage

```bash
python microgpt.py
```

~32,000 names are downloaded automatically from [karpathy/makemore](https://github.com/karpathy/makemore) on first run and cached in `input.txt`. After 1000 training steps the model generates 20 sample names.

## Colab notebook (PyTorch + GPU)

[microgpt-colab.ipynb](microgpt-colab.ipynb) is a PyTorch version designed to run on a free Colab T4 GPU.

Key differences from `microgpt.py`:

| | `microgpt.py` | `microgpt-colab.ipynb` |
|---|---|---|
| Backend | Pure Python (no deps) | PyTorch |
| Hardware | CPU | T4 GPU (Colab) |
| Dataset | Names (~32k, GitHub) | TinyStories (1000 stories, HuggingFace) |
| Vocabulary | Dynamic (from dataset) | Fixed 74-char ASCII |
| Output | Hallucinated names | Hallucinated story snippets |

> **Runtime:** Go to **Runtime → Change runtime type → T4 GPU** before running.

### Estimated runtime (Colab T4 GPU)

| Section | Estimate |
|---|---|
| Setup & imports | < 5 s |
| Dataset download (10 API calls × 100 rows) | ~20–30 s (first run only) |
| Tokenizer + model init | < 1 s |
| **Training** (1000 steps × up to 16 tokens each, tiny 16-dim model) | **~1–3 min on T4** |
| Inference (20 samples × ≤16 tokens) | < 5 s |

**Total: ~2–4 minutes on a Colab T4 GPU.**

- The model is extremely small (n_embd=16, n_layer=1), so GPU kernel launch overhead will dominate over compute.
- Training uses a Python `for` loop over tokens (no batching), which is the main bottleneck — on CPU this could stretch to 10–20 min.
- After the first run, `input.txt` is cached so the download step is skipped.

## How it works

1. **Dataset** — Load names from `input.txt` (downloaded automatically from GitHub if missing)
2. **Tokenizer** — Dynamic character vocabulary derived from the dataset plus a special BOS token
3. **Autograd** — A minimal scalar `Value` class that tracks a computation graph for backpropagation
4. **Model parameters** — Initialise GPT weights: token/position embeddings, attention projections, MLP weights
5. **Forward pass** — For each token: embed → RMSNorm → multi-head attention → MLP → logits
6. **Training** — 1000 steps of forward pass, cross-entropy loss, backprop, and Adam weight updates
7. **Inference** — Sample 20 names character-by-character using temperature-scaled softmax