# microgpt

A minimal GPT trained from scratch — in a single Python file with no dependencies, or as a PyTorch Colab notebook with GPU support.

Based on [microgpt.py](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py) by Andrej Karpathy.

The updated `microgpt-colab.ipynb` and `gpt.py` took significant inspiration from [`model.py`](https://github.com/EN10/modded-llama2.c/blob/main/model.py) and [`train.py`](https://github.com/EN10/modded-llama2.c/blob/main/train.py) from [EN10/modded-llama2.c](https://github.com/EN10/modded-llama2.c).

---

## `microgpt.py` — Pure Python, no dependencies

Trains a tiny GPT on ~32,000 first names entirely in plain Python (no PyTorch, no NumPy).

```bash
python microgpt.py
```

`input.txt` is downloaded automatically from [karpathy/makemore](https://github.com/karpathy/makemore) on first run. After 1000 training steps the model generates 20 hallucinated names.

### How it works

1. **Dataset** — Character-level names loaded from `input.txt`
2. **Tokenizer** — Vocabulary built dynamically from the dataset, plus a special BOS token
3. **Autograd** — A minimal scalar `Value` class that builds a computation graph for backpropagation
4. **Model** — GPT weights: token/position embeddings, multi-head attention projections, MLP layers
5. **Forward pass** — For each token: embed → RMSNorm → multi-head attention → MLP → logits
6. **Training** — Cross-entropy loss, backprop through the scalar graph, Adam updates with linear LR decay
7. **Inference** — Greedy or sampled decoding to generate new names

---

## `microgpt-colab.ipynb` — PyTorch + Colab T4 GPU

[microgpt-colab.ipynb](microgpt-colab.ipynb) is a PyTorch version of the same model, designed to run on a free Colab T4 GPU and trained on short story snippets instead of names.

> **Before running:** go to **Runtime → Change runtime type → T4 GPU**.

### Differences from `microgpt.py`

| | `microgpt.py` | `microgpt-colab.ipynb` |
|---|---|---|
| Backend | Pure Python | PyTorch |
| Hardware | CPU | T4 GPU (Colab) |
| Dataset | Names (~32k names, GitHub) | TinyStories (1000 stories, HuggingFace) |
| Vocabulary | Dynamic (from dataset) | Fixed 74-char ASCII |
| Output | Hallucinated names | Hallucinated story snippets |

### Model configuration

| Hyperparameter | Value | Role |
|---|---|---|
| `n_layer` | 5 | Number of transformer blocks |
| `n_embd` | 128 | Embedding / hidden dimension |
| `block_size` | 256 | Max context length (tokens) |
| `n_head` | 8 | Number of attention heads |
| `batch_size` | 32 | Sequences per gradient step |
| `grad_accum_steps` | 4 | Gradient accumulation steps |

### Architecture and training changes (inspired by [EN10/modded-llama2.c](https://github.com/EN10/modded-llama2.c))

The updated notebook and `gpt.py` adopt a Llama-style design in place of the original minimal GPT:

**Architecture**
- **RMSNorm** instead of LayerNorm — applied before each sub-layer (pre-norm)
- **RoPE** (Rotary Position Embeddings) instead of learned position embeddings
- **relu²** activation (`F.relu(...).square()`) in the MLP instead of GELU
- **Weight tying** — the token embedding matrix is reused as the language model head
- **Padded vocab** — vocabulary size padded to a multiple of 128 for GPU efficiency
- **KV cache** — inference uses a per-layer key/value cache for efficient token-by-token generation
- **Flash attention** — `F.scaled_dot_product_attention` with `is_causal=True` during training

**Training**
- **AdamW with param groups** — embeddings use `weight_decay=0.0`; all other matrices use `weight_decay=0.1`
- **Gradient accumulation** — 4 micro-steps per optimizer update (effective batch = 32 × 256 × 4 = 32,768 tokens)
- **Mixed precision** — `torch.amp.autocast` with `float16` and a `GradScaler`
- **Gradient clipping** — global norm clipped to 1.0
- **LR schedule** — constant at `base_lr=5e-4` for the first 90% of steps, then cosine cooldown to `min_lr=5e-5`
- **Zero-init output projections** — `attn_wo` and `mlp_fc2` are initialised to zero for training stability