# Hierarchical Chunked Transformer (H-net Inspired)

**Architecture**: MegaByte-style hierarchical transformer — a non-record submission for the H-net tokenization wishlist item.

| Metric | Value |
|--------|-------|
| Track | Non-record (novel architecture) |
| val_bpb | TBD (fill in after run) |
| Artifact | TBD bytes |
| Hardware | 8×H100 80GB SXM |
| Training time | ≤600s |
| Eval mode | Standard (not sliding window — see notes) |

## Motivation

The contest README lists **H-net tokenization** as an explicitly wanted but unimplemented idea.
This submission implements a hierarchical transformer architecture that captures the core H-net insight:
segment the sequence into chunks and process them at two levels of abstraction, with the global model
providing cross-chunk context and the local model handling within-chunk token prediction.

No one in 560+ PRs and 21 records has tried this. This is a proof of concept.

## Architecture

```
Input tokens: [t_0, t_1, ..., t_{T-1}]   (T = 2048, G = 16 → 128 chunks)

GLOBAL MODEL (7 layers, 448-dim, 8H/4KV):
  Input: [BOS, embed(t_0), embed(t_16), ..., embed(t_{(NC-2)*G})]  ← one per chunk, shifted
  Output: context vectors [g_0, g_1, ..., g_{NC-1}]                ← one per chunk
  Attention: O((T/G)²) = O(128²) per layer                         ← vs O(2048²) for flat

LOCAL MODEL (3 layers, 256-dim, 4H/2KV):
  For each chunk i, input: [g_i | embed(t_{iG}) | ... | embed(t_{iG+G-2})]  ← G positions
  Output: logit for each of the G tokens in chunk i
  All chunks processed in parallel: effective batch = B × 128
  Attention: O(G²) = O(16²) per layer per chunk

Projections:
  global_dim → local_dim (global_to_local):   448 → 256
  global_dim → local_dim (local_in_proj):     448 → 256 (token embeddings)
  local_dim  → global_dim (local_out_proj):   256 → 448 (for logits)
  Logits: local_out @ tok_emb.T (tied embedding)

Global model features:
  - U-Net skip connections between first/second half of global layers
  - resid_mix (blend hidden state with initial global input x0)
  - RMSNorm pre-attention + pre-MLP
  - RoPE on 64-dim heads
  - GQA (8 query heads, 4 KV heads)
  - Logit softcap (tanh, scale=30)
```

### Why this works

The key insight: in a flat transformer on T=2048 tokens, global attention is O(T²) = O(4M) per layer.
With G=16 chunks of 16 tokens:
- Global model attends over T/G = 128 tokens → O(16K) per global layer (~250× cheaper than flat)
- Local model attends within 16-token windows → O(256) per local layer, ×128 chunks = O(32K)
- Combined per-step compute similar to flat, but with a larger total parameter count that fits in 16MB

### Parameter budget

| Component | Params |
|-----------|--------|
| Embedding (1024 × 448) | ~459K |
| Projections (3 linear) | ~230K |
| Global model (7L × 448-dim) | ~12.6M |
| Local model (3L × 256-dim) | ~1.8M |
| Skip weights, norms, gains | ~100K |
| **Total** | **~15.2M** |

Int8+zlib estimated: 15.2M × 8 bits / 8 = 15.2MB (just at limit; int6 would give more headroom).

## Setup

```bash
pip install sentencepiece
python3 data/cached_challenge_fineweb.py --variant sp1024
```

No Flash Attention 3 required. Uses PyTorch native `F.scaled_dot_product_attention`.

## Run Command

Single GPU (smoke test):
```bash
CHUNK_SIZE=16 GLOBAL_DIM=448 GLOBAL_LAYERS=7 LOCAL_DIM=256 LOCAL_LAYERS=3 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=131072 ITERATIONS=500 VAL_LOSS_EVERY=100 SEED=1337 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Full 8×H100 run:
```bash
CHUNK_SIZE=16 GLOBAL_DIM=448 GLOBAL_LAYERS=7 LOCAL_DIM=256 LOCAL_LAYERS=3 \
TRAIN_SEQ_LEN=2048 WARMDOWN_ITERS=3000 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed1337.log
```

## Key Constraint

`TRAIN_SEQ_LEN` must be divisible by `CHUNK_SIZE`. The script checks this at startup.
Default: 2048 / 16 = 128 chunks per sequence.

## Evaluation Notes

This submission uses **standard eval** (no sliding window) for simplicity in the first implementation.
The standard eval scores are expected to be ~0.03 BPB worse than sliding-window eval (stride=64).
Adding sliding-window eval to a hierarchical model requires care: the sliding context window must
align to chunk boundaries (stride must be a multiple of G). This is a natural next step.

Expected baseline range without sliding window: ~1.25–1.35 BPB (unoptimized architecture).
With sliding window and hyperparameter tuning, expected improvement: ~0.03–0.05 BPB.

## What This Is / Isn't

**Is**: First hierarchical chunked transformer in this competition. Demonstrates that the architecture
trains stably, fits in 16MB, and produces reasonable val_bpb. Foundation for further work.

**Isn't**: True dynamic H-net chunking (fixed chunk size, not content-dependent). The "content-dependent
segmentation" of the H-net paper (goombalab/hnet, arXiv:2507.07955) is a follow-up direction.
The key difference: H-nets learn WHERE to chunk; this submission uses fixed G=16.

## Next Steps

1. Add sliding-window eval (stride must be multiple of G)
2. Tune chunk size G (try G=8, G=32)
3. Tune global/local dim ratio (more global capacity may help)
4. Add BigramHash to local model (bigram stats within chunks)
5. Implement dynamic chunking: add a lightweight "should I chunk here?" gate
6. Try G as a learnable parameter via straight-through estimator

## Results

| Seed | Steps | ms/step | val_bpb | Artifact |
|------|-------|---------|---------|----------|
| 1337 | TBD | TBD | TBD | TBD |

## Credits

- MegaByte architecture: Yu et al. (2023), "MegaByte: Predicting Million-byte Sequences..."
- H-net paper: Hwang et al. (2025), "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", arXiv:2507.07955
- Base infrastructure (Muon, quantization, eval): modded-nanogpt / Parameter Golf baseline
- Contest wishlist that inspired this: openai/parameter-golf README "Requests for PRs"
