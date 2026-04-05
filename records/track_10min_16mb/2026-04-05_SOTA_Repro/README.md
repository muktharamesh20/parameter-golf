# SOTA Repro: AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112

Reproduction of the current SOTA ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), **1.1147 BPB**) by abaybektursun.
This folder is a clean copy of that record's `train_gpt.py` for local reproduction and as a baseline for further experiments.

## Setup

Flash Attention 3 (Hopper, H100 only) is required:
```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece zstandard
python3 -c "from flash_attn_interface import flash_attn_func; import sentencepiece, zstandard; print('deps OK')"
```

Download data (run from repo root):
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

## Run Command

Seed 1337:
```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed1337.log
```

Seed 42:
```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed42.log
```

Seed 2025:
```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=2025 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed2025.log
```

## Architecture

Identical to PR #1019. See [that record's README](../2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md) for full architecture table.

Key techniques (all proven in prior records):
- **Full Hessian GPTQ** with AR self-generated calibration (no train/val data accessed post-600s)
- **XSA on all 11 layers** (zero params, forces cross-position mixing from layer 0)
- **BigramHash 3072×112** (hash table over token bigrams, ~0.5MB budget)
- **LeakyReLU(0.5)²** in MLP (replaces ReLU², −0.003 BPB)
- **Partial RoPE** (16/64 dims), **LN Scale** (1/√layer+1)
- **EMA(0.997) + SWA(every 50)** weight averaging
- **Parallel Muon + Parameter Banking** optimizer
- **LZMA preset=9** compression
- **Sliding window eval** (stride=64)

## Results

| Seed | Steps | ms/step | val_bpb | Artifact |
|------|-------|---------|---------|----------|
| 1337 | TBD | TBD | TBD | TBD |
| 42   | TBD | TBD | TBD | TBD |
| 2025 | TBD | TBD | TBD | TBD |
| **Mean** | | | **TBD** | |

## Credits

All techniques due to prior contributors — see PR #1019 and its lineage.
