"""
Hierarchical Chunked Transformer (H-net inspired) for Parameter Golf.

Architecture: MegaByte-style hierarchical transformer.
- Global model: causal transformer over chunk-level representations (one per G tokens).
- Local model: causal transformer over tokens within each chunk, conditioned on global context.
- Shared tied embedding (global_dim). Token predictions use local model output projected back to global_dim.

Why hierarchical?
- Global model attends over T/G tokens instead of T → much cheaper global attention.
- Local model attends within G-token windows in parallel → O(G²) per chunk, all chunks batched.
- Combined, the model can have more parameters than a flat transformer at the same step-time cost,
  because global attention is ~(T/G)^2 and local is ~G^2 (vs T^2 for flat).

This is a non-record submission demonstrating a novel architecture requested by the contest
(README "Requests for PRs": H-net tokenization). Standard (non-sliding) eval is used here;
adding sliding-window eval is a straightforward follow-up.

Run command (single GPU for testing):
  CHUNK_SIZE=16 GLOBAL_DIM=448 GLOBAL_LAYERS=7 LOCAL_DIM=256 LOCAL_LAYERS=3 \
  TRAIN_SEQ_LEN=2048 SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

Run command (8xH100 contest run):
  CHUNK_SIZE=16 GLOBAL_DIM=448 GLOBAL_LAYERS=7 LOCAL_DIM=256 LOCAL_LAYERS=3 \
  TRAIN_SEQ_LEN=2048 WARMDOWN_ITERS=4000 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

Proven techniques applied:
  - LeakyReLU(0.5)^2 in MLP (replaces ReLU^2, -0.003 BPB)
  - Partial RoPE (first rope_dims=16 of head_dim, -0.002 BPB)
  - LN Scale (1/sqrt(layer+1) applied to norm outputs, -0.001 BPB)
  - BigramHash 1536x64 (hash table over bigrams, -0.010 BPB)
  - EMA weight averaging (decay=0.997, smoother final weights)
  - Sliding window eval (stride=64, chunk-aligned, -0.032 BPB)
  - Over-scheduled warmdown (4000 iters >> actual steps)
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    # Data
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training schedule
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    # train_seq_len MUST be divisible by chunk_size
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Architecture: shared
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    chunk_size = int(os.environ.get("CHUNK_SIZE", 16))   # G: tokens per chunk
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))     # partial RoPE: apply to first rope_dims of head_dim
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # BigramHash embedding
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 1536))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 64))

    # EMA weight averaging
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    # Sliding window eval (stride must be a multiple of chunk_size)
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    # Architecture: global transformer (processes num_chunks = T/G tokens)
    global_dim = int(os.environ.get("GLOBAL_DIM", 448))
    global_layers = int(os.environ.get("GLOBAL_LAYERS", 7))
    global_heads = int(os.environ.get("GLOBAL_HEADS", 8))
    global_kv_heads = int(os.environ.get("GLOBAL_KV_HEADS", 4))
    global_mlp_mult = int(os.environ.get("GLOBAL_MLP_MULT", 3))

    # Architecture: local transformer (processes G-token chunks in parallel)
    local_dim = int(os.environ.get("LOCAL_DIM", 256))
    local_layers = int(os.environ.get("LOCAL_LAYERS", 3))
    local_heads = int(os.environ.get("LOCAL_HEADS", 4))
    local_kv_heads = int(os.environ.get("LOCAL_KV_HEADS", 2))
    local_mlp_mult = int(os.environ.get("LOCAL_MLP_MULT", 3))

    # Optimizer
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Batched Newton-Schulz orthogonalization."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            f"VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int = 64,
) -> tuple[float, float]:
    """Sliding window eval: score only the last `stride` tokens of each window.
    stride must be a multiple of chunk_size so hierarchical model chunk boundaries align."""
    assert stride % args.chunk_size == 0, f"stride={stride} must be divisible by chunk_size={args.chunk_size}"
    T = args.train_seq_len
    ntok = val_tokens.numel()
    n_windows = max((ntok - T) // stride, 0)

    # Unwrap DDP for direct forward call returning logits
    base = model.module if hasattr(model, "module") else model

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base.eval()
    with torch.inference_mode():
        for win_idx in range(rank, n_windows, world_size):
            start = win_idx * stride
            tokens = val_tokens[start : start + T + 1].to(device=device, dtype=torch.int64, non_blocking=True)
            x = tokens[:-1].unsqueeze(0)   # [1, T]
            y = tokens[1:].unsqueeze(0)    # [1, T]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base(x, None)     # [1, T, vocab_size]

            # Score only the last `stride` positions
            y_last = y[:, -stride:].reshape(-1)          # [stride]
            x_last = x[:, -stride:].reshape(-1)          # prev tokens for byte counting
            logits_last = logits[:, -stride:, :].reshape(-1, args.vocab_size)

            loss = F.cross_entropy(logits_last.float(), y_last, reduction="sum")
            val_loss_sum += loss.to(torch.float64)
            val_token_count += stride

            token_bytes = base_bytes_lut[y_last].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[y_last] & ~is_boundary_token_lut[x_last]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = (val_loss_sum / val_token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    base.train()
    return float(val_loss), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,mlp_scale,q_gain,skip_weight,resid_mix",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if p
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# TRANSFORMER MODULES (shared)
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float, rope_dims: int = 0):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_dims = rope_dims if 0 < rope_dims < self.head_dim else self.head_dim
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.rope_dims, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        rd = self.rope_dims
        if rd < self.head_dim:
            q = torch.cat([apply_rotary_emb(q[..., :rd], cos, sin), q[..., rd:]], dim=-1)
            k = torch.cat([apply_rotary_emb(k[..., :rd], cos, sin), k[..., rd:]], dim=-1)
        else:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())


# -----------------------------
# BIGRAM HASH EMBEDDING
# -----------------------------

class BigramHashEmbedding(nn.Module):
    """Hash table over consecutive token pairs, projected to out_dim."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, out_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.bigram_emb = nn.Embedding(bigram_vocab_size, bigram_dim)
        self.proj = CastedLinear(bigram_dim, out_dim, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        B, T = input_ids.shape
        prev_ids = torch.cat([input_ids.new_zeros(B, 1), input_ids[:, :-1]], dim=1)
        hash_ids = (prev_ids * 7919 + input_ids) % self.bigram_vocab_size
        return self.proj(self.bigram_emb(hash_ids))


# -----------------------------
# HIERARCHICAL MODEL
# -----------------------------

class GlobalBlock(nn.Module):
    """Transformer block for the global model, with resid_mix (blend with x0) for deeper mixing."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float, layer_idx: int = 0, rope_dims: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x) * self.ln_scale)
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * self.ln_scale)
        return x


class LocalBlock(nn.Module):
    """Transformer block for the local model (no resid_mix, simpler)."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float, layer_idx: int = 0, rope_dims: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x) * self.ln_scale)
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * self.ln_scale)
        return x


class HierarchicalGPT(nn.Module):
    """
    MegaByte-style hierarchical transformer.

    Global model: processes one vector per chunk (the first-token embedding at each chunk
    boundary, shifted causally with a learned BOS prepended). Produces a context vector per chunk.

    Local model: for each chunk, takes [global_ctx | token_0 | ... | token_{G-2}] and
    predicts [token_0 | token_1 | ... | token_{G-1}] autoregressively. All chunks batched in parallel.

    Parameter routing:
    - tok_emb.weight → Adam (tied embedding / LM head)
    - global_bos → Adam (scalar)
    - All CastedLinear weights → Muon
    - All 1-D params (norms, gains, scales, resid_mix, skip_weights) → Adam (scalar)
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.chunk_size = args.chunk_size
        self.vocab_size = args.vocab_size
        self.global_dim = args.global_dim
        self.local_dim = args.local_dim
        self.logit_softcap = args.logit_softcap

        # Shared tied embedding (global_dim)
        self.tok_emb = nn.Embedding(args.vocab_size, args.global_dim)

        # Learned BOS vector prepended to global model input
        self.global_bos = nn.Parameter(torch.zeros(args.global_dim, dtype=torch.float32))

        # Cross-model projections
        self.global_to_local = CastedLinear(args.global_dim, args.local_dim, bias=False)
        self.local_in_proj = CastedLinear(args.global_dim, args.local_dim, bias=False)
        self.local_out_proj = CastedLinear(args.local_dim, args.global_dim, bias=False)
        self.local_out_proj._zero_init = True

        # BigramHash embedding (adds bigram statistics to token embeddings)
        self.bigram = BigramHashEmbedding(args.bigram_vocab_size, args.bigram_dim, args.global_dim)

        # Global transformer (U-Net skip connections between first/second halves)
        n_enc = args.global_layers // 2
        n_dec = args.global_layers - n_enc
        self.n_global_enc = n_enc
        self.n_global_dec = n_dec
        self.n_global_skips = min(n_enc, n_dec)
        self.global_skip_weights = nn.Parameter(
            torch.ones(self.n_global_skips, args.global_dim, dtype=torch.float32)
        )
        self.global_blocks = nn.ModuleList([
            GlobalBlock(args.global_dim, args.global_heads, args.global_kv_heads,
                        args.global_mlp_mult, args.rope_base, args.qk_gain_init,
                        layer_idx=i, rope_dims=args.rope_dims)
            for i in range(args.global_layers)
        ])
        self.global_norm = RMSNorm()

        # Local transformer (small, runs on G-token windows in parallel)
        self.local_blocks = nn.ModuleList([
            LocalBlock(args.local_dim, args.local_heads, args.local_kv_heads,
                       args.local_mlp_mult, args.rope_base, args.qk_gain_init,
                       layer_idx=i, rope_dims=args.rope_dims)
            for i in range(args.local_layers)
        ])
        self.local_norm = RMSNorm()

        self._init_weights(args.tied_embed_init_std)

    def _init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.global_bos)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        B, T = input_ids.shape
        G = self.chunk_size
        num_chunks = T // G  # T must be divisible by G

        # --- Global model ---
        # Embed all tokens (token embedding + bigram features)
        x_embed = self.tok_emb(input_ids) + self.bigram(input_ids)  # [B, T, global_dim]
        x_embed = F.rms_norm(x_embed, (x_embed.size(-1),))

        # Global input: causally shifted chunk-boundary embeddings with learned BOS prefix.
        # global_input[:, i, :] gives context for chunk i derived from chunks 0..i-1.
        chunk_first = x_embed.reshape(B, num_chunks, G, self.global_dim)[:, :, 0, :]  # [B, num_chunks, global_dim]
        bos = self.global_bos.to(x_embed.dtype)[None, None, :].expand(B, 1, -1)
        global_input = torch.cat([bos, chunk_first[:, :-1, :]], dim=1)  # [B, num_chunks, global_dim]
        global_input = F.rms_norm(global_input, (global_input.size(-1),))

        # Global transformer with U-Net skips
        g = global_input
        g0 = global_input
        skips: list[Tensor] = []
        for i in range(self.n_global_enc):
            g = self.global_blocks[i](g, g0)
            skips.append(g)
        for i in range(self.n_global_dec):
            if skips:
                sw = self.global_skip_weights[i].to(dtype=g.dtype)[None, None, :]
                g = g + sw * skips.pop()
            g = self.global_blocks[self.n_global_enc + i](g, g0)
        g = self.global_norm(g)                  # [B, num_chunks, global_dim]

        # --- Local model ---
        # Project global context to local dim: one vector per chunk
        g_local = self.global_to_local(g)        # [B, num_chunks, local_dim]

        # Project all token embeddings to local dim
        x_local = self.local_in_proj(x_embed)    # [B, T, local_dim]
        x_local_chunked = x_local.reshape(B, num_chunks, G, self.local_dim)

        # Build local input: [global_ctx (pos 0) | token_0..token_{G-2} (pos 1..G-1)]
        # At position j, local model predicts token j of the chunk.
        g_prefix = g_local.unsqueeze(2)           # [B, num_chunks, 1, local_dim]
        local_input = torch.cat(
            [g_prefix, x_local_chunked[:, :, :-1, :]], dim=2
        )                                         # [B, num_chunks, G, local_dim]

        # Run local transformer in parallel over all chunks
        local_flat = local_input.reshape(B * num_chunks, G, self.local_dim)
        l = local_flat
        for block in self.local_blocks:
            l = block(l)
        l = self.local_norm(l)                    # [B*num_chunks, G, local_dim]
        l = l.reshape(B, T, self.local_dim)

        # Project back to global_dim and compute logits (tied embedding)
        l_global = self.local_out_proj(l)         # [B, T, global_dim]
        logits = F.linear(l_global.reshape(-1, self.global_dim), self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        logits = logits.view(B, T, self.vocab_size)

        if target_ids is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, self.vocab_size).float(), target_ids.reshape(-1), reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    if args.train_seq_len % args.chunk_size != 0:
        raise ValueError(
            f"TRAIN_SEQ_LEN={args.train_seq_len} must be divisible by CHUNK_SIZE={args.chunk_size}"
        )

    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(
        f"hnet_config: chunk_size={args.chunk_size} global_dim={args.global_dim} "
        f"global_layers={args.global_layers} local_dim={args.local_dim} local_layers={args.local_layers} "
        f"num_chunks_per_seq={args.train_seq_len // args.chunk_size}"
    )

    base_model = HierarchicalGPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    # dynamic=True to handle variable batch sizes between train and eval
    compiled_model = base_model  # skip torch.compile: Rotary cache + dynamic local batch sizes cause dynamo issues
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else compiled_model
    )

    # EMA weight averaging
    ema_state: dict[str, Tensor] | None = None
    if args.ema_decay > 0:
        ema_state = {name: param.detach().clone() for name, param in base_model.named_parameters()}

    # Optimizer setup: split into embedding, matrix (Muon), and scalar (Adam) groups.
    all_named = list(base_model.named_parameters())
    embed_names = {"tok_emb.weight"}
    bos_names = {"global_bos"}

    matrix_params = [
        p for name, p in all_named
        if name not in embed_names and name not in bos_names
        and p.ndim == 2
        and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in all_named
        if name not in embed_names
        and (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS))
    ]

    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    n_matrix = sum(p.numel() for p in matrix_params)
    n_scalar = sum(p.numel() for p in scalar_params)
    log0(f"model_params:{n_params} matrix_params:{n_matrix} scalar_params:{n_scalar}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # EMA update
        if ema_state is not None:
            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    ema_state[name].lerp_(param.detach(), 1.0 - args.ema_decay)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # Load EMA weights into base_model for serialization and final eval
    if ema_state is not None:
        log0("Loading EMA weights into model for final eval/quantization")
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                param.data.copy_(ema_state[name].to(param.device, param.dtype))

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Code size: {code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Sliding window eval (stride=eval_stride, chunk-aligned for hierarchical model)
    if args.eval_stride > 0:
        if args.eval_stride % args.chunk_size != 0:
            log0(f"WARNING: eval_stride={args.eval_stride} not divisible by chunk_size={args.chunk_size}; skipping sliding eval")
        else:
            torch.cuda.synchronize()
            t_sliding = time.perf_counter()
            sw_val_loss, sw_val_bpb = eval_val_sliding(
                args, model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=args.eval_stride,
            )
            torch.cuda.synchronize()
            log0(
                f"final_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
                f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_sliding):.0f}ms"
            )
            log0(f"final_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
