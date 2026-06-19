# Trackastra improvements - plan & tracker

Living checklist. After **every** step: commit (concise, single-line message) and run the
gate command to confirm training is unaffected.

## Gate command

```
python train.py -c configs/vanvliet.yaml --batch_size 32 --logger wandb \
  --name impro_<slug> --epochs 5 --train_samples 1024
```

~1 min/run, logged to wandb so steps are comparable. `<slug>` is per-step (see below).
Gate checks: imports, model builds, forward/backward runs, loss is finite (not parity).

## Status legend

`[ ]` todo  `[~]` in progress  `[x]` done (committed + gated)

---

## Step 0 - baseline gate  `[x]`

Ran on `improvements` HEAD. Reference: 0.35 min, val_loss 0.225, train_loss_epoch 0.137,
GPU 27.9 GB. wandb: sigmoidfroid/trackastra-new.

## Step 1 - fix pre-norm residual  `[x]`  (improv #1)  -> gated: val_loss 0.243, 0.35 min

`EncoderLayer.forward` / `DecoderLayer.forward` (`trackastra/model/model.py:64-77,123-137`)
overwrite `x` with `norm1(x)` before the residual, so the skip carries the normalized tensor,
not the raw stream. Fix to true pre-norm:

```python
x = x + self.attn(self.norm1(x), ...)   # encoder
x = x + self.mlp(self.norm2(x))
# decoder: h = self.norm1(x); x = x + self.attn(h, self.norm2(y), self.norm2(y), ...)
```

No new params (checkpoints still load), only the compute graph changes.

- Commit: `Fix pre-norm residual path in encoder/decoder layers`
- Gate: `--name impro_prenorm`

## Step 2 - rope canonical, fully remove bias path  `[x]`  (improv #3)  -> gated: val_loss 0.248

rope cross-check vs muvit: `_rotate_half` (adjacent-pair) + `repeat_interleave(2)` cos/sin
matches muvit's `rotate_half`/`apply_rotary_pos_emb` exactly. Differences are only frequency
parameterization and a uniform `1/sqrt(L)` scale that preserves RoPE's relative-position
property -> no functional rope fix needed. Removed `RelativePositionalBias` + all `"bias"` mode
branches; kept now-unused n_spatial/n_temporal kwargs for checkpoint-config compat.

- Make `"rope"` the default in `EncoderLayer`/`DecoderLayer` (already default on
  `TrackingTransformer`). Keep `"none"` to disable positional info.
- **Fully remove** the `"bias"` mode (no backward-compat shim): delete `RelativePositionalBias`,
  the `pos_bias` branch in `RelativePositionalAttention.__init__`/`build_attn_mask`, and the
  `"bias"` literal from all `Literal[...]` type hints / config defaults.
- Cross-check `trackastra/model/rope.py::RotaryPositionalEncoding` against the trusted axis-rope
  in `~/python/muvit-dev/muvit/rope.py` (`_rotate_half` interleave, per-axis freq split,
  `0.5*pi` scaling). Fix trackastra's only if it diverges.

- Commit: `Default to rope attention, remove learned bias path`
- Gate: `--name impro_rope`

## Step 3 - sparse kNN attention  `[x]`  (new module)  -> gated

Results (batch 32, train_samples 1024): dense val_loss 0.217 @ 32.5 GB / 0.35 min;
sparse K=16 val_loss 0.217 @ **17.2 GB** / 0.47 min. Same loss at ~47% less GPU mem
(headroom for larger batches). The slight wall-clock cost is the gradient-checkpoint
recompute. Parity tests (`tests/test_sparse_attn.py`, K>=N == dense) pass for v0/v1.

Note: naive fixed-K gather stores `k_g`/`v_g` (B,H,N,K,hd) per layer -> OOM at batch 32
vs fused flash SDPA. Fixed by gradient-checkpointing the sparse core in training (recompute
in backward); eval/inference runs it directly under no_grad.

New file `trackastra/model/sparse_attn.py`. The dense mask is effectively local (each node only
attends within `cutoff_spatial` + temporal window), so a fixed kNN neighbor list makes attention
O(N*K) instead of O(N^2) (~125x fewer entries at N=2000, K=16).

- Precompute once per forward (mirrors the shared-mask precompute in `model.py:403-409`):
  from `coords`+`padding_mask`, take the top-K nearest neighbors -> `nbr_idx (B,N,K)`.
  `K = max_neighbors` (new config, **default 16**). Any slot whose neighbor distance exceeds
  `max_dist` (= `cutoff_spatial`) gets `nbr_idx = -1` (sentinel), as do padded slots; nodes can
  thus have fewer than K real neighbors. Mask is `nbr_valid = nbr_idx >= 0`; gather uses
  `nbr_idx.clamp(min=0)` and softmax sets invalid slots to -inf.
- `SparseRelativePositionalAttention.forward`: reuse existing q/k/v projections + rope
  (no new params - a dense-trained checkpoint runs in sparse mode), gather k/v at `nbr_idx`
  -> `(B,nh,N,K,hd)`, scores `q.k_nbr/sqrt(hd)` -> `(B,nh,N,K)`, add gathered distance bias
  `(B,1,N,K)`, mask invalid slots, softmax over K, weight v.
- Wiring: add `attn_mode: Literal["dense","sparse"]="dense"` + `max_neighbors` to
  `TrackingTransformer.config`; precompute `nbr_idx` in `forward`, pass through layers.
  Default `dense` = bit-for-bit current behavior; sparse is opt-in via `--attn_mode sparse`.
- Parity test `tests/test_sparse_attn.py`: with `K >= N`, sparse output == dense (atol ~1e-5).

- Commit: `Add opt-in sparse kNN attention (sparse_attn module)`
- Gate twice: `--name impro_sparse_dense`, then `--attn_mode sparse --name impro_sparse_knn`
  (log speed/mem delta).

## Step 4 - cheap wins  `[ ]`

- De-dup the double `cdist` in `build_attn_mask` (`model_parts.py:244,254`, `attn_dist_mode="v0"`
  computes the distance twice).
  - Commit: `Avoid duplicate cdist in attn mask build` - Gate: `--name impro_dedup_cdist`
- Vectorize `normalize_output`'s per-batch Python loop (`model.py:459`).
  - Commit: `Vectorize normalize_output over batch` - Gate: `--name impro_vec_normout`

---

## Done (already on `improvements`)

- improv #2 (precompute masks): the additive attn mask is built once and shared across layers
  for rope/none modes (`model.py:403-409`). Only the cdist de-dup in Step 4 remains.
- Inference speedup (fast_regionprops backend, COO edge accumulation), training attention-mask
  precompute - commit a306769.
</content>
