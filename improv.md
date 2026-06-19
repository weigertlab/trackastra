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
  `K = max_neighbors` (new config, **default 64**). Any slot whose neighbor distance exceeds
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

## Step 4 - cheap wins  `[x]`  -> gated: dense 32.5 -> 24.3 GB, val_loss 0.245, 0.35 min

Done: **broadcast the additive attn mask over heads.** With the per-head `pos_bias` removed
(Step 2), `build_attn_mask` has no head-dependent term, so it now returns `(B, 1, N, N)`
and SDPA broadcasts over heads -> n_head x smaller mask alloc. Cut the dense path from
32.5 GB to 24.3 GB at batch 32 with no loss/speed change.

Re-scoped (the two originally-listed items did not hold up):
- "double cdist in build_attn_mask" is NOT a duplicate: v0 uses the spatial-only distance for
  the cutoff and the full-coord distance (incl. time) for the bias; the default v1 already
  reuses `spatial_dist`. The per-layer recompute (improv #2) is already fixed by the shared
  precompute. Nothing to dedup.
- Vectorizing `normalize_output` was dropped: `blockwise_causal_norm` is inherently per-sample
  (variable block structure + padding), inference-only, and not worth the correctness risk.

---

## Step 5 - normalized association readout (logit_norm)  `[x]`  -> gated: val_loss 0.231

The association logits were a raw unscaled dot product `A = x.y` (logit variance grows with
d_model; the fp16 `clamp_` in train.py is a symptom). Now L2-normalize the head embeddings
and scale the cosine-similarity logits by a learned CLIP-style temperature
(`logit_scale = exp(param).clamp(max=100)`), default on via `logit_norm: bool = True`.
Dense gate: val_loss 0.231 @ 22.5 GB.

## Step 6 - drop soft distance bias from sparse path  `[x]`  -> gated: val_loss 0.240

`build_knn_index` no longer adds the `exp(-dist)` soft bias (a redundant dense-path crutch:
the kNN neighbourhood already encodes locality and rope encodes relative position). `nbr_bias`
is now purely the additive validity mask (0 real / -1e3 sentinel). Sparse no longer matches
dense numerically, so the parity tests were replaced by mechanism tests (`_sparse_attend` ==
full attention at K=N) + sentinel checks. Sparse K=64 gate: val_loss 0.240 @ 32.3 GB.

## Step 7 - dense `attn_dist_mode="none"` (boolean mask)  `[x]`  -> gated: val_loss 0.237

New dense option: drop the soft distance bias and build a **boolean** attn mask (True=attend,
hard spatial-cutoff + padded-keys only). A bool mask is n_bytes smaller than the float additive
mask and dispatches SDPA to the memory-efficient kernel (not flash - flash needs no mask).
Self-attention is forced on the diagonal so no query row is fully masked (a fully-masked bool
row softmaxes to NaN; the `-1e3` float mask was finite/NaN-safe). The float `forward` dtype cast
is now guarded to skip bool masks. Gate (`--attn_dist_mode none`): val_loss 0.237 @ 23.5 GB.

## Step 8 - fix dropped cutoffs_start in PositionalEncoding  `[x]`  -> gated: val_loss 0.224

Bug: `PositionalEncoding.__init__` bound `cutoff_start` in the zip but never passed it to
`_pos_embed_fourier1d_init`, so `feat_embed` (built with `cutoffs_start=0.01`, intending max
frequency 100) silently used the default 1 -> 100x too-low high-frequency resolution for the
shape/intensity features (esp. `intensity_mean`, `border_dist` in [0,1]). One-line fix +
regression test (`test_positional_encoding_cutoffs_start`). Only `feat_embed` was affected
(pos_embed doesn't pass cutoffs_start; rope has its own init). Gate: val_loss 0.224 @ 24.0 GB.

## Feature-extraction findings (not yet acted on)

- No per-feature standardization (`data.py:1231`): raw features span [0,1]..~1000s; the Fourier
  feat_embed absorbs scale (now that Step 8 restored its frequency range). log-scaling
  inertia_tensor/area could still help. Lower priority.
- `inertia_tensor` stored redundantly (symmetric: off-diagonals duplicated). 2D 4->3 unique,
  3D 9->6 unique. Wasted feature/embedding dims, not corrupting.
- `get_features` branch order (`wrfeat.py:561` vs `576`): `n_workers>0` checked before
  `pretrained_feats`, so parallel + pretrained silently uses shallow features. Pretrained-only.
- Backend parity (fast_regionprops vs skimage) verified clean in 2D+3D incl. key order. Not a bug.

## Backlog / ideas (not yet done)

- Decoder self-attention (currently cross-attn only) - lets candidate associations coordinate.
- Time-aware kNN (top-K per relative frame) - spatial-only top-K can miss the true successor
  in crowded frames.
- Spatial translation invariance: center spatial coords per window (only the absolute Fourier
  `pos_embed` breaks it; the distance terms are relative). 
- Raw (maskless) flash attention in dense mode: needs `build_attn_mask` to return None AND
  padding handled off the additive mask (loss-level mask or sequence packing). The bool mask
  (Step 7) gets the mem-efficient kernel but not flash.

## Known break (accepted)

The shipped pretrained models (`ctc`, `general_2d`) are **no longer compatible** with this
branch: Step 1's pre-norm residual fix changes the forward pass, so their weights (trained under
the old residual) produce degraded tracking. `tests/test_pretrained.py` integration tests fail
as a result. Accepted - models will be retrained later. (`test_train.py` error is an unrelated
network download failure.)

## Done (already on `improvements`)

- improv #2 (precompute masks): the additive attn mask is built once and shared across layers
  for rope/none modes (`model.py:403-409`). Only the cdist de-dup in Step 4 remains.
- Inference speedup (fast_regionprops backend, COO edge accumulation), training attention-mask
  precompute - commit a306769.
</content>
