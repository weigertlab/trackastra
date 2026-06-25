# newfeats: model representation experiments

Measured model-side experiments that are NOT mechanical refactors. Each item is an
ablation with a clear hypothesis and a fallback to the current behavior. Keep the data-layer
refactor and the `max_distance`/`max_neighbors` unification in `ctcdata.md` (Phase 6); this
file is for representational changes that need a training+metric comparison before adoption.

## Background: how positional encoding works today

The transformer uses two positional mechanisms (`trackastra/model/model.py`,
`trackastra/model/model_parts.py`, `trackastra/model/rope.py`):

- Additive Fourier `PositionalEncoding` on the input embeddings:
  `cutoffs=(window,) + (spatial_pos_cutoff,)*coord_dim`, `n_pos=(pos_embed_per_dim,)*(1+coord_dim)`
  (model.py:451). One frequency group per axis (time + each spatial dim).
- RoPE in attention (`RelativePositionalAttention.rot_pos_enc`, mode `"rope"`):
  `RotaryPositionalEncoding(cutoffs=(cutoff_temporal,) + (cutoff_spatial,)*coord_dim)`,
  head dims split as time = `embed_dim//n_head - coord_dim*n_split`, each spatial dim =
  `n_split` where `n_split = 2*(embed_dim // (2*(coord_dim+1)*n_head))` (model_parts.py:197-203).
  So time already receives the leftover head dims (slightly more than each spatial dim).

The hard spatial attention mask (`invalid = dist > spatial_pos_cutoff`, model.py:567) and the
sparse kNN radius use the same spatial cutoff. The spatial side is consistent and is being
renamed to a single `max_distance` in `ctcdata.md` Phase 6; the temporal side is the open
question here.

Key property of the temporal axis: it is a tiny FIXED discrete range. A window has ~`window`
timepoints (commonly 6-10), and the association decision is fundamentally a function of the
relative offset delta-t. Continuous Fourier/RoPE frequencies are overkill for ~4-10 integer
positions.

## Items

### 1. Verify and fix the temporal cutoff wiring (`cutoff_temporal` vs `window`)

The additive PE uses `cutoff=window` for the time axis, but RoPE's `cutoff_temporal` defaults
to 16 in `RelativePositionalAttention`. A cutoff of 16 while `window`~6-10 is a latent
misalignment (the temporal analog of the spatial 256-vs-128 issue).

- [ ] Confirm whether the Encoder/Decoder layers pass `cutoff_temporal=window` or leave the
      default 16. If unset, wire it to `window` (or to a single temporal parameter, mirroring
      the spatial `max_distance` unification).
- [ ] Cheap to verify and low-risk; do this before the larger experiments below so the
      baseline is clean.

### 2. Learned relative-time bias (preferred experiment)

Hypothesis: because association is essentially a per-delta-t decision over a small fixed range,
an explicit learned relative-time bias will match or beat continuous temporal Fourier/RoPE.

- Add a T5/Swin-style learned bias indexed by delta-t over `[-window, window]` (a small
  `2*window+1` table per head, or shared across heads), added to the attention logits. Keep
  RoPE/Fourier for the SPATIAL axes unchanged.
- Compare against the current temporal RoPE/Fourier on the standard val metrics
  (val_TRA / val_LNK / val_DET / val_AOGM).
- [ ] Implement behind a flag (e.g. `temporal_pos_mode={"fourier","rope","learned_bias"}`),
      default unchanged, so it is an opt-in ablation.

### 3. Learned absolute temporal embedding (simpler variant)

Hypothesis: since the window length is fixed, a learned absolute temporal embedding table
indexed by integer timepoint can replace the additive temporal Fourier component with no loss.

- [ ] Replace only the temporal component of the additive `PositionalEncoding` with an
      `nn.Embedding(window, d_t)`; keep spatial Fourier. Same flag/default-off treatment.

### Not recommended on its own

Allocating MORE RoPE dims to time: a handful of frequencies already disambiguate a tiny
discrete range, so adding temporal dims is low-value. The lever is expressiveness for relative
time (items 2-3), not dim count.

## Implementation observations

Append short dated entries with results that affect later work (which variant won, by how
much, on which datasets, and any training-stability notes).
