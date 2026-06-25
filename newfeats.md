# newfeats: model feature/representation ideas

New model-side features and efficiency ideas that are NOT mechanical refactors (those live in
`ctcdata.md`). Each section below is one semantically self-contained feature, with a hypothesis
and a default-off / fallback path so it can be adopted only after a training + metric
comparison.

## 1. Temporal positional encoding

The transformer uses two positional mechanisms (`trackastra/model/model.py`,
`trackastra/model/model_parts.py`, `trackastra/model/rope.py`):

- Additive Fourier `PositionalEncoding` on the input embeddings:
  `cutoffs=(window,) + (max_distance,)*coord_dim`, `n_pos=(pos_embed_per_dim,)*(1+coord_dim)`.
  One frequency group per axis (time + each spatial dim).
- RoPE in attention (`RelativePositionalAttention.rot_pos_enc`, mode `"rope"`):
  `RotaryPositionalEncoding(cutoffs=(cutoff_temporal,) + (cutoff_spatial,)*coord_dim)`, head
  dims split as time = `embed_dim//n_head - coord_dim*n_split`, each spatial dim = `n_split`
  with `n_split = 2*(embed_dim // (2*(coord_dim+1)*n_head))`. Time already gets the leftover
  head dims (slightly more than each spatial dim).

The spatial side is consistent (single `max_distance` drives mask + additive PE + RoPE + kNN -
done in `ctcdata.md` Phase 6). The temporal axis is the open question: it is a tiny FIXED
discrete range (a window has ~`window` timepoints, commonly 6-10) and the association decision
is fundamentally a function of the relative offset delta-t, so continuous Fourier/RoPE
frequencies are overkill for ~4-10 integer positions.

- [ ] Verify/fix the temporal cutoff wiring: the additive PE uses `cutoff=window`, but RoPE's
      `cutoff_temporal` defaults to 16. A cutoff of 16 while `window`~6-10 is a latent
      misalignment (temporal analog of the spatial issue). Confirm the Encoder/Decoder layers
      pass `cutoff_temporal=window` (or wire it). Cheap, low-risk; do first for a clean baseline.
- [ ] Learned relative-time bias (preferred): a T5/Swin-style learned bias indexed by delta-t
      over `[-window, window]` (a `2*window+1` table, per head or shared), added to attention
      logits; keep RoPE/Fourier for the SPATIAL axes. Hypothesis: matches or beats continuous
      temporal RoPE/Fourier since association is essentially per-delta-t. Behind a flag
      (e.g. `temporal_pos_mode={"fourier","rope","learned_bias"}`), default unchanged.
- [ ] Learned absolute temporal embedding (simpler variant): since the window length is fixed,
      replace only the temporal component of the additive `PositionalEncoding` with an
      `nn.Embedding(window, d_t)`; keep spatial Fourier. Same flag/default-off treatment.
- Not recommended on its own: allocating MORE RoPE dims to time. A handful of frequencies
  already disambiguate a tiny discrete range; the lever is expressiveness for relative time
  (the two items above), not dim count.

## 2. Skip image loading for intensity-free feature modes

For `wrfeat2_no_intensity`, `intensity_mean` is the ONLY feature that uses the real image, and
it is dropped at the stacking stage (`features_stacked_for` does `del channels[1]`). The other
regionprops2 properties are mask-derived (`equivalent_diameter_area` = area, `inertia_tensor` =
binary region moments, `border_dist` = edt of the mask). So for this mode the image is loaded,
normalized, and used to compute a feature that is immediately discarded - pure waste - and
`track(imgs=None, masks)` should be possible.

Two consequences make this cross-cutting, not a one-liner:

1. Extraction is currently mode-agnostic: `from_mask_img` always extracts the full
   `regionprops2` set (incl. `intensity_mean`) regardless of feature mode; the mode only
   matters later at stacking. Skipping the image needs a no-intensity property set and an
   `img=None` path through `from_mask_img` / `_regionprops_table` (including the
   `fast_regionprops` backend, which may expect an image).
2. It couples extraction (and its joblib cache) to the feature mode. Today `from_ctc` extracts
   everything once so modes can be switched without re-extracting and the cache is
   mode-independent; skipping intensity makes the cache mode-specific. Acceptable for the
   speedup, but a real consequence.

- [ ] Add a `regionprops2_no_intensity` property set and an `img=None` path in
      `from_mask_img` / `_regionprops_table` (handle the skimage and `fast_regionprops`
      backends when no intensity property is requested).
- [ ] `from_ctc` (and the Phase 7 raster loader in `ctcdata.md`): when the feature mode is
      `wrfeat2_no_intensity`, skip image load + normalize and pass `img=None`. This requires
      `from_ctc` to know the feature mode (today it is mode-agnostic) - thread it through.
- [ ] `Trackastra.track(imgs=None, masks)`: allow `imgs=None` when the model's feature mode is
      intensity-free; otherwise require `imgs` with a clear error. Make CLI `-i/--imgs`
      optional for such models.
- [ ] Tests + ruff; confirm a `wrfeat2_no_intensity` train + track run loads no images
      (and matches the previous features minus the intensity channel).

## Implementation observations

Append short dated entries with results that affect later work (which variant won, by how
much, on which datasets, and any training-stability notes).
