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
done in `ctcdata.md` Phase 6). The temporal axis is the open question: today RoPE is used for
both spatial coordinates and time, but time inside a training/inference window is a tiny fixed
discrete range (`0..window-1`, commonly 4-10 positions). It may be better to treat time as a
small categorical vocabulary with `window` learned embeddings and reserve RoPE for spatial
coordinates only.

- [x] Verify temporal cutoff wiring: additive PE uses `cutoff=window`, and both
      Encoder/Decoder layers pass `cutoff_temporal=window` into RoPE. Current tests cover the
      additive PE cutoff indirectly, but not the RoPE temporal cutoff explicitly; add a focused
      regression test before changing temporal embedding modes.
- [ ] Learned absolute temporal embedding (preferred first experiment): since the window length
      is fixed, replace only the temporal component of the additive `PositionalEncoding` with an
      `nn.Embedding(window, d_t)` and disable temporal RoPE; keep Fourier/RoPE for the spatial
      axes. Make this a parameter/flag with the current behavior as the default, e.g.
      `temporal_pos_mode={"rope","learned_abs"}`.
- [ ] Learned relative-time bias (second experiment): a T5/Swin-style learned bias indexed by
      delta-t over `[-window, window]` (a `2*window+1` table, per head or shared), added to
      attention logits; keep RoPE/Fourier for the spatial axes. This is more directly aligned
      with association scores depending on temporal offsets, but touches attention logits rather
      than only input embeddings.
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

## 3. Link-type-aware error logging (regular vs division links)

Today the only tracking-quality signals logged during training are `TRA/AOGM/LNK/DET`
(`_summarize_tracking_metrics`, `scripts/train.py`). These are aggregate scores: a model can
have a good TRA while systematically failing on the rare-but-important division links, and the
curves would not show it.

This is purely an EDGE problem. In the in-training validation the detections are the GT TRA
masks, so the predicted graph's nodes are identical to the GT nodes (perfect 1:1 match) and
`fp_nodes/fn_nodes/ns_nodes` are ~0 by construction. Every error is a linking error. (Node
errors only become nonzero in the offline `scripts/predict.py` path when a non-GT detection
folder like ST/SEG is used, which is the only place worth logging node counts.)

Hypothesis: splitting the FP/FN error *rate* by link type - continuation ("regular", a single
edge leaving a non-dividing node) vs division ("vision", one of the two edges leaving a parent
that splits) - exposes the failure mode that AOGM hides and lets us compare runs on the axis we
actually care about (divisions). There are two natural, complementary measurement points:

### 3a. Association-level, in `_common_step` (cheap, per-step, pre-solver proxy)

`_common_step` already computes the division split - it is used today only for loss
upweighting (`scripts/train.py`):

```python
block_sum1 = blockwise_sum_batched(A, timepoints, dim=-1, reduce="sum")  # row (out) degree
block_sum2 = blockwise_sum_batched(A, timepoints, dim=-2, reduce="sum")  # col (in)  degree
block_sum  = A * (block_sum1 + block_sum2)
normal_tracks   = block_sum == 2   # 1->1 continuation edges
division_tracks = block_sum > 2    # edges leaving a dividing parent
```

`division_tracks` IS the per-edge regular-vs-division classification, already on-GPU. With
`A_pred`, `A`, and the existing `mask_time`/`mask_valid` in the same scope, a link-typed FN/FP
rate is a few masked sums. Binarize the prediction with the same transform the loss uses
(sigmoid, or `blockwise_causal_norm` then `>0.5`), restrict to `mask_time & mask_valid`, then:

- FN(div) = `(A==1) & division & (pred<0.5)` over `(A==1) & division`
- FP(div): binarize the prediction, mark its source degree the same way (a second
  `blockwise_sum` on the binarized prediction), then `fp & pred_division` over `pred_division`.

Only the division rates are kept (regular links are the easy majority); the GT-division mask is
`division_tracks` and the predicted-division mask is computed from the thresholded prediction.

- [ ] Add a no-grad block in `_common_step` returning these counts; log epoch means
      (`self.log(..., on_epoch=True)`) as `{train,val}_assoc_{fn,fp}_div` plus a derived
      `{train,val}_assoc_f1_div` (division F1 from the pooled rates: precision = 1 - fp,
      recall = 1 - fn). Only division links are logged (regular links are the easy majority and
      uninformative). Guard `n==0` denominators (skip the log, do not emit NaN). Behind a
      default-on flag for wandb/tensorboard; it only adds scalars. See 3d for the naming.

Caveat (be explicit in the curves): this measures PRE-SOLVER association quality - thresholded
`A_pred` vs `A`, before candidate pruning + greedy/ILP. An edge can be correct here yet lost to
`max_distance`/kNN pruning or greedy competition (the "mean over windows dilutes it" effect in
`check_errors.py`). It is a representation-quality proxy, not the final tracking outcome. Each
GT edge is also counted once per sliding window it appears in (a position-weighted average),
not once per movie - fine for a consistent training curve.

### 3b. Tracking-level, parse the error report into `val_` metrics (real outcome, every N epochs)

The post-solver truth, logged from the same validation tracking run that already produces
`TRA/AOGM/LNK/DET`. `evaluate_ctc` already returns `fp_edges, fn_edges` from `CTCMetrics`, but
`_summarize_tracking_metrics` forwards only `TRA/AOGM/LNK/DET` - the edge counts are dropped
every epoch.

Parse the per-edge error breakdown and roll it into `val_` scalars. The classification already
exists: `link_error_report` emits `error_type` (fn/fp) + `gt_source_degree`/`pred_source_degree`
(+ a division-aware `category`), and the `matched` graphs carry the CTC edge flags and node
degrees. Split each error by source out-degree into regular (deg 1) vs division (deg >=2) and
divide by the matching denominator (edge totals counted from the matched graphs, since the error
report alone has no denominators):

  - `val_track_fn_div = #FN(gt_outdeg>=2)  / #GT division edges`
  - `val_track_fp_div = #FP(pred_outdeg>=2)/ #pred division edges`
  - `val_track_f1_div = F1(precision = 1 - fp_div, recall = 1 - fn_div)`

(Only division links are logged; regular-link rates are dropped as uninformative.)

(CTC's third edge error, `ws_edges` / `WRONG_SEMANTIC`, is dropped: a WS edge is a correctly
drawn edge whose division role flipped, which always co-occurs with the FP_div / FN_div edge
that changed the source degree - so it is redundant with the rates above for monitoring.)

- [ ] A `link_type_breakdown(matched)` helper (graph-only: uses `matched.gt_graph`/
      `matched.pred_graph` flags + `out_degree`, no model internals) returning the counts and
      rates above. `evaluate_ctc` already supports `return_matched=True`; thread a
      `link_breakdown` flag through `predict_and_evaluate` -> `log_tracking_metrics` (default
      off so `scripts/predict.py` and its tests are unchanged; on for the training val path),
      merge the dict into each movie row, and extend `_summarize_tracking_metrics` to
      `nanmean` the new columns (skip movies with 0 divisions: NaN, not 0, so a division-free
      movie does not read as perfect). Cheap enough to run every tracking epoch.
- [ ] While editing `_summarize_tracking_metrics`: log `val_LNK_ERR = 1 - LNK` instead of
      `val_LNK`. LNK saturates near 1.0, so a raw score curve compresses late-training gains;
      `1 - LNK` (lower = better) makes them legible and sits naturally beside the new
      lower-is-better FN/FP rates. Update the printed epoch summary line accordingly. (Same
      saturation argument applies to TRA/DET, but change only LNK for now unless asked.)
- [ ] Offline depth (prediction-time only): aggregate the existing `link_error_report`
      `category` + `rejection_reason` columns into a wandb `Table`/bar for `scripts/predict.py`
      runs (how many division FNs were `candidate_pruned` vs `below_greedy_threshold` vs
      `decoder_not_selected`). Needs `details` (candidate graph + weights) -> keep on the
      `--error-report` path, not the per-epoch validation.

The 3a-vs-3b gap is the diagnostic signal: a division edge that 3a scores correctly but 3b
counts as FN was lost in the solver, not the model - which points the fix at pruning/greedy
rather than the transformer.

### 3c. No custom charts (decided against)

Considered logging the errors as wandb CustomCharts: a time-indexed line
(`wandb.plot.line_series` over `t`, the live `check_errors_per_t.csv` view) and a per-link-type
PR curve (`wandb.plot.pr_curve(..., classes_to_plot=["reg","div"])`). Decided against:

- Every `wandb.plot.*` is a per-step SNAPSHOT - it does not trend across epochs in the scalar
  panel, so it cannot show whether divisions improve over training (the thing we actually care
  about). The scalar rates from 3a/3b already do that.
- They are wandb-only (no tensorboard/none equivalent), breaking the logger-agnostic rule the
  scalars follow.
- The per-time and threshold-sweep detail still exists offline in `scripts/check_errors.py`
  (`check_errors_per_t.csv` + the window-position analysis), the right place for a single-movie
  deep dive.

If a within-movie temporal view is ever wanted live, `line_series` is the right primitive for an
ordered time axis (a line, not a scatter). Not in scope now.

### 3d. Metric naming

All rates are trending scalars logged with `self.log`/`self.log_dict` (backend-agnostic:
tensorboard, wandb, csv, or none - exactly like `train_loss`/`val_loss`/`val_TRA`), underscore-
flat names. No wandb dependency, no custom charts (3c). Only DIVISION links are logged
(regular-link rates are dropped as uninformative); the two measurement stages get explicit,
symmetric tokens so the proxy and the truth never get confused:

- 3a (pre-solver association proxy, `_common_step`): `{train,val}_assoc_{fn,fp,f1}_div`.
- 3b (post-solver tracking truth, val tracking epoch): `val_track_{fn,fp,f1}_div`.

`assoc` = before pruning/greedy (model representation), `track` = after the solver (real
outcome); the gap between an `assoc_*_div` and its `track_*_div` localizes a fault to the model
vs the solver. F1 = harmonic mean of precision (1 - fp) and recall (1 - fn) on division links;
fn/fp are lower-is-better, f1 is higher-is-better. `_rate` is omitted (implicit). These sit
alongside the existing
`val_TRA`/`val_AOGM`/`val_DET` and `val_LNK_ERR` (= `1 - LNK`); those CTC scores keep their
established names (not renamed).

### Decision (2026-06-25) and DDP note

Confirmed: implement BOTH 3a (per-step association FP/FN by link type, in `_common_step`) and
3b (post-solver FP/FN by link type, parsed from the error report into `val_` metrics). 3a is
the cheap per-epoch curve; 3b is the ground-truth check on tracking epochs.

DDP correctness for 3a: do NOT `self.log` a per-step rate and skip when a batch has no
divisions - asymmetric metric keys across ranks deadlock the epoch-end sync. Instead accumulate
numerator/denominator counts per epoch (reset in `on_{train,validation}_epoch_start`), pool
with `self.all_gather(...).sum()` (called symmetrically over a fixed key list on every rank) in
`on_{train,validation}_epoch_end`, then log `num/den` once with `rank_zero_only=True`, skipping
only keys whose pooled denominator is 0.

- [ ] Tests: 3a on a tiny hand-built `A`/`A_pred`/`timepoints` with one continuation and one
      division parent, asserting each error lands in the right bucket and zero-division batches
      report no division rate (denominator 0, key skipped); 3b (`link_type_breakdown`) on a
      small synthetic `matched`-like graph (one continuation edge, one division parent)
      likewise. ruff.

## Implementation observations

Append short dated entries with results that affect later work (which variant won, by how
much, on which datasets, and any training-stability notes).
