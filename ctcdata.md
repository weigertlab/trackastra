# Tracking data overhaul plan

## Scope and rationale

`CTCData` currently combines CTC path discovery, TIFF loading, matching, lineage and
feature extraction, windows, augmentation, sampling, compression, conversion, and
caching. Its duplicate WRFeatures path also makes runtime-only changes such as
augmentation or `max_detections` invalidate expensive preprocessing caches.

The overhaul will replace `CTCData` with a format-neutral `TrackingSequence` data
model and a lightweight `TrackingData` PyTorch dataset. CTC ingestion remains a
`TrackingSequence.from_ctc` class method in `trackastra/data/data.py`; no separate CTC
module is introduced. Expensive, deterministic per-frame preprocessing is cached with
joblib, while windowing and stochastic training behavior remain outside the cache.
Backward compatibility with legacy feature modes, arguments, or cached datasets is
explicitly out of scope.

## Target structure

During Phases 1 and 2, the new implementation is developed in
`trackastra/data/datanew.py` while the current `data.py` remains unchanged as the
behavioral oracle. At the final migration gate, move `data.py` to `_legacy_data.py`
and rename `datanew.py` to `data.py`. This keeps the package runnable throughout the
incremental work and avoids temporary production imports from the legacy module.

The final `trackastra/data/data.py` will contain:

- `DetectionFrame`: immutable aligned per-detection arrays for one frame.
- `DetectionSeries`: frames originating from one detection folder.
- `TrackingSequence`: immutable preprocessed sequence and lineage supervision.
- `TrackingSequence.from_ctc`: CTC and simple-folder ingestion.
- `TrackingData`: windowing, sampling, augmentation, and tensor conversion.
- Association, collation, and small sampling helpers used by `TrackingData`.

After the final rename, `_legacy_data.py` remains temporarily as a behavioral oracle.
Production code must not depend on it; delete it once parity is established.

## Canonical data model

Define frozen dataclasses with validation. `DetectionFrame` contains `timepoint`,
`coords`, `labels`, a property-to-array `features` mapping, and `track_indices`.
`DetectionSeries` contains a name and tuple of frames. `TrackingSequence` contains
the root, dimensionality, detection series, and boolean lineage-relation matrix.

All arrays in a frame must share the same leading detection dimension.
`track_indices == -1` denotes an unmatched detection. `lineage_relation[i, j]` is
true when tracklets are identical or one is an ancestor of the other; siblings remain
unrelated, preserving the current target semantics.

`DetectionFrame.features` stores the canonical raw WR property superset, not a final
feature vector. `wrfeat`, `wrfeat2`, and `wrfeat2_no_intensity` are derived at runtime
after geometric augmentation. Images, masks, embeddings, or other aligned data may be
added later as optional fields without changing the windowing abstraction.

## CTC ingestion

Implement `TrackingSequence.from_ctc` directly in `data.py`. It will:

1. Resolve the sequence root, images, GT masks, track file, and detection folders.
2. Load the selected frames with spatial and temporal downsampling.
3. Apply the existing TRA/ST correction rules.
4. Parse and validate `man_track.txt`, or build isolated tracklets for `use_gt=False`.
5. Extract canonical WR properties once per detection and frame.
6. Match non-GT detections to GT tracklets using the existing matching behavior.
7. Assign compact integer track indices, reserving `-1` for unmatched detections.
8. Build the compact boolean lineage relation.
9. Return an immutable `TrackingSequence` without constructing overlapping windows.

Support both development layouts: the standard CTC fixture with images in
`Fluo-C2DL-Huh7/01` and GT in `01_GT/TRA`, and the simple fixture with images in
`151031-03/img` and masks plus `man_track.txt` in `151031-03/TRA`.

Retain automatic resolution for standard CTC and `img/TRA` layouts. Allow explicit
image, GT, detection-folder, and track-file overrides so `img/masks` or future custom
folder names do not require changes to `TrackingData`.

## Runtime dataset

`TrackingData` receives a sequence plus `window_size`, feature mode, augmentation,
detection budget, and detection-dropout settings.

It stores only a window index of `(detection_series, start_frame)` pairs. For each
item it concatenates the selected frames, constructs the association target from
`track_indices` and `lineage_relation`, then applies operations in this order:

1. Lineage-preserving `max_detections` selection.
2. Lineage-level detection dropout, retaining at least one lineage.
3. WR geometric and feature augmentation.
4. Runtime projection to the selected WR feature representation.
5. Relative-time coordinate construction and tensor conversion.

Spatial cropping was part of the original design but has been removed (see the
2026-06-25 decision in the observations log); `max_detections` plus coordinate
augmentation cover its role.

Association construction becomes a direct relation lookup rather than repeated
NetworkX ancestor/descendant traversal. `n_objects` and `n_divs` should be computed
from compact frame and lineage metadata without materializing dense matrices solely
for sampler metadata.

## Feature and API cleanup

Only these public feature modes remain:

- `wrfeat`
- `wrfeat2`
- `wrfeat2_no_intensity`

Remove the legacy `none`, `regionprops`, `regionprops2`, `patch`, and
`patch_regionprops` paths. Region-property computation inside WRFeatures remains an
implementation detail. Also remove `max_tokens`, `**kwargs`, `from_arrays`, custom
array compression, legacy window builders, duplicated getters, and obsolete skipped
tests. Simplify training feature-dimension and feature-embedding selection accordingly.

Remove the nonstandard `dataset.__getitem__(index, return_dense=True)` contract.
Dense training visualizations should either use a separate CTC visualization loader
or be removed initially. They must not force images and masks into every cached frame.

## Caching

Keep the existing `--cache` and `--cachedir` training options. When caching is enabled,
wrap only `TrackingSequence.from_ctc`:

```python
memory = joblib.Memory(cachedir, verbose=1)
loader = memory.cache(TrackingSequence.from_ctc, ignore=["n_workers"])
sequence = loader(**sequence_kwargs)
```

Assume files at a path do not change between runs. Do not add manifests, content
fingerprints, migration logic, or a custom cache implementation. Users clear the
configured cache directory manually after source-data or preprocessing changes.

Arguments included in the cached loader call are `root`, resolved folder overrides,
`detection_folders`, `ndim`, `use_gt`, `slice_pct`, spatial/temporal downsampling, and
matching parameters. Worker count is ignored because it only affects performance.

The following remain outside the cached call: `window_size`, feature mode,
augmentation, detection budget, detection dropout, batch settings, and
sampler settings. Changing any of them must reuse the cached `TrackingSequence`.

`BalancedDataModule` will receive separate `sequence_kwargs` and
`tracking_data_kwargs`. `prepare_data()` populates sequence caches; `setup()` loads the
sequences and constructs distinct train and validation `TrackingData` instances.

## Parity harness

Create the temporary script `scripts/compare_tracking_data.py`. It compares
`LegacyCTCData` with the new implementation on:

- `scripts/data/ctc_2026/2d/Fluo-C2DL-Huh7/01`
- `scripts/data/vanvliet2/recA/151031-03`

Start with augmentation, cropping, detection dropout, and detection limits disabled.
For every window compare dataset length, frame range, coordinates, labels, timepoints,
raw properties, all three feature views, association targets, `n_objects`, `n_divs`,
tensor keys, shapes, and dtypes. Use exact equality for integer/boolean data and
`assert_allclose` for floating-point data.

Then compare cropping, detection limits, dropout, and each augmentation level
independently by resetting NumPy and Torch seeds before calling each implementation.
On failure, report the fixture, window, field, shapes, and maximum numerical error.

The real-data script is a local migration tool, not a permanent CI test. Convert its
essential cases into small synthetic pytest fixtures before deleting it.

## Commit-gated implementation phases

Use this document as the implementation ledger. Mark completed work with `[x]`, leave
pending work as `[ ]`, and add important discoveries to the observations log below.
Update it during implementation rather than reconstructing progress at phase end.

### Phase 1: data model, CTC loading, and deterministic parity

- [x] Keep current `data.py` unchanged and create the staged `datanew.py` implementation.
- [x] Add the frozen data models and validation tests to `datanew.py`.
- [x] Implement `TrackingSequence.from_ctc` for both reference layouts.
- [x] Extract canonical WR properties and aligned compact track indices.
- [x] Build compact lineage relations and prove association-target parity.
- [x] Implement deterministic `TrackingData` windowing.
- [x] Add and run the real-data parity script for windows, features, targets, and metadata.
- [x] Run Phase 1 tests and record results and deviations below.

Pause after Phase 1. Report tests, parity results, remaining differences, and the exact diff
scope. Do not start Phase 2 until the user has reviewed and committed Phase 1.

### Phase 2: runtime behavior, caching, and training integration

- [x] Spatial cropping removed entirely (see decision below); no parity check needed.
- [x] Add lineage-aware detection limits and dropout with parity checks.
- [x] Add WR augmentation levels with seeded parity checks.
- [x] Add configurable joblib caching around `TrackingSequence.from_ctc`.
- [x] Split `BalancedDataModule` into sequence and runtime arguments.
- [x] Switch training, collation, sampler metadata, and maintained configurations.
- [x] Run both fixtures and CPU smoke tests for every supported feature mode.
- [x] Verify cache reuse across all runtime-only configuration changes.

Pause after Phase 2. Report tests, cache-hit checks, smoke-test results, and the exact
diff scope. Do not start Phase 3 until the user has reviewed and committed Phase 2.

### Phase 3: canonical CTC-folder I/O interface

Goal: a single canonical way to load images, masks, and tracking info from CTC-like
folders, exposed from the new data module, so inference no longer needs the legacy
`CTCData.load_for_inference`. This phase is a prerequisite for the Phase 4 migration
because `predict.py`/`check_errors.py` must move off legacy before `CTCData` is deleted.

Agreed design (decided 2026-06-25):

- Carry optional raw arrays on the immutable data model, defaulting to `None`, so they
  are absent for training and never forced into every cached frame: raw normalized
  `images` on `TrackingSequence` (the shared movie) and label `masks` on
  `DetectionSeries` (per detection folder).
- Add a `load_images: bool = False` flag to `TrackingSequence.from_ctc`. When set, the
  loader also retains `images`/`masks` using the existing loading helpers
  (`_resolve_paths`, `_resolve_detection_folder`, `_load_tiffs`, `_correct_with_st`,
  `_ensure_ndim`, `normalize`). Update the frozen-dataclass validation and `__reduce__`
  for the new optional fields.
- The flag is part of the joblib cache key, so cached inference sequences (with images)
  stay separate from lean training sequences (without). This makes repeated wandb
  predict-logging on the same validation data cache-fast.
- Provide a thin canonical accessor for scripts that only want arrays
  (e.g. `images, masks, image_path, gt_path`) on top of the same path-discovery logic;
  do not drag any legacy `CTCData` helper methods along.
- Tradeoff recorded: `from_ctc` still does the expensive matching/feature extraction
  even when only images/masks are needed; acceptable because the cache amortizes it for
  the repeated-logging case, and a future `model.track` that accepts pre-extracted
  features would remove the waste.

- [x] Add optional `images`/`masks` fields and `load_images` flag with validation and
      pickle support.
- [x] Provide the canonical array accessor and repoint `scripts/predict.py` and
      `scripts/check_errors.py` onto it.
- [x] Convert `test_load_for_inference_refines_tra_with_st` into a synthetic test of the
      new interface.

### Phase 4: legacy removal and final hardening

Migration discoveries (2026-06-25) that refine the original plan:

- `data.py` is not purely legacy: `collate_sequence_padding`, `densify_assoc`, and
  `warn_association_distances` are standalone functions still needed by production
  (`model/predict.py` uses `collate_sequence_padding` on the installed inference path;
  `train.py` uses `densify_assoc`; `distributed.py` uses `warn_association_distances`).
  These must move into the new module before the rename so no production code imports
  `_legacy_data.py`.
- `extract_features_regionprops` is defined in `features.py` and only re-exported by
  `data.py`; the package `__init__` should import it from `features` after the rename.
- `_FileCache` in `distributed.py` and its `isinstance(dataset, CTCData | TrackingData)`
  branch are legacy CTCData plumbing to remove (only `TrackingData` remains).
- `tests/test_data.py` targets legacy `CTCData` (most tests already skipped); its still
  relevant cases (`load_for_inference`, dropout/neighborhood helpers now living in the
  new module) move to `test_datanew.py`, the rest are deleted.

Steps:

- [x] Move the three shared functions (+`pad_tensor`) into the new module.
- [x] Rename `data.py` -> `_legacy_data.py` and `datanew.py` -> `data.py`; rewrite the
      package `__init__` exports (drop `CTCData`/`_ctc_lineages`, add `TrackingSequence`,
      `TrackingData`, the canonical loader, and the shared functions).
- [x] Repoint `distributed.py` imports and remove `cache_class` and the CTCData isinstance.
- [x] Repoint test imports (`trackastra.data.datanew` -> `trackastra.data`).
- [x] Remove obsolete skipped tests; migrate the essential ones (Phase 3 already moved
      the inference test).
- [x] Convert essential real-data parity checks into synthetic tests.
- [x] Run the full relevant test suite and final real-data smoke tests.
- [ ] Delete `_legacy_data.py` and the temporary comparison script (deferred: kept one
      review cycle as the parity oracle; delete after a real training run confirms the
      new pipeline).
- [x] Confirm no production import or configuration references legacy behavior
      (`_legacy_data` is imported only by `scripts/compare_tracking_data.py`).

Pause after Phase 4 with the final verification report; never create commits automatically.

### Phase 5: split data.py into a torch-free sequence/IO module and a torch dataset module

After Phase 4, `data.py` (~970 lines) mixes two concerns: the format-neutral data model
plus CTC loading (numpy/pandas/tifffile/scipy/networkx, no torch) and the PyTorch dataset
plus collation (torch). Split them into two modules with a strict one-way dependency
(`dataset` imports `sequence`, never the reverse). A useful side effect: `sequence` becomes
importable without torch, so pure data inspection / CTC loading does not pull in torch.

Chosen names: `io.py` (data model + CTC I/O) and `dataset.py` (torch dataset +
collation). Note: a package-local `trackastra/data/io.py` does not shadow the stdlib
`io` for other modules, since absolute imports resolve `import io` to the stdlib; only
`from .io import ...` / `from . import io` reach the local module.

`io.py` (torch-free):

- `Segmentation` (flat), `TrackingSequence`, `_immutable_array`. (`DetectionFrame` and
  `DetectionSeries` are merged into `Segmentation` - see the data-model decision below.)
- `TrackingSequence.from_ctc`, `_load_ctc_sequence`, and the loading helpers
  (`_resolve_paths`, `_resolve_detection_folder`, `_load_tiffs`, `_ensure_ndim`,
  `_correct_with_st`, `_filter_tracks`, `_isolated_tracks`, `_lineage_arrays`).
- `load_ctc_for_inference`, `_resolve_inference_paths`.

IO API direction (no new class): `TrackingSequence.from_ctc(root, ...)` IS the canonical
loader. It is already addressed purely by root (as in the configs) and auto-resolves both
reference layouts - standard CTC (`<ds>/01` images, `<ds>/01_GT/TRA`, `<ds>/01_ST/SEG`)
and the simple `<root>/img` + `<root>/TRA` - and the `load_images` flag covers both the
no-image dataset case and the images-retained prediction case. `load_ctc_for_inference`
already delegates to `from_ctc(use_gt=False, load_images=True)` and only adds the
array-returning convenience for prediction, so it stays a thin wrapper. The `_resolve_*`
helpers remain internal support for `from_ctc`; no separate loader class is introduced.

Optional DRY cleanup: `load_ctc_for_inference` currently re-resolves `(image_path,
gt_tra)` via `_resolve_inference_paths`, duplicating logic inside `from_ctc`. If desired,
store the resolved `image_path`/`gt_path` on the `TrackingSequence` (alongside `root`) so
the wrapper reads them off the returned object instead of re-resolving. Low priority.

`dataset.py` (torch; imports from `io.py`):

- `TrackingDataset` and its runtime helpers (`_association_target`,
  `_sample_neighborhood_indices`, `_sample_detection_keep_indices`, `_subset_features`,
  `_wr_augmenter`, `_division_count`). `_concat_frames` is dropped - a window becomes a
  `timepoints`-range slice of the flat `Segmentation` arrays (`searchsorted`).
- `pad_tensor`, `densify_assoc`, `collate_sequence_padding`.
- `association_distances`, `warn_association_distances`.

Data-model decision (confirmed 2026-06-25): KEEP a two-level model (sequence -> flat
segmentation), one image folder + multiple segmentation folders + one common GT. The
common training case has several segmentation folders for one movie
(`scripts/configs/general2d.yaml` uses `detection_folders: [TRA, RES]`), so multiple
segmentations per sequence is real and intended. `images` and `lineage` stay shared on
`TrackingSequence`; `masks` and the detections stay per `Segmentation`. Do NOT collapse the
sequence/segmentation levels into one, and `images`/`masks` are dense `(T, H, W)` rasters
stored at the level where they are invariant (image on the sequence, masks per
segmentation) - never per detection. Per-series GT / differing images per mask are out of
scope (use separate `TrackingSequence`s if ever needed). (`general2d.yaml` itself is an
older config: it uses the removed `patch_regionprops` mode and absent data dirs, so it is
not maintained for the new pipeline; it only documents the multi-segmentation intent.)

Final class set (confirmed 2026-06-25):

- `TrackingSequence` (keep): `root, ndim, segmentations, lineage_relation,
  lineage_parents, images`.
- `Segmentation` (new, replaces `DetectionSeries` + `DetectionFrame`): a single
  segmentation of the movie stored as flat columnar arrays - `coords, labels, timepoints
  (sorted), features (dict), track_indices`, plus optional `masks (T, H, W)`. The former
  per-frame `DetectionFrame` grouping becomes transient locals during `from_ctc`
  extraction, then concatenated into these flat arrays. Name avoids "Series" (no clash
  with `Sequence`).
- `DetectionFrame` -> removed (merged into the flat `Segmentation`).
- `TrackingData` -> `TrackingDataset` (idiomatic `torch.utils.data.Dataset` subclass).

- [ ] Merge `DetectionFrame` + `DetectionSeries` into the flat `Segmentation` class
      (columnar arrays + sorted `timepoints` + optional `masks`); update `from_ctc`
      assembly and `TrackingDataset` windowing (timepoint-range slice, drop
      `_concat_frames`); update the tests that construct these directly.
- [ ] Apply the remaining renames (`TrackingData` -> `TrackingDataset`) across the
      package, scripts, and tests.
- [ ] Keep `from_ctc` as the single canonical loader (no new class); confirm both
      reference layouts resolve from root alone for dataset (`load_images=False`) and
      prediction (`load_images=True`). Optionally DRY the inference path resolution.
- [ ] Create `io.py` and `dataset.py`; move the symbols above (keep one-way
      dependency, no circular imports).
- [ ] Remove `data.py`; update package `__init__` to import from `io` and `dataset`.
- [ ] Repoint `distributed.py` and tests (`trackastra.data.data` -> the new modules or the
      package re-exports).
- [ ] Update the `scripts/compare_tracking_data.py` import (still references the package).
- [ ] Confirm `io.py` imports with torch uninstalled/unimported (no torch in its
      import graph).
- [ ] Run the full relevant test suite, ruff, and the parity script; update the repository
      map in `CLAUDE.md` (the `data.py` responsibilities line).

Pause after Phase 5 with the verification report; never create commits automatically.

## Implementation observations

Append short dated entries containing only information that affects later work: discovered
invariants, intentional parity differences, cache behavior, fixture limitations, benchmark
results, blockers, and decisions that revise this plan.

- 2026-06-25: Phase 1 uses `datanew.py` while `data.py` remains the unchanged oracle;
  the rename is deferred to the final migration gate to keep production imports working.
- 2026-06-25: `tests/test_datanew.py` passes 5 tests covering frozen validation,
  direct lineage targets, unmatched detections, both reference layouts, all three feature
  views, and detection-only `use_gt=False` loading.
- 2026-06-25: `scripts/compare_tracking_data.py` reports exact deterministic parity for
  21 Huh7 windows and 14 Van Vliet windows. It compares frame ranges, coordinates,
  labels, timepoints, canonical properties, three feature views, association targets,
  metadata, tensor keys, shapes, and dtypes.
- 2026-06-25: Phase 1 intentionally excludes cropping, detection limits, dropout, and
  augmentation; their seeded parity checks remain Phase 2 work.
- 2026-06-25: Phase 2 runtime transforms are implemented in `TrackingData`. A manual
  seeded comparison on Huh7 window 0 matched the legacy implementation exactly for
  cropping, detection limits, dropout, and augmentation levels 1 through 4. The parity
  script now covers every window for these configurations but has not been rerun since
  that extension, so the first three Phase 2 checklist entries remain open.
- 2026-06-25: `mappingproxy` was not pickleable. `DetectionFrame.features` now uses a
  plain `dict` (arrays inside are already read-only via `_immutable_array` and the frozen
  dataclass guards the field), and explicit `__reduce__` reconstruction re-applies the
  read-only flags on cached arrays. The joblib test confirms that changing `n_workers`
  reuses the cache entry.
- 2026-06-25: `BalancedDataModule` now caches only `TrackingSequence.from_ctc` and
  constructs split-specific `TrackingData` instances. `scripts/train.py` passes slicing
  and downsampling as sequence arguments, while windowing, dropout, augmentation,
  and detection limits are runtime arguments.
- 2026-06-25: Decision - spatial cropping (`WRRandomCrop`, `crop_size`,
  `crop_ensure_all_centers`) is removed from the new pipeline. `max_detections` already
  bounds per-window token count via lineage-complete selection, and the geometric
  coordinate augmentation still provides spatial/translation variety, so the cropper was
  redundant. Removed the cropper plumbing and the crop-only `loss_mask` censoring path
  from `TrackingData` (`_association_subset_loss_mask` deleted; in the new dataset the
  cropper was the only producer of `loss_mask`, since `max_detections` and dropout keep
  complete lineages and never split a positive association). `train.py`'s common_step
  still honours an incoming `loss_mask` for the legacy oracle and its unit test. Dropped
  `--crop_size`/`--crop_ensure_all_centers` train flags, the dead `--compress` and
  `--from_subfolder` flags, and the corresponding config entries (crop, compress, and the
  duplicate `features` key in `vanvliet2.yaml`, which now resolves to `wrfeat2`).
- 2026-06-25: Restored a `--device {auto,cpu,cuda}` override in `scripts/train.py`
  (accelerator was hardcoded to cuda-when-available) and fixed
  `association_distances(dataset.windows)` -> `association_distances(dataset)` in
  `distributed.py`. End-to-end CPU training now runs on local data
  (`scripts/data/vanvliet2/recA/151031-03`) for both the bare CLI and the `vanvliet2`
  config; cache reuse verified by changing `--augment` between two cached runs.
- 2026-06-25: Phase 2 completed and committed. All Phase 2 checklist items pass (parity
  script, per-feature-mode CPU smoke test, cache reuse, training/collation/sampler
  switch). Cropping was removed rather than parity-checked.
- 2026-06-25: Inference-loading design decided and deferred to its own phase (now
  Phase 3) rather than improvised during the migration. The original Phase 3 (legacy
  removal) became Phase 4. Reason: `CTCData.load_for_inference` is the only inference
  entry point and must be replaced by a canonical CTC-folder loader before `CTCData`
  can be deleted, and the chosen approach (optional `images`/`masks` on the data model
  via a `from_ctc(load_images=...)` flag) touches the immutable model enough to warrant
  a planned phase. No code for this was landed yet.

- 2026-06-25: Phase 3 implemented in `datanew.py`. `DetectionSeries` gained an optional
  `masks` field and `TrackingSequence` an optional `images` field (both default `None`,
  read-only, pickled via `__reduce__`); `from_ctc(load_images=True)` attaches the already
  loaded per-folder masks and normalized images. `load_ctc_for_inference` is the canonical
  array accessor (exported from `trackastra.data`), reusing the loading helpers; it accepts
  an optional `loader` so callers can pass a joblib-cached `from_ctc`. `scripts/predict.py`
  and `scripts/check_errors.py` now use it instead of `CTCData.load_for_inference`.
  Synthetic test added (`test_load_ctc_for_inference_refines_tra_with_st`,
  `test_load_images_flag_attaches_arrays_and_survives_pickle`). Parity unchanged (21 + 14),
  32 focused tests pass, real-data smoke confirmed on Van Vliet.

- 2026-06-25: Phase 4 migration done (not committed). Shared functions
  (`collate_sequence_padding`, `densify_assoc`, `warn_association_distances`, `pad_tensor`)
  moved into the new module; `datanew.py` renamed to `data.py` and old `data.py` to
  `_legacy_data.py` via `git mv`. Package `__init__` now exports the new API
  (`TrackingData`, `TrackingSequence`, `load_ctc_for_inference`, the shared funcs) and
  `extract_features_regionprops` from `features`; `CTCData`/`_ctc_lineages` dropped.
  `distributed.py` lost the dead `cache_class` and the `CTCData` isinstance branch.
  `test_data.py` rewritten to keep only the still-valid helper/collate/dropout/warn tests
  (now untagged from `train`), with the dedup test rewritten to the new
  `association_distances(dataset, ...)` API. `test_cli.py` repointed off
  `CTCData.load_for_inference` (its `example_dataset` helper moved in-file) and its stale
  `evaluate_ctc` mock fixed; this CLI predict test had been silently broken since Phase 3.
  Verified: 105 passed / 3 skipped, parity still exact (21 + 14), end-to-end CPU training
  and the inference accessor work through the renamed module. The 8 `test_pretrained`
  integration failures are pre-existing (identical on the prior commit; expected-value
  drift, unrelated to this refactor). `_legacy_data.py` and the comparison script are kept
  one cycle as the parity oracle.
- 2026-06-25: Phase 5 design settled (planning only, no code yet). Decisions:
  (a) NO new loader class - `TrackingSequence.from_ctc(root, ...)` is the canonical loader;
  it already resolves both reference layouts from root alone and the `load_images` flag
  covers dataset (no images) vs prediction (images retained); `load_ctc_for_inference`
  stays a thin wrapper. (b) Module split names: `io.py` (torch-free data model + CTC I/O)
  and `dataset.py` (torch dataset + collation), `data.py` removed. (c) Dense-array
  placement is by sharing scope and is correct as-is: `images` on `TrackingSequence` (one
  movie shared by all series) and `masks` on `DetectionSeries` (one per detection folder);
  `DetectionFrame` stays strictly per-detection (no dense rasters). (d) A `TrackingSequence`
  may hold multiple `DetectionSeries`, one per `detection_folders` entry, all sharing the
  same `images` and lineage; default `("TRA",)` gives one series. (e) Verified on disk:
  `data/vanvliet/` and `data/vanvliet2/` sequences are all the simple `img/` + `TRA/`
  layout (no `mask` folder); `Fluo-C2DL-Huh7` is the standard CTC layout. Both load via
  root alone.

## Current handoff state

Phases 2-4 are committed (`7cdcae3`, `5b8a6fb`, `c1675c7`). The new `data.py`
(`TrackingSequence` + `TrackingData` + `load_ctc_for_inference`) is the production module;
legacy `CTCData` lives in `_legacy_data.py` as a parity oracle (deletion deferred until a
real training run confirms the new pipeline).

Next up: execute Phase 5 (split `data.py` into `io.py` + `dataset.py`; keep `from_ctc`
canonical, no new class) per the spec above. No Phase 5 code has been written yet. Resume
recipe after the split: `ruff check`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
-p no:cacheprovider tests/test_data.py tests/test_datanew.py tests/test_train.py
--deselect tests/test_train.py::test_train_dry_run`, then `python
scripts/compare_tracking_data.py` (expect 21 + 14 windows match). Known pre-existing
failures unrelated to this work: 8 `tests/test_pretrained.py::test_integration` (expected-
value drift). Untracked working-tree edits to repoint off legacy: `scripts/check_errors.py`
and `scripts/compare_tracking_data.py` (both intentionally left untracked).

Earlier handoff notes below predate Phase 2 completion and are kept only for historical
context.

Verified before the final test edit:

- `14 passed, 14 deselected` for the focused `test_datanew.py` and `test_train.py`
  selection, including runtime selection, augmentation immutability, pickle/joblib cache,
  split data-module arguments, and feature dimensions.
- Ruff and Python compilation passed for all then-modified Python files.
- Manual seeded Huh7 parity was exact for crop, limit, dropout, and augment levels 1-4.

Added but not yet verified:

- A CPU training smoke test in `tests/test_train.py` for all three feature modes.
- Full runtime checks in `scripts/compare_tracking_data.py` for both real fixtures.
- The latest edits need formatting/static checks after the CPU smoke test addition.

Resume with:

```bash
ruff format trackastra/data/datanew.py trackastra/data/distributed.py \
  tests/test_datanew.py tests/test_train.py scripts/compare_tracking_data.py scripts/train.py
ruff check trackastra/data/datanew.py trackastra/data/distributed.py \
  tests/test_datanew.py tests/test_train.py scripts/compare_tracking_data.py scripts/train.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  tests/test_datanew.py tests/test_train.py
python scripts/compare_tracking_data.py
```

Then verify cache reuse across changes to every runtime-only option, update maintained
configuration files if needed, run an actual data-module batch, and complete the remaining
Phase 2 checklist items. Current implementation scope is `trackastra/data/datanew.py`,
`trackastra/data/distributed.py`, `scripts/train.py`, `tests/test_datanew.py`,
`tests/test_train.py`, and `scripts/compare_tracking_data.py`. The Phase 1 ledger, tests,
and parity script are still untracked even though `datanew.py` itself was committed.

## Definition of done

- Both reference layouts produce equivalent deterministic windows and supervision.
- All maintained feature modes train through one common `TrackingData.__getitem__`.
- Runtime configuration changes reuse the same joblib sequence cache.
- Train and validation can share a cached sequence while using different transforms.
- Cached objects contain no augmentation, cropper, sampler, or window configuration.
- No production imports reference `_legacy_data.py`.
- Tests cover layout resolution, matching, lineage targets, runtime transforms, cache
  reuse, multi-worker loading, and one CPU training step per feature mode.
