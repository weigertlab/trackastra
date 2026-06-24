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
crop settings, detection budget, and detection-dropout settings.

It stores only a window index of `(detection_series, start_frame)` pairs. For each
item it concatenates the selected frames, constructs the association target from
`track_indices` and `lineage_relation`, then applies operations in this order:

1. Spatial crop and its censoring mask, if configured.
2. Lineage-preserving `max_detections` selection.
3. Lineage-level detection dropout, retaining at least one lineage.
4. WR geometric and feature augmentation.
5. Runtime projection to the selected WR feature representation.
6. Relative-time coordinate construction and tensor conversion.

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
augmentation, crop settings, detection budget, detection dropout, batch settings, and
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

- [ ] Add cropping with seeded legacy parity checks.
- [ ] Add lineage-aware detection limits and dropout with parity checks.
- [ ] Add WR augmentation levels with seeded parity checks.
- [x] Add configurable joblib caching around `TrackingSequence.from_ctc`.
- [x] Split `BalancedDataModule` into sequence and runtime arguments.
- [ ] Switch training, collation, sampler metadata, and maintained configurations.
- [ ] Run both fixtures and CPU smoke tests for every supported feature mode.
- [ ] Verify cache reuse across all runtime-only configuration changes.

Pause after Phase 2. Report tests, cache-hit checks, smoke-test results, and the exact
diff scope. Do not start Phase 3 until the user has reviewed and committed Phase 2.

### Phase 3: legacy removal and final hardening

- [ ] Remove legacy feature modes, compression, dense-item plumbing, and obsolete arguments.
- [ ] Remove unused helpers and replace obsolete skipped tests.
- [ ] Convert essential real-data parity checks into synthetic tests.
- [ ] Run the full relevant test suite and final real-data smoke tests.
- [ ] Delete `_legacy_data.py` and the temporary comparison script.
- [ ] Confirm no production import or configuration references legacy behavior.

Pause after Phase 3 with the final verification report; never create commits automatically.

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
  and downsampling as sequence arguments, while windowing, crop, dropout, augmentation,
  and detection limits are runtime arguments.

## Current handoff state

Work stopped during Phase 2 on 2026-06-25. Do not start Phase 3.

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
