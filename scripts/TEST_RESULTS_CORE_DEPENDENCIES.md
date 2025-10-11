# Test Results: Core Dependencies Only

## Summary

Successfully verified that trackastra core inference API works with only core dependencies installed (without training dependencies like lightning, wandb, tensorboard, kornia).

## Changes Made

### 1. Updated `setup.cfg`
Added missing core dependencies that are required for inference:
- `tqdm` - Used for progress bars in model loading, data processing, and tracking
- `requests` - Used for downloading pretrained models from GitHub releases

### 2. Modified `trackastra/data/data.py`
Implemented lazy loading for training-only augmentation dependencies:
- Removed top-level imports of `AugmentationPipeline`, `RandomCrop`, and `default_augmenter`
- Added lazy imports inside the `_setup_features_augs()` method where they are actually used
- This ensures that `kornia` (a training dependency) is only required when instantiating `CTCData` for training, not for inference

## Test Results

Tested in a clean virtual environment with only core dependencies:

### Installed Core Dependencies
- numpy: 2.3.3
- scipy: 1.16.2
- pandas: 2.3.3
- scikit-image: 0.25.2
- torch: 2.8.0
- torchvision: 0.23.0
- pyyaml: 6.0.3
- edt: 3.0.0
- joblib: 1.5.2
- lz4: 4.4.4
- imagecodecs: 2025.8.2
- chardet: 5.2.0
- dask: 2025.9.1
- numba: 0.62.1
- geff: 1.1.1.1.1
- **tqdm: 4.67.1** (newly added)
- **requests: 2.32.5** (newly added)

### Training Dependencies Status
Verified that these are NOT installed (correct for core-only installation):
- lightning: not installed ✓
- wandb: not installed ✓
- tensorboard: not installed ✓
- kornia: not installed ✓

### Core Functionality Tests

All core import tests passed:

1. ✓ Basic package import (`import trackastra`)
2. ✓ Model class import (`from trackastra.model import Trackastra`)
3. ✓ Data utilities import (`from trackastra.data import example_data_*`)
4. ✓ Feature extraction utilities import (`from trackastra.data import extract_features_regionprops, WRFeatures`)
5. ✓ Training dependencies verification (confirmed not installed)
6. ✓ Example data loading (`example_data_bacteria()`)
7. ✓ Pretrained model loading (`Trackastra.from_pretrained("general_2d", device="cpu")`)

## Installation Commands

### Core dependencies only (for inference)
```bash
pip install -e .
```

### With training dependencies
```bash
pip install -e .[train]
```

### With all dependencies
```bash
pip install -e .[all]
```

## Files Modified

1. `/Users/bgallusser/code/trackastra/setup.cfg`
   - Added `tqdm` to core dependencies
   - Added `requests` to core dependencies

2. `/Users/bgallusser/code/trackastra/trackastra/data/data.py`
   - Converted augmentation imports to lazy loading
   - Added lazy import inside `_setup_features_augs()` method

3. `/Users/bgallusser/code/trackastra/scripts/test_core_imports.py`
   - New test script to verify core imports work without training dependencies

## Conclusion

The trackastra inference API now works correctly with only core dependencies installed. Training-specific functionality (like `CTCData` dataset augmentation) will only require training dependencies when actually instantiated, not when importing the package.
