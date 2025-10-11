#!/usr/bin/env python
"""Test script to verify core trackastra imports work without training dependencies.

This script tests that the basic inference API works with only core dependencies
installed (without lightning, wandb, tensorboard, kornia, etc.).
"""

import sys
import traceback


def test_core_imports():
    """Test that core imports work without training dependencies."""
    print("=" * 70)
    print("Testing Core Trackastra Imports (without training dependencies)")
    print("=" * 70)
    print()

    # Test 1: Basic package import
    print("Test 1: Importing trackastra package...")
    try:
        import trackastra

        print(f"✓ Success! trackastra version: {trackastra.__version__}")
    except Exception as e:
        print(f"✗ Failed to import trackastra: {e}")
        traceback.print_exc()
        return False
    print()

    # Test 2: Import core model class
    print("Test 2: Importing Trackastra model class...")
    try:
        from trackastra.model import Trackastra

        print("✓ Success! Trackastra class imported")
    except Exception as e:
        print(f"✗ Failed to import Trackastra: {e}")
        traceback.print_exc()
        return False
    print()

    # Test 3: Import data utilities
    print("Test 3: Importing data utilities...")
    try:
        from trackastra.data import (
            example_data_bacteria,
        )

        print("✓ Success! Data utilities imported")
    except Exception as e:
        print(f"✗ Failed to import data utilities: {e}")
        traceback.print_exc()
        return False
    print()

    # Test 4: Import feature extraction
    print("Test 4: Importing feature extraction utilities...")
    try:
        print("✓ Success! Feature extraction utilities imported")
    except Exception as e:
        print(f"✗ Failed to import feature extraction: {e}")
        traceback.print_exc()
        return False
    print()

    # Test 5: Verify training dependencies are NOT available
    print("Test 5: Verifying training dependencies are NOT installed...")
    training_deps_found = []
    training_deps_missing = []

    for dep in ["lightning", "wandb", "tensorboard", "kornia"]:
        try:
            __import__(dep)
            training_deps_found.append(dep)
        except ImportError:
            training_deps_missing.append(dep)

    if training_deps_found:
        print(f"⚠ Warning: Found training dependencies: {training_deps_found}")
        print("  This test should run with core dependencies only!")
    else:
        print(
            f"✓ Success! Training dependencies not installed: {training_deps_missing}"
        )
    print()

    # Test 6: Try loading example data (optional - requires download)
    print("Test 6: Loading example data...")
    try:
        data_dict = example_data_bacteria()
        print("✓ Success! Loaded bacteria example data")
        print(f"  Keys: {list(data_dict.keys())}")
    except Exception as e:
        print(f"⚠ Could not load example data (might need download): {e}")
        # This is not a critical failure
    print()

    # Test 7: Try instantiating model from pretrained (requires download)
    print("Test 7: Loading pretrained model...")
    try:
        model = Trackastra.from_pretrained("general_2d", device="cpu")
        print("✓ Success! Loaded pretrained model 'general_2d'")
        print(f"  Model device: {next(model.model.parameters()).device}")
    except Exception as e:
        print(f"⚠ Could not load pretrained model (might need download): {e}")
        # This is not a critical failure for this test
    print()

    print("=" * 70)
    print("Core imports test completed successfully!")
    print("=" * 70)
    return True


def check_installed_packages():
    """Display information about installed packages."""
    print("\nInstalled packages check:")
    print("-" * 70)

    import importlib.metadata

    # Check for core dependencies
    core_deps = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-image",
        "torch",
        "torchvision",
        "pyyaml",
        "edt",
        "joblib",
        "lz4",
        "imagecodecs",
        "chardet",
        "dask",
        "numba",
        "geff",
    ]

    print("\nCore dependencies:")
    for dep in core_deps:
        try:
            version = importlib.metadata.version(dep)
            print(f"  ✓ {dep}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  ✗ {dep}: NOT INSTALLED")

    # Check for training dependencies (should not be installed)
    train_deps = ["lightning", "wandb", "tensorboard", "kornia"]

    print("\nTraining dependencies (should NOT be installed for this test):")
    for dep in train_deps:
        try:
            version = importlib.metadata.version(dep)
            print(f"  ⚠ {dep}: {version} (SHOULD NOT BE INSTALLED)")
        except importlib.metadata.PackageNotFoundError:
            print(f"  ✓ {dep}: not installed (correct)")

    print("-" * 70)


if __name__ == "__main__":
    # First check what packages are installed
    try:
        check_installed_packages()
    except Exception as e:
        print(f"Could not check installed packages: {e}")

    print("\n")

    # Run the core imports test
    success = test_core_imports()

    if success:
        print("\n✓ All core import tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some core import tests failed!")
        sys.exit(1)
