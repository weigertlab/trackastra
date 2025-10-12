#!/usr/bin/env python3
"""Test script to verify [train] extras installation and imports."""

import sys


def main():
    """Test training imports as specified in the task."""
    print("\n" + "=" * 60)
    print("Testing [train] extras imports...")
    print("=" * 60 + "\n")

    # Test the exact imports from the task
    try:
        from trackastra.data.augmentations import AugmentationPipeline
        from trackastra.data.distributed import BalancedDataModule

        print(f"✓ BalancedDataModule: {BalancedDataModule.__name__}")
        print(f"✓ AugmentationPipeline: {AugmentationPipeline.__name__}")
    except ImportError as e:
        print(f"✗ Failed to import trackastra training modules: {e}")
        return 1

    try:
        import lightning
        import tensorboard  # noqa: F401
        import wandb

        print(f"✓ lightning v{lightning.__version__}")
        print(f"✓ wandb v{wandb.__version__}")
        print("✓ tensorboard")
    except ImportError as e:
        print(f"✗ Failed to import training dependencies: {e}")
        return 1

    print("\nTraining imports successful!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
