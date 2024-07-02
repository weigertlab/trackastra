import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pytest
import torch
from trackastra.data import example_data_hela
from trackastra.model import Trackastra


@pytest.mark.parametrize("name", ["ctc", "general_2d"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pretrained(name, device):
    """Each pretrained model should run on all (available) device."""
    if device == "cuda":
        if torch.cuda.is_available():
            run_predictions(name, "cuda")
        else:
            pytest.skip("cuda not available")
    elif device == "mps":
        # TODO mps on GitHub actions only has 8GB memory, so it's disabled for now.
        if torch.backends.mps.is_available():
            run_predictions(name, "mps")
        else:
            pytest.skip("mps not available")
    elif device == "cpu":
        run_predictions(name, "cpu")
    else:
        raise ValueError()

    assert True


def run_predictions(name, device):
    model = Trackastra.from_pretrained(
        name=name,
        device=device,
    )
    imgs, masks = example_data_hela()

    _ = model._predict(imgs, masks)
    assert True
