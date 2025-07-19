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


@pytest.mark.parametrize("name", ["ctc", "general_2d"])
# @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [None, 3])
def test_integration(name, device, batch_size):
    """
    Test that the number of edges and nodes in the track graph is consistent with the pretrained model.
    """
    length_edges_nodes = {"ctc": (3122, 3269), "general_2d": (3121, 3268)}

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda not available")
    elif device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps not available")

    model = Trackastra.from_pretrained(
        name=name,
        device=device,
    )
    imgs, masks = example_data_hela()
    track_graph, _ = model.track(imgs, masks, batch_size=batch_size)
    assert (len(track_graph.edges), len(track_graph.nodes)) == length_edges_nodes[name]


if __name__ == "__main__":
    test_integration("ctc", "mps", 3)
