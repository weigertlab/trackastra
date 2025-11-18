import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pytest
import torch
from trackastra.data import example_data_bacteria, example_data_hela
from trackastra.model import Trackastra

try:
    import trackastra_pretrained_feats  # noqa: F401

    SAM2_TEST = True
except ModuleNotFoundError:
    SAM2_TEST = False


# Mark all tests in this module as core/inference tests
pytestmark = pytest.mark.core


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


def limit_mps_memory(target=5 * 2**30):
    """Limit MPS memory usage to target bytes."""
    if torch.backends.mps.is_available():
        # Get Metal's recommended max working set (bytes)j
        max_bytes = torch.mps.recommended_max_memory()
        print(f"Recommended max MPS memory: {max_bytes / 2**30:.2f} GB.")
        max_bytes = int(
            max_bytes / 0.8
        )  # 80% is recommended, this gets the full amount

        fraction = target / max_bytes
        if fraction > 1.0:
            raise ValueError(
                f"Target memory limit ({target / 2**30:.2f} GB) exceeds maximum available memory ({max_bytes / 2**30:.2f} GB)."
            )
        else:
            torch.mps.set_per_process_memory_fraction(fraction)
            print(
                f"Set MPS memory limit to {target / 2**30:.2f} GB ({fraction * 100:.1f}% of max)."
            )


@pytest.mark.skipif(
    not SAM2_TEST, reason="Package for using SAM2 features not installed"
)
@pytest.mark.parametrize("device", ["mps", "cuda"])
@pytest.mark.parametrize("batch_size", [1])
def test_integration_SAM2(device, batch_size):
    """
    Test that the number of edges and nodes in the track graph is consistent with the pretrained model.
    """

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda not available")
    elif device == "mps":
        if not torch.backends.mps.is_available():
            pytest.skip("mps not available")
        limit_mps_memory()

    model = Trackastra.from_pretrained(
        name="general_2d_w_SAM2_features",
        device=device,
    )
    imgs, masks = example_data_bacteria()
    track_graph, _ = model.track(imgs, masks, batch_size=batch_size)

    assert len(track_graph.edges) == 126
    assert len(track_graph.nodes) == 128


if __name__ == "__main__":
    test_integration_SAM2("mps", 1)
