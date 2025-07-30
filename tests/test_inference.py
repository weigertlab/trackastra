import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
from trackastra.model import Trackastra


def test_empty_intermediate_mask():
    """minimal test case with an intermediate empty mask"""

    imgs = np.zeros((3, 100, 100), dtype=np.uint16)
    masks = np.zeros((3, 100, 100), dtype=np.uint16)

    masks[0, 5:10, 5:10] = 1  # Detection in frame 0
    # frame 1 is empty
    masks[2, 80:85, 80:85] = 2  # Detection in frame 2

    model = Trackastra.from_pretrained("general_2d", device="cpu")

    predictions = model._predict(imgs, masks)

    model._track_from_predictions(predictions)

    model.track(
        imgs,
        masks,
        mode="greedy",
    )


if __name__ == "__main__":
    test_empty_intermediate_mask()
