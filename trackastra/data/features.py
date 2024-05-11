import itertools

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

# the property keys that are supported for 2 and 3 dim

_PROPERTIES = {
    2: {
        # FIXME: The only image regionprop possible now (when compressing) is mean_intensity,
        # since we store a mask with the mean intensity of each detection as the image.
        "regionprops": (
            "label",
            "area",
            "intensity_mean",
            "eccentricity",
            "solidity",
            "inertia_tensor",
        ),
        # faster
        "regionprops2": (
            "label",
            "area",
            "intensity_mean",
            "inertia_tensor",
        ),
        "patch_regionprops": (
            "label",
            "area",
            "intensity_mean",
            "inertia_tensor",
        ),
    },
    3: {
        "regionprops2": (
            "label",
            "area",
            "intensity_mean",
            "inertia_tensor",
        ),
        "patch_regionprops": (
            "label",
            "area",
            "intensity_mean",
            "inertia_tensor",
        ),
    },
}


def extract_features_regionprops(
    mask: np.ndarray,
    img: np.ndarray,
    labels: np.ndarray,
    properties="regionprops2",
):
    ndim = mask.ndim
    assert ndim in (2, 3)
    assert mask.shape == img.shape

    prop_dict = _PROPERTIES[ndim]
    if properties not in prop_dict:
        raise ValueError(f"properties must be one of {prop_dict.keys()}")
    properties_tuple = prop_dict[properties]

    assert properties_tuple[0] == "label"

    labels = np.asarray(labels)

    # remove mask labels that are not present
    # not needed, remove for speed
    # mask[~np.isin(mask, labels)] = 0

    df = pd.DataFrame(
        regionprops_table(mask, intensity_image=img, properties=properties_tuple)
    )
    assert df.columns[0] == "label"
    assert df.columns[1] == "area"

    # the bnumber of inertia tensor columns depends on the dimensionality
    n_cols_inertia = ndim**2
    assert np.all(["inertia_tensor" in col for col in df.columns[-n_cols_inertia:]])

    # Hack for backwards compatibility
    if properties in ("regionprops", "patch_regionprops"):
        # Nice for conceptual clarity, but does not matter for speed
        # drop upper triangular part of symmetric inertia tensor
        for i, j in itertools.product(range(ndim), repeat=2):
            if i > j:
                df.drop(f"inertia_tensor-{i}-{j}", axis=1, inplace=True)

    table = df.to_numpy()
    table[:, 1] *= 0.001
    table[:, -n_cols_inertia:] *= 0.01
    # reorder according to labels
    features = np.zeros((len(labels), len(df.columns) - 1))

    # faster than iterating over pandas dataframe
    for row in table:
        # old version with tuple indexing, slow.
        # n = labels.index(int(row.label))
        # features[n] = row.to_numpy()[1:]

        # Only process regions present in the labels
        n = np.where(labels == int(row[0]))[0]
        if len(n) > 0:
            # Remove label column (0)!
            features[n[0]] = row[1:]

    return features


def extract_features_patch(
    mask: np.ndarray,
    img: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    width_patch: int = 16,
):
    """16x16 Image patch around detection."""
    ndim = mask.ndim
    assert ndim in (2, 3) and mask.shape == img.shape
    if len(coords) == 0:
        return np.zeros((0, width_patch * width_patch))

    pads = (width_patch // 2,) * ndim

    img = np.pad(
        img,
        tuple((p, p) for p in pads),
        mode="constant",
    )

    coords = coords.astype(int) + np.array(pads)

    ss = tuple(
        tuple(slice(_c - width_patch // 2, _c + width_patch // 2) for _c in c)
        for c in coords
    )
    fs = tuple(img[_s] for _s in ss)

    # max project along z if 3D
    if ndim == 3:
        fs = tuple(f.max(0) for f in fs)

    features = np.stack([f.flatten() for f in fs])
    return features
