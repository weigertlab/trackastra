from pathlib import Path

import tifffile

root = Path(__file__).parent / "resources"


def example_data_bacteria():
    """Bacteria images and masks from.

    Van Vliet et al. Spatially Correlated Gene Expression in Bacterial Groups: The Role of Lineage History, Spatial Gradients, and Cell-Cell Interactions (2018)
    https://doi.org/10.1016/j.cels.2018.03.009

    subset of timelapse trpL/150310-11
    """
    img = tifffile.imread(root / "trpL_150310-11_img.tif")
    mask = tifffile.imread(root / "trpL_150310-11_mask.tif")
    return img, mask


def example_data_hela():
    """Hela data from the cell tracking challenge.

    Neumann et al. Phenotypic profiling of the human genome by time-lapse microscopy reveals cell division genes (2010)

    subset of Fluo-N2DL-HeLa/train/02
    """
    img = tifffile.imread(root / "Fluo_Hela_02_img.tif")
    mask = tifffile.imread(root / "Fluo_Hela_02_ERR_SEG.tif")
    return img, mask


def example_data_fluo_3d():
    """Fluo-N3DH-CHO data from the cell tracking challenge.

    Dzyubachyk et al. Advanced Level-Set-Based Cell Tracking in Time-Lapse Fluorescence Microscopy (2010)

    subset of Fluo-N3DH-CHO/train/02
    """
    img = tifffile.imread(root / "Fluo-N3DH-CHO_02_img.tif")
    mask = tifffile.imread(root / "Fluo-N3DH-CHO_02_ERR_SEG.tif")
    return img, mask
