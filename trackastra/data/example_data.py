import tifffile 
from pathlib import Path


root = Path(__file__).parent/'resources'

def test_data_bacteria():
    """ Bacteria images and masks from

    Van Vliet et al. Local interactions lead to spatially correlated gene expression levels in bacterial group (2018)

    subset of timelapse trpL/150310-11
    """    
    img = tifffile.imread(root/'trpL_150310-11_img.tif')
    mask = tifffile.imread(root/'trpL_150310-11_mask.tif')
    return img, mask 


def test_data_hela():
    """ Hela data from the cell tracking challenge 

    Neumann et al. Phenotypic profiling of the human genome by time-lapse microscopy reveals cell division genes (2010)

    subset of Fluo-N2DL-HeLa/train/02
    """
    img = tifffile.imread(root/'Fluo_Hela_02_img.tif')
    mask = tifffile.imread(root/'Fluo_Hela_02_ERR_SEG.tif')
    return img, mask 
