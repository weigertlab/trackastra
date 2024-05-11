# *Trackastra* - Tracking by Association with Transformers

*Trackastra* links already segmented cells in a microscopy timelapse using a learnt transformer model that was trained on a diverse set of microscopy videos.
<!-- TODO ## Overview, including an image/video -->

If you are using this code in your research, please cite
> Benjamin Gallusser and Martin Weigert, *Trackastra - Transformer-based cell tracking for live-cell microscopy*, 2024 (in preparation).


## Installation
```bash
pip install git+https://github.com/weigertlab/trackastra.git
```

For tracking with an integer linear program:
```bash
conda install -c conda-forge -c gurobi -c funkelab ilpy
pip install "trackastra[ilp] @ git+https://github.com/weigertlab/trackastra.git"
```

## Usage

> This package is still under active development, please expect breaking changes in the future. If you encounter any problems please file an [issue](https://github.com/weigertlab/trackastra/issues) on the GitHub repo.

### Tracking with a pretrained model

Consider the following python example script for tracking already segmented cells. All you need are the following two numpy arrarys:
- `imgs`: a microscopy time lapse of shape `time,(z),y,x`.
- `masks`: corresponding instance segmentation of shape `time,(z),y,x`.

No hyperparameters to choose :)

```python
import numpy as np
from trackastra.utils import normalize
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks

imgs = ...
masks = ...

# Normalize your images
imgs = np.stack([normalize(x) for x in imgs])

# Load a pretrained model
model = Trackastra.load_pretrained(name="ctc", device="cuda")  # or device="cpu"

# Track the cells
track_graph = model.track(imgs, masks, mode="greedy")  # or mode="ilp"


# Write to cell tracking challenge format
ctc_tracks, masks_tracked = graph_to_ctc(
      track_graph,
      masks,
      outdir="tracked",
)

# Visualise in napari
napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)
v = napari.Viewer()
v.add_image(imgs)
v.add_labels(masks_tracked)
v.add_tracks(data=napari_tracks, graph=napari_tracks_graph)
```

### Training a model on your own data

To run an example
- clone this repository and got into the scripts directory with `cd trackastra/scripts`.
- download the [Fluo-N2DL-HeLa](http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip) dataset from the Cell Tracking Challenge into `data/ctc`.

Now, run
```bash
python train.py --config example_config.yaml
```

Generally, training data needs to be provided in the [Cell Tracking Challenge (CTC) format](http://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf), i.e. annotations are located in a folder containing one or several subfolders named `TRA`, with masks and tracklet information.
