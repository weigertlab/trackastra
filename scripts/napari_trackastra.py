import torch 
import napari
import numpy as np
from napari.layers import Image, Labels
from napari.utils import progress
from typing import List
from magicgui import magicgui
import argparse 
import tifffile 
from tqdm import tqdm
from pathlib import Path
from trackastra.utils import normalize
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
 
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_widget(model_path:Path):
    @magicgui(call_button="track", model_path={"label": "Model Path", "mode": "d"})
    def my_widget(imgs: Image, masks:Labels, model_path:Path=model_path) -> List[napari.types.LayerDataTuple]:
        img = np.asarray(imgs.data)
        mask = np.asarray(masks.data)
        imgs = np.stack([normalize(x) for x in img])

        model = Trackastra.load_from_folder(model_path, device=device)
        
        track_graph = model.track(img, mask, mode="greedy", progbar_class=progress)  # or mode="ilp"

        # Visualise in napari
        napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)
        _, masks_tracked = graph_to_ctc(track_graph,mask,outdir=None)
        masks.visible = False
        return [(napari_tracks, dict(name='tracks',tail_length=10), "tracks"), (masks_tracked, dict(name='masks_tracked', opacity=0.3), "labels")]
    
    return my_widget

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img", type=str, default=None) 
    parser.add_argument("-m", "--mask", type=str, default=None) 
    parser.add_argument("--model", type=str, default=None)
    
    args=parser.parse_args()


    viewer = napari.Viewer()
    
    if args.img is not None:
        f_img = Path(args.img)
        imgs = np.stack(tuple(tifffile.imread(f) for f in tqdm(tuple(sorted(f_img.glob("*.tif"))))))
        viewer.add_image(imgs)
    else: 
        imgs = None 
        
    if args.mask is not None:
        f_mask = Path(args.mask)
        masks = np.stack(tuple(tifffile.imread(f) for f in tqdm(tuple(sorted(f_mask.glob("*.tif"))))))
        viewer.add_labels(masks)

    if args.model is not None:
        model_path = Path(args.model)
    else: 
        model_path = None
    viewer.window.add_dock_widget(create_widget(args.model))

    napari.run()