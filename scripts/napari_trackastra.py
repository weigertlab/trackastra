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

def track(model, imgs, masks, **kwargs):    
    imgs = np.stack([normalize(x) for x in imgs])
    
    
    track_graph = model.track(imgs, masks, mode="greedy", 
                                max_distance=128,
                                progbar_class=progress,
                                **kwargs)  # or mode="ilp"
    # Visualise in napari
    df, masks_tracked = graph_to_ctc(track_graph,masks,outdir=None)
    napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)
    return track_graph, masks_tracked, napari_tracks


def create_widget(model_path:Path):
    @magicgui(call_button="track", 
              model_path={"label": "Model Path", "mode": "d"})
    def my_widget(img_layer: Image, mask_layer:Labels, model_path:Path=model_path, distance_costs:bool=False) -> List[napari.types.LayerDataTuple]:
        if model_path.exists():
            model = Trackastra.from_folder(model_path, device=device)
        else: 
            model = Trackastra.from_pretrained(model_path.name, device=device)
        imgs = np.asarray(img_layer.data)
        masks = np.asarray(mask_layer.data)
        track_graph, masks_tracked, napari_tracks = track(model, imgs, masks, use_distance=distance_costs)
        mask_layer.visible = False
        return [(napari_tracks, dict(name='tracks',tail_length=5), "tracks"), (masks_tracked, dict(name='masks_tracked', opacity=0.3), "labels")]
    
    return my_widget

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img", type=str, default=None) 
    parser.add_argument("-m", "--mask", type=str, default=None) 
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    
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
        
    if args.debug:
        model = Trackastra.load_from_folder(model_path, device=device)
        track_graph, masks_tracked, napari_tracks = track(model, imgs, masks)
    else:
        viewer.window.add_dock_widget(create_widget(args.model))

    napari.run()