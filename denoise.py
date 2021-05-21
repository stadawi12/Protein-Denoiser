# adding utils and ml_toolbox to PATH
import sys
sys.path.insert(1, 'utils/')
sys.path.insert(1, 'utils/ml_toolbox/')
sys.path.insert(1, 'utils/ml_toolbox/src/')
 
# utils imports 
from unet import UNet
import loader as l
from csv_filter import get_entries

# library imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-tm', '--model', type=str,
        help="Filename of trained model to be used for denoising")
parser.add_argument('--id', type=str,
        help="ID of map to be denoised")
parser.add_argument('-r', '--res', default=[1.0], type=list,
        help='"Resolution" name of folder containing maps (default set to [1.0])')
args = parser.parse_args()

# Check if machine has a GPU
if torch.cuda.is_available():
    dev = "cuda:0"
    device = torch.device(dev)
else:
    dev = "cpu"
    device = torch.device(dev)

path = 'data/downloads/1.0/' # global path to maps
hlf = args.res         # half_maps_1
n_ex = None             # number of examples to load

# load all saved maps
data = l.data_loader(path, hlf, n_ex) # half maps 1

# extract ids of loaded maps
data_ids = [d.id for d in data]

wanted_id = args.id
index = 0
for i, id in enumerate(data_ids):
    if id == wanted_id:
        index = i

data = [data[index]]

# Tile filtered maps
data_tiled = l.data_preprocess(data)

# extract tiles of half maps
tiles_np = [tile for d in data_tiled for tile in d.tiles]

# convert tiles to tensors
tiles_torch = [torch.tensor(t) for t in tiles_np]

# unsqueeze so the shape of tiles are [bs,c,h,w,d]
tiles_uncat = [t.unsqueeze(0).unsqueeze(0) \
                     for t in tiles_torch]

# concatonate tiles into one array
tiles_torch = torch.cat(tiles_uncat, 0)

# Model (UNET)
unet  = UNet()
setup = args.model

# Load the trained unet model
unet.load_state_dict(torch.load(f'trained_models/{setup}',
    map_location=device))
unet = unet.to(device)  # move unet to device

# pass the test tiles through the trained network
print("Testing model...")
outs = []
with torch.no_grad():
    """
        torch.no_grad ensures that we are not remembering
        all the gradients of each map,  this is essential
        for memory reasons.
    """
    for i, tile in enumerate(tiles_uncat):
        print(f"tile: {i}")
        tile = tile.to(device)
        out = unet(tile)
        outs.append(out)

# convert output tiles from tensor to numpy arrays
outs    = [tile.cpu() for tile in outs]
outs_np = [tile.numpy() for tile in outs] # (1,1,64,64,64)
outs_np = [t.squeeze()  for t in outs_np] # squeeze
# assign to pred
# recompose
data[0].pred = outs_np
data[0].recompose()
data[0].map = data[0].pred
data[0].save_map(f'data/downloads/1.0preds/{setup[:-3]}.map')
print(f"Saved map to: data/downloads/1.0preds/{setup[:-3]}.map")
