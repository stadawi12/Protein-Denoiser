# adding utils and ml_toolbox to PATH
import sys
sys.path.insert(1, 'utils/')
sys.path.insert(1, 'utils/ml_toolbox/')
sys.path.insert(1, 'utils/ml_toolbox/src/')
 
# utils imports 
from unet import UNet
from load import Data
from csv_filter import get_entries

# library imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os

def list_to_torch(lst):
    lst = [torch.from_numpy(tile) for tile in lst]
    lst = [tile.unsqueeze(0).unsqueeze(0) for tile in lst]
    return lst


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str,
        help="ID of map to be denoised")
parser.add_argument('--model', type=str,
        help="Name of model you want to use")
args = parser.parse_args()

# Check if machine has a GPU
if torch.cuda.is_available():
    dev = "cuda:0"
    device = torch.device(dev)
else:
    dev = "cpu"
    device = torch.device(dev)

path = 'data/downloads/1.5/' # global path to test maps

# Load all maps and store ids
data = Data(path)
data.load_batch(0)
ids = []
for Map in data.batch:
    map_id = Map.id
    ids.append(map_id)

# Search for index of map to be denoised
index = None
for i, ID in enumerate(ids):
    if args.id == ID:
        index = i
assert index != None, "No map with this ID has been found"

print(f"Shape of loaded map: {data.batch[index].map.shape}")
plt.imshow(data.batch[index].map[50])
plt.show()

tiles = data.batch[0].tiles
tiles = list_to_torch(tiles)


# Model (UNET)
unet  = UNet()
model = args.model


# Load the trained unet model
unet.load_state_dict(torch.load(f'trained_models/{model}',
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
    for i, tile in enumerate(tiles):
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
print(f"shape of map before recomposing: {data.batch[index].map.shape}")
data.batch[index].pred = outs_np
print(f"number of tiles used for recomposing: {len(data.batch[index].pred)}")
print(f"Shape of tiles used for recomposing: {data.batch[index].pred[0].shape}")
data.batch[index].recompose()
print(f"Shape of recmposed map: {data.batch[index].map.shape}")
plt.imshow(data.batch[index].map[50])
plt.show()
data.batch[index].map = data.batch[index].pred
data.batch[index].save_map(f'data/downloads/1.0preds/{model[:-3]}.map')
print(f"Saved map to: data/downloads/1.0preds/{model[:-3]}.map")
