# adding utils and ml_toolbox to PATH
import sys
sys.path.insert(1, 'utils/')
sys.path.insert(1, 'utils/ml_toolbox/src/')
sys.path.insert(1, 'net')
 
# utils imports 
from unet import UNet
from proteins import Sample

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

class Process:

    def __init__(self, Network, input_data, 
            out_path = '../out', data_path = '../data'):

        # INITS
        self.Network = Network
        self.input_data = input_data
        self.out_path = out_path
        self.data_path = data_path
        self.test_maps_path = os.path.join(data_path,
                '1.5/1.0')
        self.available_test_maps = os.listdir(self.test_maps_path)

        # INPUT DATA
        self.map_name      = input_data["proc_map_name"]
        self.cshape        = input_data["cshape"]
        self.margin        = input_data["margin"]
        self.norm          = input_data["norm"]
        self.norm_vox      = input_data["norm_vox"]
        self.norm_vox_lim  =(input_data["norm_vox_min"],
                             input_data["norm_vox_max"])
        self.proc_model    = input_data["proc_model"]
        self.proc_epoch    = input_data["proc_epoch"]
        self.proc_map_name = input_data["proc_map_name"]
        self.device        = torch.device(input_data["device"])

        # GET PATH TO MODEL
        self.models_path = self.get_path_to_models()
        self.training_outputs_path =  \
            self.get_path_to_training_outputs()

    def process(self):

        # Load tiles
        sample = self.load_map(self.proc_map_name)
        tiles = sample.tiles

        # Load model
        unet = self.Network.UNet
        # Load the trained unet model
        unet.load_state_dict(torch.load(self.model_path,
            map_location=self.device))
        unet = unet.to(self.device)  # move unet to device

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
        sample.pred = outs_np
        sample.recompose()
        sample.map = sample.pred

        # path to outputs
        path = self.training_outputs_path
        if 'denoised' not in os.listdir(path):
            os.mkdir(os.path.join(path,'denoised'))

        sample.save_map(os.path.join(path,'denoised',
            f"epoch_{self.proc_epoch}.map")
        print(f"Saved map to: data/downloads/1.0preds/{model[:-3]}.map")




    def get_path_to_models(self):
        p = os.path.join(self.out_path, self.proc_model,
                'models', f"epoch_{self.proc_epoch}.pt")
        return p

    def get_path_to_training_outputs(self):
        p = os.path.join(self.out_path, self.proc_model)

        return p



    def load_map(self, map_name):

        assert map_name in self.available_test_maps, \
        f"The map {map_name} is not stored, try one from" \
                + f"{self.available_test_maps}"

        map_path = os.path.join(self.test_maps_path,
          map_name)
        sample = Sample(1, map_path) 
        sample.decompose(
                cshape       = self.cshape,
                margin       = self.margin,
                norm         = self.norm,
                norm_vox     = self.norm_vox,
                norm_vox_lim = self.norm_vox_lim)
        return sample





# def Process(Network, input_data, out_path = '../out', 
#         data_path = '../data' ):
# 
#     test_maps_path = os.path.join(data_path, '1.5/1.0')
#     available_test_maps = os.listdir(test_maps_path)
# 
#     device = input_data["device"]
#     proc_model = input_data["proc_model"]
#     proc_epoch = input_data["proc_epoch"]
# 
#     device = torch.device(device)
# 
#     path = 'data/downloads/1.5/' # global path to test maps
# 
#     # Load all maps and store ids
#     data = Data(path)
#     data.load_batch(0)
#     ids = []
#     for Map in data.batch:
#         map_id = Map.id
#         ids.append(map_id)
# 
#     # Search for index of map to be denoised
#     index = None
#     for i, ID in enumerate(ids):
#         if args.id == ID:
#             index = i
#     assert index != None, "No map with this ID has been found"
# 
#     tiles = data.batch[0].tiles
#     tiles = list_to_torch(tiles)
# 
# 
#     # Model (UNET)
#     unet  = UNet()
#     model = args.model
# 
# 
#     # Load the trained unet model
#     unet.load_state_dict(torch.load(f'trained_models/{model}',
#         map_location=device))
#     unet = unet.to(device)  # move unet to device
# 
#     # pass the test tiles through the trained network
#     print("Testing model...")
#     outs = []
#     with torch.no_grad():
#         """
#             torch.no_grad ensures that we are not remembering
#             all the gradients of each map,  this is essential
#             for memory reasons.
#         """
#         for i, tile in enumerate(tiles):
#             print(f"tile: {i}")
#             tile = tile.to(device)
#             out = unet(tile)
#             outs.append(out)
# 
#     # convert output tiles from tensor to numpy arrays
#     outs    = [tile.cpu() for tile in outs]
#     outs_np = [tile.numpy() for tile in outs] # (1,1,64,64,64)
#     outs_np = [t.squeeze()  for t in outs_np] # squeeze
#     # assign to pred
#     # recompose
#     data.batch[index].pred = outs_np
#     data.batch[index].recompose()
#     data.batch[index].map = data.batch[index].pred
#     data.batch[index].save_map(f'data/downloads/1.0preds/{model[:-3]}.map')
#     print(f"Saved map to: data/downloads/1.0preds/{model[:-3]}.map")
