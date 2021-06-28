# adding utils and ml_toolbox to PATH
import sys
sys.path.insert(1, 'utils/')
sys.path.insert(1, 'utils/ml_toolbox/src/')
sys.path.insert(1, 'net')
 
# utils imports 
import unet
from proteins import Sample
from Inputs import Read_Input

# library imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os


class Process:

    def __init__(self, Network, input_data, 
            out_path = 'out', data_path = 'data'):

        # INITS
        self.Network = Network
        self.out_path = out_path
        self.data_path = data_path
        self.test_maps_path = data_path
        self.available_test_maps = os.listdir(self.test_maps_path)

        # INPUT DATA
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
        if input_data["proc_res"] != None:
            self.res       = input_data["proc_res"]

        # GET PATH TO MODEL
        self.models_path = self.get_path_to_models()
        self.training_outputs_path =  \
            self.get_path_to_training_outputs()

    # process function for synthetic data
    def process_synthetic(self):
        res = self.res
        # Load map and normalise it
        sample = self.load_map(self.proc_map_name,
                self.norm)
        print(f"Shape of map: {sample.map.shape}")
        print("min, max of map: {}, {}".format(
            np.min(sample.map), np.max(sample.map)))
        tiles = sample.tiles
        tiles = self.to_torch_list(tiles)

        # Load model
        unet = self.Network.UNet()
        # Load the trained unet model
        unet.load_state_dict(torch.load(self.models_path,
            map_location=self.device))
        unet = unet.to(self.device)  # move unet to device

        print(f"Denoising {self.proc_map_name}...")
        outs = []
        with torch.no_grad():
            """
                torch.no_grad ensures that we are not 
                remembering all the gradients of each map,
                this is essential for memory reasons.
            """
            # For each tile
            for i, tile in enumerate(tiles):
                print(f"tile: {i}")
                # Move tile to device cpu or gpu
                tile = tile.to(self.device)
                # Pass tile through network
                out = unet(tile)
                # Append the output file to outs
                outs.append(out)
        
        # convert output tiles from tensor to numpy arrays
        # First move them to the cpu
        outs    = [tile.cpu() for tile in outs]
        # Convert each tile from tensort to numpy
        outs_np = [tile.numpy() for tile in outs] 
        # Squeeze from (1,1,64,64,64) to (64,64,64)
        outs_np = [t.squeeze()  for t in outs_np] # squeeze

        # Normalise tiles so that they have same range as 
        # the noisy map
        # Load noisy map with norm = False
        map_raw = self.load_map(self.proc_map_name, 
                False)
        # Range set by noisy raw map
        a = np.min(map_raw.map) 
        b = np.max(map_raw.map)
        # Get minimum and maximum of current output tiles
        Min = np.min(outs_np)
        Max = np.max(outs_np)
        # Function to rescale tiles from [Min,Max]->[a,b]
        outs_np = (((b - a) * (outs_np - Min)) / \
                (Max - Min)) + a
        # Ensure the new min and max is same as a and b
        print(a, b)
        print(np.min(outs_np), np.max(outs_np))

        # With tiles rescaled, recompose tiles into a map
        # set everything to outs_np as I do not know which
        # one needs it for recompose() to work,
        # only one of them does, just laziness here.
        sample.tiles = outs_np
        sample.map = outs_np
        sample.pred = outs_np

        # Using recompose to turn tiles into map
        sample.recompose(map=True)

        # path where to save the recomposed map
        path = os.path.join('denoised',res,
            self.proc_model,
            f"e_{self.proc_epoch}_{self.proc_map_name}")
            
        # Finally save the map to the path
        sample.save_map(map_rec=True, path=path)
                
        # Print statement saying where map was saved to
        print(f"Saved map to: denoised/{self.res}/" +
              f"{self.proc_model}/e_" +
              f"{self.proc_epoch}_{self.proc_map_name}")



    # process function for real data
    def process(self):

        # Load tiles
        sample = self.load_map(self.proc_map_name,
                self.norm)
        print(sample.map.shape)
        print(np.min(sample.map), np.max(sample.map))
        tiles = sample.tiles
        tiles = self.to_torch_list(tiles)

        # Load model
        unet = self.Network.UNet()
        # Load the trained unet model
        unet.load_state_dict(torch.load(self.models_path,
            map_location=self.device))
        unet = unet.to(self.device)  # move unet to device

        print(f"Denoising {self.proc_map_name}...")
        outs = []
        with torch.no_grad():
            """
                torch.no_grad ensures that we are not 
                remembering all the gradients of each map,
                this is essential for memory reasons.
            """
            for i, tile in enumerate(tiles):
                print(f"tile: {i}")
                tile = tile.to(self.device)
                out = unet(tile)
                outs.append(out)
        
        # convert output tiles from tensor to numpy arrays
        outs    = [tile.cpu() for tile in outs]
        outs_np = [tile.numpy() for tile in outs]
        outs_np = [t.squeeze()  for t in outs_np] # squeeze

        # recompose
        sample.tiles = outs_np
        sample.map = outs_np
        sample.pred = outs_np

        sample.recompose(map=True)

        # path to outputs
        path = self.training_outputs_path
        if 'denoised' not in os.listdir(path):
            os.mkdir(os.path.join(path,'denoised'))

        sample.save_map(map_rec=True, path=os.path.join(
            path,'denoised',
            f"e_{self.proc_epoch}_{self.proc_map_name}"))

        print(f"Saved map to: out/{self.proc_model}" +
        f"/denoised/e_{self.proc_epoch}_" +
        f"{self.proc_map_name}")


    def get_path_to_models(self):
        p = os.path.join(self.out_path, self.proc_model,
                'models', f"epoch_{self.proc_epoch}.pt")
        return p

    def get_path_to_training_outputs(self):
        p = os.path.join(self.out_path, self.proc_model)

        return p

    def load_map(self, map_name, norm):

        assert map_name in self.available_test_maps, \
        f"The map {map_name} is not stored, try one from" \
                + f"{self.available_test_maps}"

        map_path = os.path.join(self.test_maps_path, map_name)
        sample = Sample(1, map_path) 
        sample.decompose(
                cshape       = self.cshape,
                margin       = self.margin,
                norm         = norm,
                norm_vox     = self.norm_vox,
                norm_vox_lim = self.norm_vox_lim)
        return sample

    def to_torch_list(self, lst):
        lst = [torch.from_numpy(tile) for tile in lst]
        lst = [tile.unsqueeze(0).unsqueeze(0) for tile in lst]
        return lst

if __name__ == '__main__':

    input_data = Read_Input('../inputs.yaml')
    p1 = Process(unet, input_data, out_path='../out', 
            data_path='../data')

