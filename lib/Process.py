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

    def process(self, norm=False):

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
                torch.no_grad ensures that we are not remembering
                all the gradients of each map,  this is essential
                for memory reasons.
            """
            for i, tile in enumerate(tiles):
                print(f"tile: {i}")
                tile = tile.to(self.device)
                print(f"tile: {torch.min(tile), torch.max(tile)}")
                print(f"tile: {tile.shape}")
                out = unet(tile)
                print(f"out tile: {torch.min(out), torch.max(out)}")
                print(f"out: {out.shape}")
                outs.append(out)
        
        # convert output tiles from tensor to numpy arrays
        outs    = [tile.cpu() for tile in outs]
        outs_np = [tile.numpy() for tile in outs] # (1,1,64,64,64)
        outs_np = [t.squeeze()  for t in outs_np] # squeeze

        if norm:
            map_raw = self.load_map(self.proc_map_name, 
                    False)
            a = np.min(map_raw.map) 
            b = np.max(map_raw.map)
            Min = np.min(outs_np)
            Max = np.max(outs_np)
            print(a,b,Min,Max)
            outs_np = (((b - a) * (outs_np - Min)) / \
                    (Max - Min)) + a
            print(np.min(outs_np), np.max(outs_np))

        # assign to map
        # recompose
        sample.tiles = outs_np
        sample.map = outs_np
        sample.pred = outs_np


        print(type(sample.map))
        print(np.min(sample.map),np.max(sample.map))
        sample.recompose(map=True)
        print(np.min(sample.map),np.max(sample.map))
        print(f"type of map: {type(sample.map)}")


        # path to outputs
        path = self.training_outputs_path
        if 'denoised' not in os.listdir(path):
            os.mkdir(os.path.join(path,'denoised'))

        sample.save_map(map_rec=True, path=os.path.join(path,'denoised',
            f"e_{self.proc_epoch}_{self.proc_map_name}"))
        print(f"Saved map to: out/{self.proc_model}/denoised/e_{self.proc_epoch}_{self.proc_map_name}")


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
        print(sample.map.shape)
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

