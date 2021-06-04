# Standard library imports
import sys
sys.path.insert(1, 'ml_toolbox/src')
import torch
import os

# Related third party imports (Jola's toolbox)
from .ml_toolbox.src.multi_proteins import load_data, \
    preproces_data


class Data:
    
    def __init__(self, path):
        self.path_global = path
        self.ls = os.listdir(self.path_global)
        if '.gitkeep' in self.ls:
            self.ls.remove('.gitkeep')
        self.ls = [float(f) for f in self.ls]
        self.ls.sort()
        self.maps = [os.listdir(self.path_global + 
            str(f)) for f in self.ls]
        self.no_of_batches = len(self.ls)
        self.batch_id = None
        self.tiles = None
        self.get_contents()
        self.batch_sizes = [len(d) for d in self.rest_contents]
        self.batch_maps = None


    def load_batch(self, i, n=None):
        self.batch_id = i
        self.batch = None
        data = load_data(self.path_global, [self.ls[i]], n)

        data = preproces_data(data,
                norm=True,
                mode='tile',
                cshape=64,
                norm_vox=1.4,
                norm_vox_lim=(1.4,1.4)
                )

        self.batch = data
        self.convert()

    def load_maps(self, i, n=None):
        self.batch_id = i
        self.batch_maps = load_data(self.path_global,
                [self.ls[i]], n)

    def convert(self):
        self.tiles = None
        assert self.batch != None, "Need to load batch"
        tiles_np = [tile for d in self.batch for tile in d.tiles]
        # convert tiles to tensors
        tiles_torch = [torch.tensor(t) for t in tiles_np]
        # unsqueeze so the shape of tiles are [bs,c,h,w,d]
        tiles_uncat = [t.unsqueeze(0).unsqueeze(0) \
                             for t in tiles_torch]
        # concatonate tiles into one array
        self.tiles = torch.cat(tiles_uncat, 0)



    def get_contents(self):
        # obtains the contents of the sub directories of
        # the global directory, stores the in 
        # self.rest_contents
        dir_paths = [self.path_global + str(b) + '/' \
                for b in self.ls]
        dir_contents = [os.listdir(dir_path) 
                for dir_path in dir_paths]
        self.rest_contents = dir_contents

