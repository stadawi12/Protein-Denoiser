import sys
sys.path.insert(1, 'ml_toolbox/src')

import torch
from torch.utils.data import DataLoader, Dataset
from proteins import Sample
import os
from Inputs import Read_Input

class dataset(Dataset):
    """Class that takes care of loading data"""

    def __init__(self, input_data, path_global='data', data_set="Training"):
        """Initialising attributes

        Parameters
        ----------
        path_global : str
            Path pointing to the global data directory 
            '../data' or 'data' etc.
        data_set : str
            Specifies whether we want to load a training or 
            validation dataset
        """
        # INPUT DATA
        self.cshape = input_data["cshape"]
        self.margin = input_data["margin"]
        self.norm_vox = input_data["norm_vox"]
        self.norm_vox_lim = (input_data["norm_vox_min"],
                             input_data["norm_vox_max"])
        self.norm = input_data["norm"]

        # Allowed data_set values 
        data_sets = ['Training', 'Validation']

        # Check if data_set specified is allowed
        if data_set not in data_sets:
            raise ValueError("Invalid data_set. Expected one " + \
                "of : %s" % data_sets)

        # Check if path_global exists
        if not os.path.exists(path_global):
            raise NotADirectoryError("Directory does not exist")


        # Specifying attributes
        self.path_global = path_global
        self.data_set    = data_set

        # Specifying correct directories depending on data_set
        if self.data_set == 'Training':
            self.inpt_tail = '1.0'
            self.trgt_tail = '2.0'
        else:
            self.inpt_tail = '1.5'
            self.trgt_tail = '2.5'

        # Constructing full path to datasets
        self.inpt_path = os.path.join(path_global, self.inpt_tail)
        self.trgt_path = os.path.join(path_global, self.trgt_tail)

        # Get contents of input and target directories
        self.inpt_maps = os.listdir(self.inpt_path)
        self.trgt_maps = os.listdir(self.trgt_path)
        self.inpt_maps.sort()
        self.trgt_maps.sort()

        # Filter to remove anything that isn't a .map file
        self.inpt_maps = self.clean(self.inpt_maps)
        self.trgt_maps = self.clean(self.trgt_maps)

        # Assert input and target maps are ordered correctly
        assert self.inpt_maps == self.trgt_maps, \
                "Input and target maps do not match"

    def __len__(self):
        """Function required for DataLoader specifies the
        behaviour of Python's built-in len() function when
        applied to an object of our dataset class
        
        Returns
        -------
        number_of_examples : int
            number of training examples stored inside our
            directory
        """

        number_of_examples = len(self.inpt_maps)
        return number_of_examples

    def __getitem__(self, i):
        """Function required for DataLoader, specifies the 
        behaviour of indexing an object of our dataset class.
        For example, if x = dataset(foo,bar), x[i] would 
        load the i'th input and target maps into memory and
        return their tiles
        
        Parameter
        ---------
        i : int
            index corresponding to the i'th input/target 
            examples

        Returns
        -------
        inpt_tiles : torch.Tensor
            Tiles that will serve as input for the neural
            network
        trgt_tiles : torch.Tensor
            Tiles that will not be processed by the network
            but will be used in the loss function ("labels")
        """
        # Path to input maps, ex. data/1.0/0552.map
        inpt_path = os.path.join(self.inpt_path, 
                                 self.inpt_maps[i])
        # Path to target maps, ex. data/2.0/0552.map
        trgt_path = os.path.join(self.trgt_path, 
                                 self.trgt_maps[i])

        # Use Sample class from ml_toolbox (Jola's toolbox)
        # to deal with any processing of maps (loading,
        # tiling, saving etc.) 
        inpt_sample = Sample(
                float(self.inpt_tail), inpt_path)
        
        trgt_sample = Sample(
                float(self.trgt_tail), trgt_path)

        # Use the decompose function to tile our maps
        inpt_sample.decompose(
                cshape=self.cshape,
                margin=self.margin,
                norm=self.norm,
                norm_vox=self.norm_vox,
                norm_vox_lim=self.norm_vox_lim)

        trgt_sample.decompose(
                cshape=self.cshape,
                margin=self.margin,
                norm=self.norm,
                norm_vox=self.norm_vox,
                norm_vox_lim=self.norm_vox_lim)

        # Extract input tiles and transform from list of numpy 
        # arrays to a single torch tensor with shape
        # [no_of_tiles,1,64,64,64]
        inpt_tiles = inpt_sample.tiles
        inpt_tiles = [torch.tensor(t) for t in inpt_tiles]
        inpt_tiles = [t.unsqueeze(0).unsqueeze(0) \
                             for t in inpt_tiles]
        inpt_tiles = torch.cat(inpt_tiles, 0)

        # Same as above but for target tiles
        trgt_tiles = trgt_sample.tiles
        trgt_tiles = [torch.tensor(t) for t in trgt_tiles]
        trgt_tiles = [t.unsqueeze(0).unsqueeze(0) \
                             for t in trgt_tiles]
        trgt_tiles = torch.cat(trgt_tiles, 0)

        return inpt_tiles, trgt_tiles
    

    
    def clean(self, contents):
        """ Function that filters the list of maps removing 
            anything that doesn't end with a .map extension 
            e.g. the .gitkeep file
        """
        for name in contents:
            if name[-4:] != '.map':
                contents.remove(name)
        return contents

def collate_fn(tiles):
    batch_size = len(tiles)
    inpt_tiles = torch.cat([tiles[i][0] 
                    for i in range(batch_size)],0)
    trgt_tiles = torch.cat([tiles[i][1] 
                    for i in range(batch_size)],0)
    return inpt_tiles, trgt_tiles


if __name__ == '__main__':
    import time

    input_data = Read_Input('../../inputs.yaml')
    data = dataset(input_data, 
            path_global='../../data', data_set='Training')
    data_loader = DataLoader(data, batch_size = 1, shuffle=False,
            collate_fn=collate_fn, num_workers=2)

    tic = time.perf_counter()

    for inpt_tiles, trgt_tiles in data_loader:
        assert inpt_tiles.shape == trgt_tiles.shape, \
            "WARNING: Shapes do NOT match"
        print(inpt_tiles.shape, trgt_tiles.shape)

    toc = time.perf_counter()
    print(f"time: {(toc-tic) / 60}m")




