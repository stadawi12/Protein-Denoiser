import sys
sys.path.insert(1, 'ml_toolbox/src')

import torch
from torch.utils.data import DataLoader, Dataset
from proteins import Sample
import os

class dataset(Dataset):
    """Class that takes care of loading data"""

    def __init__(self, path_global='data', data_set="Training"):
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
        pass

    
    def clean(self, contents):
        """ Function that filters the list of maps removing 
            anything that doesn't end with a .map extension 
            e.g. the .gitkeep file
        """
        for name in contents:
            if name[-4:] != '.map':
                contents.remove(name)
        return contents

if __name__ == '__main__':
    pass

