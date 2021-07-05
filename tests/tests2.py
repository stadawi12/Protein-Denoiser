import sys
import os
sys.path.insert(1, '../lib')
sys.path.insert(1, '../lib/net')
sys.path.insert(1, '../lib/utils')
sys.path.insert(1, '../lib/utils/ml_toolbox/src')

# Unit test import
import unittest

from Inputs import Read_Input
from Train import Train
from Process import Process
import unet
import shutil

input_data = Read_Input('inputs.yaml')

class Test_functions(unittest.TestCase):

    def test_train(self):
        """Passing a single tile through training"""
        Train(unet, inputs_path='inputs.yaml', data_path='data',
                out_path='out')
        # Check if an output directory has been created
        path = 'out/m1_e1_mbs1_test'
        is_exist = os.path.exists(path)
        self.assertTrue(is_exist)

        # Check if models directory has been created
        path_models = os.path.join(path, 'models')
        self.assertTrue(os.path.exists(path_models))

        # Check if model has been saved
        path_model = os.path.join(path_models, 'epoch_1.pt')
        self.assertTrue(os.path.exists(path_model))

        # Check if plots directory has been created
        path_plots = os.path.join(path, 'plots')
        self.assertTrue(os.path.exists(path_plots))

        # Check if plot has been saved
        path_plot = os.path.join(path_plots, 'e_1.png')
        self.assertTrue(os.path.exists(path_plot))

        # Check if inputs file has been copied
        path_inputs = os.path.join(path, 'inputs.yaml')
        self.assertTrue(os.path.exists(path_inputs))

        # Denoise tile using model
        p1 = Process(unet, input_data, out_path='out') 
        p1.process(input_data)

        # Check if denoised directory has been created
        path_denoised = os.path.join(path, 'denoised')
        self.assertTrue(os.path.exists(path_denoised))

        # Check if denoised map has been saved 
        path_denoised_map = os.path.join(path_denoised, 'e_1_3650.map')
        self.assertTrue(os.path.exists(path_denoised_map))

        # Remove main directory
        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
