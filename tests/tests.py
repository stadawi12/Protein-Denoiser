# Adding paths to modules to be tested
import sys
import os
sys.path.append('../lib/net')     # utils
sys.path.append('../lib/preprocess')     # utils
sys.path.append('../lib/utils')     # utils

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
# Unit test import
import unittest

# Libraries needed for testing
import numpy as np
import torch

# Importing my modules to be tested
from unet import UNet, crop_img
from res_scraper import get_reses
from size_scraper import get_dims
from download_data import download
enablePrint()


class Test_functions(unittest.TestCase):

    def test_unet_shape(self):
        """
            ensure that input shape is the same as output
            shape, I use size 64 voxals for testing.
        """
        blockPrint()
        unet = UNet()
        inpt = torch.randn(1,1,64,64,64)
        out  = unet(inpt)
        self.assertTrue(inpt.shape == out.shape)
        enablePrint()

    def test_crop_img_shape(self):
        """
            test to see if tensor shape is equal to the shape
            of the target.
        """
        tensor = torch.randn(1,64,64,64,64)
        target = torch.randn(1,64,60,60,60)
        tensor = crop_img(tensor, target)
        self.assertTrue(tensor.shape == target.shape)

    def test_crop_img_centre(self):
        """
            test to see if crop_img takes out the centre
            of the tensor
        """
        blockPrint()
        tensor = torch.randn(1,1,5,5,5)
        target = torch.randn(1,1,3,3,3)
        tensor_crop = crop_img(tensor, target)
        tensor_centre = tensor[0:1,0:1,1:4,1:4,1:4]
        enablePrint()
        self.assertTrue(torch.all(
                        tensor_crop.eq(tensor_centre)))

    def test_get_reses1(self):
        """
            test if get_reses(4) returns a list of length
            4.
        """
        blockPrint()
        reses = get_reses(4)
        enablePrint()
        l = len(reses)
        self.assertTrue(l == 4)

    def test_get_dims(self):
        """
            test if get_dims(4) returns a list of length
            4.
        """
        blockPrint()
        dims = get_dims(4)
        enablePrint()
        l = len(dims)
        self.assertTrue(l == 4)

    def test_download_map(self):
        blockPrint()
        entry = 'EMD-0552'
        tail  = 'emd_0552_half_map_1.map.gz'
        path  = 'fake_maps/'
        message = 'Map stored'
        out = download(entry, tail, path=path)
        enablePrint()
        self.assertTrue(out == message)

    def test_download_map_gz(self):
        blockPrint()
        entry = 'EMD-10161'
        tail  = 'emd_10161_half_map_1.map.gz'
        path  = 'fake_maps/'
        message = 'Map stored'
        out = download(entry, tail, path=path)
        enablePrint()
        self.assertTrue(out == message)

    def test_download(self):
        blockPrint()
        entry = 'EMD-3650'
        tail  = 'emd_3650_half_map_1.map.gz'
        path  = 'fake_maps/'
        download(entry, tail, path=path)
        contents = os.listdir('fake_maps/')
        filename = '3650.map.gz'
        enablePrint()
        self.assertTrue(filename in contents)
        os.remove('fake_maps/3650.map.gz')
        


if __name__ == '__main__':
    unittest.main()
        



