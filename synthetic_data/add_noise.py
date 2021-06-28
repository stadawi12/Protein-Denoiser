import sys
import os
sys.path.insert(1, '../lib/utils/ml_toolbox/src')
sys.path.insert(1, '../lib/utils')

import numpy as np

from proteins import Sample
from Inputs import Read_Input

def add_noise(input_data):

    filename = input_data['map_name'] + '.mrc'
    res      = str(input_data['res'])
    centre   = input_data['centre']
    sigma    = input_data['sigma']
    clip     = input_data['clip']
    seed     = input_data['seed']

    print("Adding noise to density map...")

    # Path to clean map generated from model
    path = os.path.join('maps', res, filename)

    # Object to deal with map data
    s1 = Sample(3.0, path)

    # Save shape of map to variable
    shape = s1.map.shape
    Min = np.min(s1.map)
    Max = np.max(s1.map)

    np.random.seed(seed)
    noise = np.random.normal(centre, sigma, size=shape)

    if clip:
        # Clip the noise
        s1.map = np.clip(s1.map + noise, Min, Max)
    else: 
        # Add noise
        s1.map = s1.map + noise
        # Rescale back to original range
        a = Min
        b = Max
        MIN = np.min(s1.map)
        MAX = np.max(s1.map)
        s1.map = (((b - a) * (s1.map - MIN))/(MAX-MIN)) + a


    save_path = os.path.join('noisy', res,
            f"c_{centre}_s_{sigma}", filename)

    s1.save_map(save_path)

    print("noisy map saved to: {}".format(save_path))


if __name__ == '__main__':
    
    # Read input data
    input_data = Read_Input('inputs.yaml')
    
    add_noise(input_data)
