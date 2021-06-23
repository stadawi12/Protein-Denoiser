import sys
sys.path.insert(1, '../lib/utils/ml_toolbox/src')
sys.path.insert(1, '../lib/utils')

import numpy as np
import os

from proteins import Sample
from Inputs import Read_Input


if __name__ == '__main__':

    print("Adding noise to density map...")

    input_data = Read_Input('inputs.yaml')

    filename = input_data['noise_fname']
    res      = input_data['noise_res']
    centre   = input_data['centre']
    sigma    = input_data['sigma']
    clip     = input_data['clip']

    path = os.path.join('maps', res, filename)

    s1 = Sample(3.0, path)

    shape = s1.map.shape
    Min = np.min(s1.map)
    Max = np.max(s1.map)

    noise = np.random.normal(centre, sigma, size=shape)
    if clip:
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


    save_path = os.path.join('noisy', res, filename)

    s1.save_map(save_path)

    print("noisy maps saved to: ")
    print(save_path)
