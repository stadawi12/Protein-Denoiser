import sys
sys.path.insert(1, '../lib/utils/ml_toolbox/src')
sys.path.insert(1, '../lib/utils')

import numpy as np
import os

from proteins import Sample
from Inputs import Read_Input


if __name__ == '__main__':

    input_data = Read_Input('inputs.yaml')

    filename = input_data['filename']
    res      = input_data['res']
    centre   = input_data['centre']
    sigma    = input_data['sigma']

    path = os.path.join('maps', res, filename)

    s1 = Sample(3.0, path)

    shape = s1.map.shape
    Min = np.min(s1.map)
    Max = np.max(s1.map)

    noise = np.random.normal(centre, sigma, size=shape)
    s1.map = np.clip(s1.map + noise, Min, Max)
    print(Min, np.min(s1.map))
    print(Max, np.max(s1.map))

    save_path = os.path.join('noisy', res, filename)

    s1.save_map(save_path)
