import sys
sys.path.insert(1, '../lib')
sys.path.insert(1, '../lib/net')
sys.path.insert(1, '../lib/utils')
sys.path.insert(1, '../lib/utils/ml_toolbox/src')

import unet                     # from lib/net
from Process import Process     # from lib/
from Inputs import Read_Input   # from lib/utils

input_data = Read_Input('../inputs.yaml')

p1 = Process(unet, input_data, out_path='../out',
    data_path='noisy/3.0')

p1.process(norm=True)
