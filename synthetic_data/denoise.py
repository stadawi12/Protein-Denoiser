import sys
sys.path.insert(1, '../lib')
sys.path.insert(1, '../lib/net')
sys.path.insert(1, '../lib/utils')
sys.path.insert(1, '../lib/utils/ml_toolbox/src')
import os

import unet                     # from lib/net
from Process import Process     # from lib/
from Inputs import Read_Input   # from lib/utils

input_data = Read_Input('inputs.yaml')

# Read input data
proc_map_name  = input_data["proc_map_name"]
proc_model     = input_data["proc_model"]
epoch          = input_data["proc_epoch"]
proc_res       = input_data["proc_res"]
denoise_anyway = input_data["denoise_anyway"]

# Check if map already denoised using this model
check_path = os.path.join('denoised', proc_res,
        proc_model, f"e_{epoch}_{proc_map_name}")
path_exists = os.path.exists(check_path)

if not path_exists:
    p1 = Process(unet, input_data, out_path='../out',
        data_path='noisy/3.0')

    p1.process_synthetic()
else:
    print("Map already denoised using this model")

if denoise_anyway:
    print("Denoising anyway")
    p1 = Process(unet, input_data, out_path='../out',
        data_path='noisy/3.0')

    p1.process_synthetic()
    
