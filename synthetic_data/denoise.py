import sys
sys.path.insert(1, '../lib')
sys.path.insert(1, '../lib/net')
sys.path.insert(1, '../lib/utils')
sys.path.insert(1, '../lib/utils/ml_toolbox/src')
import os

import unet                     # from lib/net
from Process import Process     # from lib/
from Inputs import Read_Input   # from lib/utils

def denoise(input_data):
    # Read input data
    proc_map_name  = input_data["map_name"] + '.mrc'
    proc_model     = input_data["model_name"]
    epoch          = input_data["epoch"]
    proc_res       = str(input_data["res"])
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
    

if __name__ == '__main__':

    input_data = Read_Input('inputs.yaml')

    denoise(input_data)

