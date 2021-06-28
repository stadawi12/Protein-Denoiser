import sys
sys.path.insert(1, '../lib/utils/ml_toolbox/src')
sys.path.insert(1, '../lib/utils')

from proteins import Sample
from Inputs import Read_Input
import numpy as np
import os

def normalise(input_data):

    print("Normalising denoised map ...")

    filename = input_data["map_name"] + '.mrc'
    res      = str(input_data["res"])
    epoch    = input_data["epoch"]
    model    = input_data["model_name"]

    clean_path = os.path.join('maps',res,filename)
    denoised_path = os.path.join('denoised',res,
            model, f"e_{epoch}_{filename}")


    clean    = Sample(3.0, clean_path)
    denoised = Sample(3.0, denoised_path)

    # min max parameters of each map
    a, b         = np.min(clean.map), np.max(clean.map)
    Min_d, Max_d = np.min(denoised.map), np.max(denoised.map)

    denoised.map = (((b-a) * (denoised.map-Min_d)) / \
            (Max_d - Min_d)) + a

    denoised.save_map(denoised_path)

    denoised = Sample(3.0, denoised_path)

    print("denoised map min, max: ") 
    print(np.min(denoised.map), np.max(denoised.map))
    print("clean map min, max: ") 
    print(np.min(clean.map), np.max(clean.map))


if __name__ == '__main__':

    input_data = Read_Input('inputs.yaml')

    normalise(input_data)
