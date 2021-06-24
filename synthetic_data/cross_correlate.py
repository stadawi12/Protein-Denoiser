import sys
sys.path.insert(1, '../lib/utils')

import numpy as np
from TEMPy.ScoringFunctions import ScoringFunctions
from TEMPy.EMMap import Map
from TEMPy.mapprocess import Filter
import os
import pandas as pd
import csv

from Inputs import Read_Input

input_data = Read_Input('inputs.yaml')

filename = input_data["cc_fname"]
res      = input_data["cc_res"]
epoch    = input_data["cc_epoch"]
model    = input_data["cc_model"]
append   = input_data["append"]
clip     = input_data["clip"]

mrcfile_import=True

try:
    import mrcfile
except ImportError:
    mrcfile_import = False

def read_mapfile(map_path):
    if mrcfile_import:
        mrcobj=mrcfile.open(map_path,mode='r')
        print("min, max: {}, {}"
        .format(np.min(mrcobj.data), np.max(mrcobj.data)))
        if np.any(np.isnan(mrcobj.data)):
            sys.exit('Map has NaN values: {}'.format(map_path))
        emmap = Filter(mrcobj)
        emmap.set_apix_tempy()
        emmap.fix_origin()
    else:
        mrcobj=MapParser.readMRC(map_path)
        emmap = Filter(mrcobj)
        emmap.fix_origin()
    return emmap

clean_path = os.path.join('maps',res,filename)
noisy_path = os.path.join('noisy',res,filename)
denoised_path = os.path.join('denoised',res,
        model,
        "e_{}_{}".format(epoch, filename))

clean_map = read_mapfile(clean_path)
noisy_map = read_mapfile(noisy_path)
denoised_map = read_mapfile(denoised_path)

sc = ScoringFunctions()

ccc_noisy, overlap_noisy = sc.CCC_map(clean_map,noisy_map)

ccc_denoised,overlap_denoised = sc.CCC_map(denoised_map,
        clean_map)
ccc_clean, overlap_clean = sc.CCC_map(clean_map,clean_map)

print("Noise vs Clean: {}, {}".format(ccc_noisy, 
    overlap_noisy))
print("Denoised vs Clean: {}, {}".format(ccc_denoised, 
    overlap_denoised))
print("Clean vs Clean: {}, {}".format(ccc_clean,
    overlap_clean))

if append:
    centre = input_data["centre"]
    sigma  = input_data["sigma"]
    if clip:
        clip='yes'
    else:
        clip='no'
    fields = [filename,res,model,epoch,clip,centre,sigma,ccc_noisy,ccc_denoised]
    with open(r'data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    data = pd.read_csv('data.csv')
    df = pd.DataFrame(data)
    print(df)
