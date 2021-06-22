import numpy as np
from TEMPy.ScoringFunctions import ScoringFunctions
from TEMPy.EMMap import Map
from TEMPy.mapprocess import Filter
import os

mrcfile_import=True

try:
    import mrcfile
except ImportError:
    mrcfile_import = False

print(mrcfile_import)

def read_mapfile(map_path):
    if mrcfile_import:
        mrcobj=mrcfile.open(map_path,mode='r')
        print(np.min(mrcobj.data), np.max(mrcobj.data))
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

clean_map = read_mapfile('maps/3.0/pdb6nyy.mrc')
noisy_map = read_mapfile('noisy/3.0/pdb6nyy.mrc')
denoised_map = read_mapfile('denoised/3.0/e_20_pdb6nyy.mrc')

sc = ScoringFunctions()

ccc_noisy, overlap_noisy = sc.CCC_map(clean_map,noisy_map)

ccc_denoised,overlap_denoised = sc.CCC_map(denoised_map,
        clean_map)
ccc_clean, overlap_clean = sc.CCC_map(clean_map,clean_map)

print(ccc_noisy, overlap_noisy)
print(ccc_denoised, overlap_denoised)
print(ccc_clean,overlap_clean)
