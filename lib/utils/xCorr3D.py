import numpy as np
from TEMPy.ScoringFunctions import ScoringFunctions
from TEMPy.EMMap import Map
from TEMPy.mapprocess import Filter

mrcfile_import=True

try:
    import mrcfile
except ImportError:
    mrcfile_import = False

def read_mapfile(map_path):
    if mrcfile_import:
        mrcobj=mrcfile.open(map_path,mode='r')
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

map_path1 = '../../data/1.5/1.0/0552.map'
map_path2 = '../../data/1.5/1.0/0552.map'

sc = ScoringFunctions()
emmap1 = read_mapfile(map_path1)
emmap2 = read_mapfile(map_path2)
ccc,overlap = sc.CCC_map(emmap1,emmap2)
