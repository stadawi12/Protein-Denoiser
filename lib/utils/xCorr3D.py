import numpy as np
from TEMPy.ScoringFunctions import ScoringFunctions
from TEMPy.EMMap import Map
from TEMPy.mapprocess import Filter
from sort import Maps

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

if __name__ == '__main__':

    m1 = Maps('../../data/1.0/')
    m2 = Maps('../../data/2.0/')

    m1.paths = [m1.path_global + m for m in m1.maps]

    m2.paths = [m2.path_global + m for m in m2.maps]

    maps = m1.maps

    sc = ScoringFunctions()
    NO_OF_MAPS = len(m1.maps)

    data = []
    ids = []

    for i in range(NO_OF_MAPS):
        assert m1.maps[i] == m2.maps[i], "Unlucky bro"
        map_id = m1.maps[i][:-4]
        emmap1 = read_mapfile(m1.paths[i])
        emmap2 = read_mapfile(m2.paths[i])
        ccc,overlap = sc.CCC_map(emmap1,emmap2)
        print "map id: {}, ccc: {}, overlap: {}".format(
                map_id,ccc,overlap)
        data.append(ccc)
        ids.append(map_id)

    for i in range(NO_OF_MAPS):
        print(i)
        map_id = maps[i]
        if data[i] <= 0.73:
            m1.move_to_bad(map_id)
            m2.move_to_bad(map_id)



