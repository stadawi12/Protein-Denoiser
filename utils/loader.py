import sys
sys.path.insert(1, 'ml_toolbox/')
sys.path.insert(1, 'utils/ml_toolbox/')
sys.path.insert(1, 'utils/ml_toolbox/src/')

def data_loader(global_path, half=[1.0], n=None):
    """
        Loads data and preprocesses it.  It loads data
        from data/downloads directory.   Inside of the 
        download/ dir, there are two directories, 1.0/
        and 2.0/, each containing half maps.   The res
        parameter should either be 1.0   or   2.0   to 
        select the necessary half maps.  This function
        then loads all half maps from that dir.    and 
        tiles them, into tiles of size 64x64x64.
    """

    from ml_toolbox.multi_proteins import \
    load_data

    data = load_data(global_path, half, n)

    # ensure self.id shows full id
    path_file = global_path + str(half[0]) + '/'
    for d in data:
        d.id = d.path_map[len(path_file):-4]
    return data

def data_preprocess(data):
    from ml_toolbox.multi_proteins import \
    preproces_data, query_data

    data = preproces_data(data, norm=True, mode='tile', cshape=64)
    # query_data(data)
    return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    global_path = '../data/downloads/'
    local = [1.0]
    n = None
    data = data_loader(global_path, local, n)
    data = data_preprocess(data)
    maps = [d.map for d in data]
    number_of_maps = len(maps)
    minMax = np.zeros((number_of_maps, 3))
    for i in range(number_of_maps):
        minMax[i,0] = np.min(maps[i])
        minMax[i,1] = np.max(maps[i])
        minMax[i,2] = np.mean(maps[i])
    print(minMax)
    data[1].pred = data[1].tiles
    data[1].recompose()
    data[1].map = data[1].pred
    data[1].save_map('../data/downloads/1.0norm/10161.map')
    # tiles = data[0].tiles
    # fig, ax = plt.subplots(3)
    # ax[0].imshow(tiles[0][8])
    # ax[1].imshow(tiles[1][8])
