import sys
sys.path.insert(1, 'ml_toolbox')
sys.path.insert(1, 'ml_toolbox/src/')

from proteins import Sample
import os, torch
import pandas as pd
from multi_proteins import load_data, preproces_data
import matplotlib.pyplot as plt
import xCorr as cc
from proteins import load_map



class Data:
    
    def __init__(self, path):
        self.path_global = path
        self.ls = os.listdir(self.path_global)
        if '.gitkeep' in self.ls:
            self.ls.remove('.gitkeep')
        self.ls = [float(f) for f in self.ls]
        self.ls.sort()
        self.maps = [os.listdir(self.path_global + 
            str(f)) for f in self.ls]
        self.no_of_batches = len(self.ls)
        self.batch_id = None
        self.tiles = None
        self.get_contents()
        self.batch_sizes = [len(d) for d in self.rest_contents]


    def load_batch(self, i, n=None):
        self.batch_id = i
        self.batch = None
        data = load_data(self.path_global, [self.ls[i]], n)

        data = preproces_data(data,
                norm=True,
                mode='tile',
                cshape=64,
                norm_vox=1.4,
                norm_vox_lim=(1.4,1.4)
                )

        self.batch = data
        self.convert()

    def load_maps(self, i, n=None):
        self.batch_id = i
        self.batch_maps = load_data(self.path_global,
                [self.ls[i]], n)

    def convert(self):
        self.tiles = None
        assert self.batch != None, "Need to load batch"
        tiles_np = [tile for d in self.batch for tile in d.tiles]
        # convert tiles to tensors
        tiles_torch = [torch.tensor(t) for t in tiles_np]
        # unsqueeze so the shape of tiles are [bs,c,h,w,d]
        tiles_uncat = [t.unsqueeze(0).unsqueeze(0) \
                             for t in tiles_torch]
        # concatonate tiles into one array
        self.tiles = torch.cat(tiles_uncat, 0)



    def get_contents(self):
        # obtains the contents of the sub directories of
        # the global directory, stores the in 
        # self.rest_contents
        dir_paths = [self.path_global + str(b) + '/' \
                for b in self.ls]
        dir_contents = [os.listdir(dir_path) 
                for dir_path in dir_paths]
        self.rest_contents = dir_contents



if __name__ == '__main__':
    sys.path.insert(1, '../data/downloads/')
    from sort import Maps

    m1 = Maps('../data/downloads/1.0/')
    m2 = Maps('../data/downloads/2.0/')

    def plot(map1,map2, i):
        fig, ax = plt.subplots(2)
        ax[0].imshow(map1[i,:,:])
        ax[1].imshow(map2[i,:,:])
        return plt.show()

    def select(xCorr_data, ids, threshold):
        """
            Anything below the threshold will be
            ouptut
        """
        bad_xCorrs = []
        bad_ids = []
        for i, xCorr in enumerate(xCorr_data):
            if xCorr < threshold:
                bad_xCorrs.append(xCorr)
                bad_ids.append(ids[i])

        dic = {"bad_xCorrs": bad_xCorrs,
               "bad_ids": bad_ids}

        return dic


    import matplotlib.pyplot as plt
    d1 = Data('../data/downloads/1.0/')
    d2 = Data('../data/downloads/2.0/')
    d3 = Data('../data/downloads/1.5/')
    d3.load_batch(0)
    print(d1.maps)
    print(d2.maps)
    # data = []
    # ids = []
    # for i in range(len(d1.ls)):
    #     d1.load_maps(i)
    #     d2.load_maps(i)
    #     print("Calculating cross-correlations...")
    #     for j in range(len(d1.batch_maps)):
    #         map1 = d1.batch_maps[j].map
    #         map2 = d2.batch_maps[j].map
    #         width = map1.shape[0]
    #         single_map_data = []
    #         ids.append(d1.batch_maps[j].id)
    #         for k in range(width):
    #             xCorr = cc.cross_correlate(map1[k,:,:],
    #                                        map2[k,:,:])[0][0]
    #             if str(xCorr) != 'nan':
    #                 single_map_data.append(xCorr)


    #         avg = sum(single_map_data)/len(single_map_data)
    #         data.append(avg)
    # print("Finished!")
    # print(f"Output: 'data' and 'ids'")
    # print(f"Use the select(data, ids, threshold) function to choose the pairs of maps you would like to get rid of.")



            
