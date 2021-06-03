# Standard library imports
import sys
sys.path.insert(1, 'ml_toolbox/src')
import torch
import os

# Related third party imports (Jola's toolbox)
from multi_proteins import load_data, preproces_data

# Local application/library specific imports
from xCorr import cross_correlate as cc
from sort import Maps


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
        self.batch_maps = None


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


def mean_xCorr3D(map1, map2):
    """Divides half-maps into slices and calculates the 
    cross correlation between two slices   and  returns 
    the mean cross-correlation value of all slices.

    Parameters
    ----------
    map1 : numpy.ndarray
        first 3D half-map 
    map2 : numpy.ndarray
        second 3D half-map
        
    Returns
    -------
    mean_xCorr_value : 
        mean cross-correlation value between two half-maps
    """

    # Number of 2D layers
    NO_OF_LAYERS = map1.shape[0]

    # Collect xCorr of each layer
    xCorrs = []

    # Calculate cross correlation for each layer
    for i in range(NO_OF_LAYERS):

        # Calculate CC for a pair of 2D layers
        xCorr = cc(map1[i,:,:], map2[i,:,:])[0][0]

        # Some calculations return nan, ignore them
        if str(xCorr) != 'nan':
            single_map_data.append(xCorr)

        # Append cc value to bin
        xCorrs.append(xCorr)

    # Calculate mean of all xCorrs
    sum_of_xCorrs = sum(xCorrs)
    mean_xCorr_value = sum_of_xCorrs / len(xCorrs)

    return mean_xCorr_value


def calculate_batch_xCorrs(batch_maps1, batch_maps2):
    """Wrapper over mean_xCorr3D, calculates   mean
    cross-correlations of all maps inside the batch

    Parameters
    ----------
    data1 : list
        Data(path).batch_maps: list of all maps inside batch
        1, each map is an object belonging to the Sample 
        class.
    data2 : list
        Data(path).batch_maps: list of all maps inside batch
        2, each map is an object belonging to the Sample 
        class.

    Returns
    -------
    xCorr : dict
        Dictionary containing batch data
        keys: "ids" and "xcorrs"

    """

    # Creating two bins for holding data
    BIN_IDS = []        # holds ID of each map in batch
    BIN_XCORRS = []     # holds xCorr value of each map

    # Obtain number of maps inside batch
    NO_OF_MAPS = len(batch_maps1)
    
    # Assert that 

    # Calculate mean_xCorr3D for all maps in batch
    for i in range(NO_OF_MAPS):

        # Obtain ID of i'th map
        ID = batch_maps1.id

        # Append ID to BIN_IDS
        BIN_IDS.append(ID)

        # Select i'th maps from batches
        map1 = batch_maps1[i].map
        map2 = batch_maps2[i].map

        # Calculate mean_xCorr of maps
        mean_xCorr_value = mean_xCorr3D(map1, map2)

        # Append the mean_xCorr value to the bin
        BIN_XCORRS.append(mean_xCorr_value)

    # Move binned data to a dictionary
    xCorr = {"xcorrs":BIN_XCORRS,
             "ids"   :BIN_IDS}

    # Return dictionary containing data
    return xCorr

# def calculate_xCorr_all(data1, data2):
#     """Wrapper over calculate_batch_xCorrs that calculates
#     and returns all mean_xCorr values for all batches 
#     inside a directory 
# 
#     Parameters
#     ----------
#     data1 : __main__.Data
#         object of Data class, deals with a directory of
#         batches, contains all batches of half-maps 1
#     data2 : __main__.Data
#         object of Data class, deals with a directory of
#         batches, contains all batches of half-maps 2
# 
#     Returns
#     -------
# 
#     """
#     
#     NO_OF_BATCHES = data1.


def cross_correlate_batch2D(batch_1, batch_2):
    """
        i: batch index
    """
    # Bins which will hold cross-correlation data and ids
    data = []   # Cross-correlation outputs
    ids = []    # Corresponding map ids

    # Print communicate
    print("Calculating cross-correlations...")

    # For every map in batch..
    for j in range(len(batch_1.batch_maps)):

        # Select j'th pair of maps to cross-correlate
        map1 = batch_1.batch_maps[j].map
        map2 = batch_2.batch_maps[j].map

        # Obtain the number of layers for 2D CC
        width = map1.shape[0]

        # Create empty bin to collect CC values
        single_map_data = []

        # Append id of map to ids
        ids.append(batch_1.batch_maps[j].id)

        # For 2D slice of map
        for k in range(width):

            # Calculate CC for a pair of 2D layers
            xCorr = cc(map1[k,:,:], map2[k,:,:])[0][0]

            # Some calculations return nan, ignore them
            if str(xCorr) != 'nan':
                single_map_data.append(xCorr)

        # Calculate average cross-correlation value
        avg = sum(single_map_data)/len(single_map_data)
        
        # Append average CC between all layers to data
        data.append(avg)

    cc_dict = {"data":data,
               "ids" : ids}

    return cc_dict

def cross_correlate_all():

    # Create two objects of the Data class
    d1 = Data('../../data/1.0/')
    d2 = Data('../../data/2.0/')

    # Obtain number of batches
    NOB = d1.number_of_batches

    # Perform cross-correlation for each batch
    for i in range(NOB):

        # Load maps
        d1.load_maps(i)
        d2.load_maps(i)

        # Perform cross-correlation for batch
        cc_dict = cross_correlate_batch2D(d1, d2)

        # Unpack dictionary

    return cc_dict 
    




if __name__ == '__main__':
    d1 = Data('../../data/1.0/')
    d1.load_batch(0)
    d2 = Data('../../data/2.0/')
    d2.load_batch(0)
    #  m1 = Maps('../data/downloads/1.0/')
    #  m2 = Maps('../data/downloads/2.0/')

    #  def plot(map1,map2, i):
    #      fig, ax = plt.subplots(2)
    #      ax[0].imshow(map1[i,:,:])
    #      ax[1].imshow(map2[i,:,:])
    #      return plt.show()

    #  def select(xCorr_data, ids, threshold):
    #      """
    #          Anything below the threshold will be
    #          ouptut
    #      """
    #      bad_xCorrs = []
    #      bad_ids = []
    #      for i, xCorr in enumerate(xCorr_data):
    #          if xCorr < threshold:
    #              bad_xCorrs.append(xCorr)
    #              bad_ids.append(ids[i])

    #      dic = {"bad_xCorrs": bad_xCorrs,
    #             "bad_ids": bad_ids}

    #      return dic


    #  import matplotlib.pyplot as plt
    #  d1 = Data('../data/downloads/1.0/')
    #  d2 = Data('../data/downloads/2.0/')
    #  d3 = Data('../data/downloads/1.5/')
    #  d3.load_batch(0)
    #  print(d1.maps)
    #  print(d2.maps)
    #  data = []
    #  ids = []
    #  for i in range(len(d1.ls)):
    #      d1.load_maps(i)
    #      d2.load_maps(i)
    #      print("Calculating cross-correlations...")
    #      for j in range(len(d1.batch_maps)):
    #          map1 = d1.batch_maps[j].map
    #          map2 = d2.batch_maps[j].map
    #          width = map1.shape[0]
    #          single_map_data = []
    #          ids.append(d1.batch_maps[j].id)
    #          for k in range(width):
    #              xCorr = cc.cross_correlate(map1[k,:,:],
    #                                         map2[k,:,:])[0][0]
    #              if str(xCorr) != 'nan':
    #                  single_map_data.append(xCorr)


    #          avg = sum(single_map_data)/len(single_map_data)
    #          data.append(avg)
    #  print("Finished!")
    #  print(f"Output: 'data' and 'ids'")
    #  print(f"Use the select(data, ids, threshold) function to choose the pairs of maps you would like to get rid of.")
