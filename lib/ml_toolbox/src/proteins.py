# std imports
import gc
import os
import warnings

# specific imports
from collections import defaultdict
from enum import Enum

# domain-specific imports
from Bio import PDB
import mrcfile as mrc
import numpy as np
from scipy.ndimage import zoom

# project imports
from src.utils import progress, grid_to_real, divx, normalise
from src.viewer import plot_3d

warnings.filterwarnings("ignore")


class Atom:
    def __init__(self, mass, waals, coval, num):
        self.mass = mass
        self.waals = waals
        self.coval = coval
        self.num = num


def residue(name):
    if name not in RESIDUES:
        return 21   # other label
    else:
        return RESIDUES.index(name)+1


ATOMS = {'FE': Atom(55.845, 2.15, 1.25, 26),
         'CA': Atom(40.078, 2.31, 1.74, 20),
         'CL': Atom(35.453, 1.75, 0.99, 17),
         'C': Atom(12.0107, 1.7, 0.77, 6),
         'N': Atom(14.0067, 1.55, 0.75, 7),
         'O': Atom(15.9994, 1.52, 0.73, 8),
         'S': Atom(32.065, 1.8, 1.02, 16),
         'MG': Atom(24.305, 1.73, 1.3, 12),
         'ZN': Atom(65.39, 1.39, 1.31, 30),
         'I': Atom(126.9045, 1.98, 1.33, 53),
         'P': Atom(30.9738, 1.8, 1.06, 15),
         'NA': Atom(22.9897, 2.27, 1.54, 11),
         'F': Atom(18.9984, 1.47, 0.71, 9),
         'K': Atom(39.0984, 2.75, 1.96, 19),
         # 'H': Atom(1.0026, 1.2, 0.38, 1),
         'SE': Atom(78.96, 1.9, 1.16, 34),
         'MN': Atom(54.938, 2.07, 1.39, 35),
         'NI': Atom(58.6934, 1.63, 1.21, 28),
         'AL': Atom(26.9815, 1.84, 1.18, 13),
         'PB': Atom(207.2, 2.02, 1.47, 82),
         'CU': Atom(63.546, 1.4, 1.38, 29),
         # '': Atom(0, 0, 0, 0)
         }

RESIDUES = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG',
            'SER', 'THR', 'VAL', 'TRP', 'TYR']  # , 'HEM', 'HOH'] # OTHER category (21) is included for non aminoacids
BACKBONE = ["N", "CA", "C", "O"]  # , "CB"]


class Sample:
    """
    Set of properties and methods associated with a single sample. Data can be extracted as follows:

        x = [d.map for d in data]
        y = [d.lab for d in data]
        head = [d.header_map for d in data]
        train = [d for d in data if d.MODE == 0]

    NOTE: In later developments self.tiles can extist which is a list of
    cubes in the image rather than whole image (if tiles preprocessing has been applied).
    To extract them rather than full map run:

        if data[0].has_tiles:
            # extends the list instead of appending tiles
            x = [tile for d in data for tile in d.tiles]
            y = [tile for d in data for tile in d.ltiles]

    * Subcubes are treated as members of the sample to prevent them from
    being separated across training and testing sets and thus biasing the model.

    If a certain resolution range is desired it can be obtained as follows:

        x = [d.map for d in data if d.res == <MY_RES>]

    For a list of resolutions:

        x = [d.map for d in data if d.res in <LIST_RES>]

    To save data:

        for d in data:
            d.save_map()
            d.save_lab()
            d.save_pred()

    To set training:

        test, train = cv.split_test_train(len(data))
        [d.set_train() for d in data[train]
        d_train = [d.map for d in data if d.mode == 0]

    """

    class Mode(Enum):
        TRAIN = 0
        VAL = 1
        TEST = 2

    class Header:
        def __init__(self, o, s, c):
            self.orig = o
            self.samp = s
            self.cell = c

    def __init__(self, res, xpath, ypath=None, wpath=None, amino_groups=False):

        self.res = res
        self.id = xpath.split('/')[-1].split('.')[0]

        self.base_path = xpath.split('/')
        self.path_map = xpath
        self.path_lab = ypath
        self.path_pred = '/'.join(self.base_path[:-1]) + 'preds/' + self.base_path[-1]
        self.path_diff = '/'.join(self.base_path[:-1]) + 'diffs/' + self.base_path[-1]
        self.path_weight = wpath

        self.mode = self.Mode.TRAIN if ypath is not None else self.Mode.TEST

        try:
            ximg, xorig, xsamp, xcell = load_map(xpath)
            ximg[ximg == -0.] = 0.
            if ypath is not None:
                yimg, yorig, ysamp, ycell = load_map(ypath)
                yimg = yimg.astype(np.int32)
            if wpath is not None:
                wimg, worig, wsamp, wcell = load_map(wpath)
        except RuntimeError:        # previous error message was not informative enough
            raise RuntimeError("Either map or label is corupted at sample {}\n    {}\n    {}",
                               self.id, self.path_map, self.path_lab)

        if amino_groups:
            yimg = amino6(yimg)

        self.map = ximg
        self.lab = None if ypath is None else yimg
        self.pred = None
        self.diff = None
        self.weight = None if wpath is None else wimg
        self.map_rec = None

        self.has_tiles = False  # this attr is used if data is decomposed into tiles - tiles flag
        self.tiles = None       # this attr is used if data is decomposed into tiles - list of tiles
        self.ltiles = None      # this attr is used if data is decomposed into tiles - list of label tiles
        self.wtiles = None      # this attr is used if data is decomposed into tiles - list of weight tiles
        self.orig_shape = None  # this attr is used if data is decomposed into tiles - shape prior decomposition
        self.margin = None      # this attr is used if data is decomposed into tiles - overlap margin
        self.step = None        # this attr is used if data is decomposed into tiles - valid data extent
        self.no_tiles = None    # this attr is used if data is decomposed into tiles - original number of tiles
        self.tile_indices = None  # this attr is used if data is decomposed into tiles - indices for threshold param
        self.crop_tile = False  # this attr is used if data is decomposed into tiles - cropping to single tile if no
                                #                                                      tiles with threshold found

        self.header_map = self.Header(xorig, xsamp, xcell)
        if ypath is not None:
            self.header_lab = self.Header(yorig, ysamp, ycell)
        self.header_pred = self.Header(xorig, xsamp, xcell)
        self.header_diff = self.Header(xorig, xsamp, xcell)
        if wpath:
            self.header_weight = self.Header(worig, wsamp, wcell)
        # TODO only need one header

    def set_train(self):
        self.mode = self.Mode.TRAIN

    def set_val(self):
        self.mode = self.Mode.VAL

    def set_test(self):
        self.mode = self.Mode.TEST

    def make_diff(self):
        if self.pred is None:
            raise RuntimeError("Cannot calculate difference map, predictions don't exist. Please generate predictions"
                               "first")
        self.diff = []
        for i in np.unique(self.pred)[1:]:
            # lab = self.lab[self.lab == i]
            lab = np.copy(self.lab)
            lab[lab == i] = 0
            pred = np.copy(self.pred)
            pred[pred == i] = 0
            self.diff.append(lab - pred)

    def save_map(self, path=None, map_rec=False):
        if map_rec:
            map = self.map_rec
        else:
            map = self.map
        if path is None:
            path = '/'.join(self.base_path[:-1]) + 'maps/' + self.base_path[-1]
            # self.path_map[:-9] + 'maps' + self.path_map[-9:]
        gpath = path.split('/')
        if len(gpath) > 1:
            if not os.path.exists('/'.join(gpath[:-1])):
                os.mkdir('/'.join(gpath[:-1]))
        save_map(map, self.header_map.orig, self.header_map.cell, path=path)

    def save_lab(self, path=None):
        if self.lab is None:
            raise RuntimeError("Cannot save the labels, they don't exist (it looks like a test sample).")
        if path is None:
            path = '/'.join(self.base_path[:-1]) + 'labs/' + self.base_path[-1]
            # self.path_map[:-9] + 'labs' + self.path_map[-9:]
        gpath = path.split('/')
        if len(gpath) > 1:
            if not os.path.exists('/'.join(gpath[:-1])):
                os.mkdir('/'.join(gpath[:-1]))
        save_map(self.lab, self.header_lab.orig, self.header_lab.cell, path=path)

    def save_pred(self, path=None):
        if self.pred is None:
            raise RuntimeError("Cannot save the predictions, they don't exist. Please generate predictions first.")
        if path is None:
            path = self.path_pred  # self.path_map[:-9] + 'preds' + self.path_map[-9:]
        gpath = path.split('/')
        if len(gpath) > 1:
            if not os.path.exists('/'.join(gpath[:-1])):
                os.mkdir('/'.join(gpath[:-1]))
        if self.has_tiles:
            self.recompose()
        amino_map(self.pred, self.header_pred.orig, self.header_pred.cell, path=path, overwrite=True)
        # save_map(self.pred, self.header_pred.orig, self.header_pred.cell, path=path)

    def save_diff(self, path=None):
        if self.diff is None:
            try:
                self.make_diff()
            except RuntimeError:
                raise RuntimeError("Cannot save difference as predictions don't yet exist. Please generate predictions"
                                   " first.")
        if path is None:
            path = self.path_diff  # self.path_map[:-9] + 'evals' + self.path_map[-9:]
        gpath = path.split('/')
        if len(gpath) > 1:
            if not os.path.exists('/'.join(gpath[:-1])):
                os.mkdir('/'.join(gpath[:-1]))
        for i, d in enumerate(self.diff):
            save_map(d, self.header_diff.orig, self.header_diff.cell, path=path[:-4]+"_"+str(i+1)+path[-4:])

    def decompose(self, cshape=64, margin=8, norm=True, norm_vox=0.7, norm_vox_lim=(0.6, 1.0), crop=True, step=None,
                  background_limit=None, threshold=None, background_label=None):
        """
        Decompose a map into a list of cubes / 3D tiles.

        :param cshape: (int) desired isotropic cube / 3D tile shape
        :param margin: (int) margin of overlap between the tiles (recommended pool^depth)
        :param norm: (bool) normalises data between 0 and 1 if True
        :param norm_vox (float) desired value for voxel size normalisation
        :param norm_vox_lim (tuple(float, float)) if not None will rescale all maps with voxels outside of range
                <norm_vox[0], norm_vox[1]> to 1.0
        :param crop: (bool) crops map to the extent of the data if labels are present
        :param step: (int) use only with threshold, step between the start of each tile, for dense it's cshape-2*margin
        :param background_limit: (float) sets limit for proportion of background in accepted extracted tiles,
                                 default None
        :param threshold: (float) if background_limit used, sets a threshold for a boolean map to extract tiles across
                          (values below are bg), default None; if not passed, executing background_label by default
        :param background_label: (float) if background_limit used, uses passed background value for a boolean map to
                                 extract tiles across
        """
        _debug = False

        if threshold:
            # mask for smaller maps do not work well for cshape >= 32
            if min(self.map.shape) / float(cshape) < 1.5:
                background_limit = 0.7
                step = 2
        
        self.has_tiles = True
        self.margin = margin    # recommended pool^depth (e.g. with 2x2 pooling and 3 depth -> 2^3)
        self.step = (cshape - (2 * margin)) if step is None else step  # 2*margin overlap
        if 2*margin >= cshape:
            raise RuntimeError("Margin of tile overlap must be smaller than half of the tile size.")

        # A) normalise voxel size
        if norm_vox is not None:
            if str(type(norm_vox_lim)) != "<class 'tuple'>":
                raise RuntimeError("Parameter 'norm_vox' must be a tuple containing the lowest and "
                                   "highest allowed voxel size.")
            self.map, self.header_map.samp = data_vox_norm(self.map, self.header_map.samp, self.header_map.cell,
                                                           vox=norm_vox,
                                                           vox_min=norm_vox_lim[0], vox_max=norm_vox_lim[1])
            if self.lab is not None:
                self.lab, self.header_lab.samp = data_vox_norm(self.lab, self.header_lab.samp, self.header_lab.cell,
                                                               vox=norm_vox,
                                                               vox_min=norm_vox_lim[0], vox_max=norm_vox_lim[1])
            if self.weight is not None:
                self.weight, self.header_weight.samp = data_vox_norm(self.weight, self.header_weight.samp,
                                                                     self.header_weight.cell,
                                                                     vox=norm_vox,
                                                                     vox_min=norm_vox_lim[0], vox_max=norm_vox_lim[1])
        # B) normalise voxel intensities
        if norm:
            self.map = normalise(self.map)

        if _debug:
            import matplotlib.pyplot as plt
            print("0:", self.map.shape)
            plt.imshow(self.map[:, :, self.map.shape[-1]//2])
            plt.show()

        if crop:
            # 0) CROP DATA TO DATA EXTENT IF LABS (MASKS) AVAILABLE (and updates global header info)
            if self.lab is not None:
                dshape, mid = borders(self.lab, cshape=cshape)  # dshape in z,y,x
                self.map, self.header_map.orig, self.header_map.samp, self.header_map.cell = \
                    data_crop(dshape, self.map, orig=self.header_map.orig, samp=self.header_map.samp,
                              cell=self.header_map.cell, mid=mid, move_o=True)
                self.lab, self.header_lab.orig, self.header_lab.samp, self.header_lab.cell = \
                    data_crop(dshape, self.lab, orig=self.header_lab.orig, samp=self.header_lab.samp,
                              cell=self.header_lab.cell, mid=mid, move_o=True)
                if self.weight is not None:
                    self.weight, self.header_weight.orig, self.header_weight.samp, self.header_weight.cell = \
                        data_crop(dshape, self.weight, orig=self.header_weight.orig, samp=self.header_weight.samp,
                                  cell=self.header_weight.cell, mid=mid, move_o=True)
                self.header_pred = self.header_lab
        self.orig_shape = self.map.shape  # updating global map shape after removing all the background

        if _debug:
            print("1:", self.map.shape)
            plt.imshow(self.map[:, :, self.map.shape[-1]//2])
            plt.show()

        # 1) PAD WITH ZEROS TO USE WHOLE IMAGE DURING CONVOLUTIONS (EQUIVALLENT OF PADDING='SAME' IN TF)
        # 2) PAD WITH MULTIPLES OF STEP TO MAKE SURE ALL DATA IS ANALYSED, BUT KEEP THE DIFF
        # sh = [self.map.shape[0] + (2 * margin), self.map.shape[1] + (2 * margin), self.map.shape[2] + (2 * margin)]
        diff = [divx(s, d=self.step) - s for s in self.map.shape]
        sh = [divx(s, d=self.step) + (2*margin) for s in self.map.shape]

        if _debug:
            print("2:", sh)

        dat = np.zeros((sh[0], sh[1], sh[2]), dtype=np.float32)
        dat[margin:-margin - diff[0], margin:-margin - diff[1], margin:-margin - diff[2]] = self.map
        if self.lab is not None:
            lab = np.zeros((sh[0], sh[1], sh[2]), dtype=np.int32)
            lab[margin:-margin - diff[0], margin:-margin - diff[1], margin:-margin - diff[2]] = self.lab
        if self.weight is not None:
            weight = np.zeros((sh[0], sh[1], sh[2]), dtype=np.float32)
            weight[margin:-margin - diff[0], margin:-margin - diff[1], margin:-margin - diff[2]] = self.weight

        if _debug:
            print("3:", dat.shape)
            plt.imshow(dat[:, :, dat.shape[-1]//2])
            plt.show()

        # 3) EXTRACT TILES
        self.tiles = []
        if background_limit:
            self.tile_indices = []
        if self.lab is not None:
            self.ltiles = []
        if self.weight is not None:
            self.wtiles = []
        for i in range(0, dat.shape[0], self.step)[:-1]:
            for j in range(0, dat.shape[1], self.step)[:-1]:
                for k in range(0, dat.shape[2], self.step)[:-1]:
                    # if i+self.step < self.margin or j+self.step < self.margin \
                    #     or k+self.step < self.margin: continue   # skip windows at edges for smaller steps
                    # if i+cshape > dat.shape[0] or j+cshape > dat.shape[1] \
                    #     or k+cshape > dat.shape[2]: continue   # skip windows beyond edge of the box
                    sh = (slice(i, i + cshape), slice(j, j + cshape), slice(k, k + cshape))
                    if dat[sh[0], sh[1], sh[2]].size != cshape**3: continue  # skip smaller tiles at edge
                    if background_limit is not None:
                        if background_label is not None and threshold is not None:
                            print("\n\nWARNING: Both 'background_label' and 'threshold' parameters were used. Use"
                                  " 'threshold' to establish the threshold point for the tiles. Use"
                                  " 'background_label' to pass the value in labels that thresholds the tiles.\n"
                                  "Parameter 'background_label' is prioritised.\n")
                        if background_label is not None:
                            if self.lab is None:
                                raise RuntimeError("Parameter 'background_label' can only be used if labels (ground"
                                                   " truth) exists for the data.")
                            window_array = lab[sh[0], sh[1], sh[2]]
                            try:
                                if np.sum(window_array == background_label)/float(window_array.size) > \
                                        background_limit: continue
                            except (TypeError, ValueError): pass
                            self.tile_indices.append((i, j, k))
                        elif threshold is not None:
                            window_array = dat[sh[0], sh[1], sh[2]]
                            if np.sum(window_array <= threshold)/float(window_array.size) \
                                > background_limit: continue
                            self.tile_indices.append((i, j, k))   # save selected tile indices
                        else:
                            print("\n\nWARNING: When using 'background_limit' param, you must provide one of "
                                  "'threshold' or 'background_limit' parameter (if labels are available). There "
                                  "was no threshold passed and there are no labels available, all tiles will be used.")
                    self.tiles.append(dat[sh[0], sh[1], sh[2]].copy())
                    if self.lab is not None:
                        self.ltiles.append(lab[sh[0], sh[1], sh[2]].copy())
                    if self.weight is not None:
                        self.wtiles.append(weight[sh[0], sh[1], sh[2]].copy())
        
        # crop if no tiles with required background limit is found
        if len(self.tiles) == 0:
            print("\nWARNING: No {} tiles met the criteria, cropping to center tile instead.".format(self.id))
            self.crop_tile = True
            dat, (z, y, x) = data_crop((cshape, cshape, cshape), self.map, ind=True)  # will pad if smaller
            self.tiles.append(dat)
            if threshold or background_limit:
                self.tile_indices.append((z, y, x))
            if self.lab is not None:
                self.ltiles.append(data_crop((cshape, cshape, cshape), self.lab))
            if self.weight is not None:
                self.wtiles.append(data_crop((cshape, cshape, cshape), self.weight))

        # remove redundant variables for memory saving
        del dat
        if self.lab is not None: del lab
        if self.weight is not None: del weight
        gc.collect()
        self.no_tiles = len(self.tiles)
        
        if _debug:
            print("Number of tiles:", len(self.tiles))
            if threshold or background_limit:
                print("Number of indices:", len(self.tile_indices))

        if _debug:
            self.recompose(map=True)
            print("4:", self.map_rec.shape, self.orig_shape)
            plt.imshow(self.map_rec[:, :, self.map_rec.shape[-1]//2])
            plt.figure()
            plt.imshow(self.map[:, :, self.map.shape[-1]//2])
            plt.show()
            if self.map_rec.shape != self.orig_shape:
                raise RuntimeError("The recomposed shape {} is not the same as the original shape {}".format(
                    self.map_rec.shape, self.orig_shape
                ))

    def recompose(self, map=False):
        rec = self.pred if not map else self.tiles
        if len(np.asarray(rec).shape) <= 3:
            print("\nWARNING: The sample has already been recomposed, skipping...")
            return
        # only recomposing predictions, as data and lab are retained anyway after decompose, unless map=True
        if rec is None:
            raise RuntimeError("Cannot recompose predictions, no predictions found. Please generate predictions first.")
        if self.margin is None or self.orig_shape is None:
            raise RuntimeError("Cannot recompose the sample {} {} as data was not decomposed yet.".format(self.id,
                                                                                                          self.res))
        if self.no_tiles is None or (len(rec) != self.no_tiles):
            raise RuntimeError("The number of tile predictions %d is not the same as the original number of tiles %d."
                               % (len(rec), self.no_tiles))

        dt = np.int32 if not map else np.float32
        dat = np.zeros(self.orig_shape, dtype=dt)   # need a float in case recomposing real map (not labels)
        t_id = 0
        if not self.crop_tile:
            for i in range(0, dat.shape[0], self.step):  # [:-1] # self.padded_shape[0]):
                for j in range(0, dat.shape[1], self.step):  # [:-1] # self.padded_shape[1]):
                    for k in range(0, dat.shape[2], self.step):  # [:-1] # self.padded_shape[2]):
                        if self.tile_indices:  # when threshold/mask was applied
                            if not (i, j, k) in self.tile_indices: continue
                        pt = rec[t_id][self.margin:-self.margin, self.margin:-self.margin, self.margin:-self.margin]
                        x = i + self.step if i + self.step <= dat.shape[0] else dat.shape[0]
                        y = j + self.step if j + self.step <= dat.shape[1] else dat.shape[1]
                        z = k + self.step if k + self.step <= dat.shape[2] else dat.shape[2]
                        try:
                            dat[i:x, j:y, k:z] = pt[:x - i, :y - j, :z - k]
                        except ValueError:
                            raise RuntimeError("Error re-stitching data.")
                        t_id += 1
        else:
            pt = rec[0][self.margin:-self.margin, self.margin:-self.margin, self.margin:-self.margin]
            i, j, k = self.tile_indices[0]
            i, j, k = (i + self.margin, j + self.margin, k + self.margin)
            if pt.shape > dat.shape:
                diff = (max(0, pt.shape[0] - dat.shape[0]),
                        max(0, pt.shape[1] - dat.shape[1]),
                        max(0, pt.shape[0] - dat.shape[2]))
                pt = pt[diff[0] // 2 : diff[0] // 2 + dat.shape[0],
                        diff[1] // 2 : diff[1] // 2 + dat.shape[1],
                        diff[2] // 2 : diff[2] // 2 + dat.shape[2]]
            x = i + self.step if i + self.step <= dat.shape[0] else dat.shape[0]
            y = j + self.step if j + self.step <= dat.shape[1] else dat.shape[1]
            z = k + self.step if k + self.step <= dat.shape[2] else dat.shape[2]

            dat[i:x, j:y, k:z] = pt[:x - i, :y - j, :z - k]
        if map:
            self.map_rec = dat
        else:
            self.pred = dat

        # remove redundant variables for memory saving
        del dat
        gc.collect()


    def plot_map(self):
        plot_3d(self.map, 'mlab')

    def plot_lab(self):
        plot_3d(self.lab, 'mlab')

    def plot_pred(self):
        if self.pred is None:
            raise RuntimeError("Cannot plot the predictions, they don't exist. Please generate predictions first.")
        plot_3d(self.pred, 'mlab')

    def plot_data(self):
        self.plot_map()
        self.plot_lab()


def gauss3d(x, y, z, x0, y0, z0, sig, scale=1):
    """ Gaussian function with normalised AUC."""
    return scale * 1/(np.power(2*np.pi, 1.5)*sig**3) * np.exp(-((x-x0)**2+(y-y0)**2+(z-z0)**2)/(2*sig**2))


def load_map(path, normalise=False, verbose=False):

    """
    Load .mrc or .map 3D volumetric electron density map.
    (or labelled density map - will need normalise=False)

    :param path: (str) path to file
    :param normalise: (bool) if True normalises data to 0-1, default True
    :param verbose: (bool) if True printing map information
    :return: img (np.ndarray[float]) 3D volume intensities
             orig (np.recarray[float]) origin coordinates
             sample (tuple(int, int, int)) sampling rate
             cella (np.recarray[float]) size of the sampled cell
    """
    f = mrc.open(path, mode='r+')
    img = f.data
    header = f.header
    f.close()

    orig = header.origin                                # origin of cell
    cella = header.cella                                # size of cell in angstroms
    sample = (header.mx, header.my, header.mz)      # sampling for each axis

    # x, y, z = np.mgrid[0:img.shape[0], 0:img.shape[1], 0:img.shape[2]]
    # x = (x * k) + orig.x  # + k/2.0
    # y = (y * k) + orig.y  # + k/2.0
    # z = (z * k) + orig.z  # + k/2.0

    if normalise:
        # Normalise to 0-1
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

    if verbose:
        print("ORIGIN:", orig)
        print("CELLA:", cella)
        print("SAMPLE:", sample)
        print("AXIS:", header.mapc, header.mapr, header.maps)
        print("\n\n")

    return img, orig, sample, cella


def load_model(path, verbose=False):
    """
    Load .cif or .pdb atomic model file.

    :param path: (str) path to file
    :param verbose: (bool) if True printing map information
    :return: x, y, z (np.ndarray[int]) list of atomic coordinates
             elements (np.ndarray[str]) list of atomic element names
             residues (np.ndarray[int]) list of residues the atoms belong to
             backbone (np.ndarray[bool]) list of backbone inclusion (True for backbone, False for sidechain)
             occupancy (np.ndarray[float]) list of atom occupancies
    """

    coords = []
    elements = []
    residues = []
    backbone = []
    occupancy = []

    parser = None
    if path[-4:] == str(".pdb"):
        parser = PDB.PDBParser()
    elif path[-4:] == str(".cif"):
        parser = PDB.MMCIFParser()

    if parser is None:
        raise RuntimeError("%s is not a supported file type. The supported types are .pdb and .cif" % path)

    structure = parser.get_structure("structure", path)

    for i, a in enumerate(structure.get_atoms()):
        # if a.parent.resname not in RESIDUES:
        #     #print(a.element, a.id, a.parent.resname)
        #     continue
        # if a.element == "H":
        #     continue
        if a.name in BACKBONE and a.parent.resname in RESIDUES:
            bblab = 1
        elif a.name not in BACKBONE and a.parent.resname in RESIDUES:
            bblab = 2
        else:
            bblab = 3  # other
        residues.append(RESIDUES.index(a.parent.resname) + 1 if a.parent.resname in RESIDUES else 21)  # other
        coords.append(a.coord)
        elements.append(a.element)
        occupancy.append(a.occupancy)
        backbone.append(bblab)

    coords = np.asarray(coords)
    x, y, z = coords[:, 2], coords[:, 1], coords[:, 0]
    # x = x / 1.049 + 1.049 / 2.0
    # y = y / 1.049 + 1.049 / 2.0
    # z = z / 1.049 + 1.049 / 2.0

    if verbose:
        print("MODEL:", path.split('/')[-1][:-4])
        print("ATOMS:", np.unique(elements))
        print("RESIDUES:", np.unique(residues))
        print("\n\n")

    return x, y, z, elements, residues, backbone, occupancy


def atm_to_map_sphere(mmap, x0, y0, z0, atms, labs, orig, sample, cell, name=''):
    """
       Align and annotate electron density map with ground-truth atomic model coordinates and values.
       Does not resolve clashes.

       :param mmap: (np.ndarray[float]) electron density map intensities
       :param x0: (np.ndarray[float]) atomic x-coordinates
       :param y0: (np.ndarray[float]) atomic y-coordinates
       :param z0: (np.ndarray[float]) atomic z-coordinates
       :param atms: (np.ndarray[float]) atomic labels
       :param labs: (np.ndarray[int]) atomic labels to be mapped (e.g. residue names for each atom from coordinates)
       :param orig: (tuple(float, float, float)) cell origin
       :param sample: (tuple (int, int, int)) sampling rate along x, y and z axis
       :param cell: (tuple(float, float, float)) size of the cell in x, y and z dimensions
       :param name: (str) name to be displayed when genreating maps, useful when many are generated or in parllalel
                    processing, default to '' (empty string)
       :return: amap (np.ndarray[int]) 3D annotated map
       """

    amap = np.zeros(mmap.shape, dtype='f')  # , dtype=np.int)

    x, y, z = np.mgrid[0:amap.shape[0], 0:amap.shape[1], 0:amap.shape[2]].astype(np.float64)
    x, y, z = grid_to_real(x, y, z, orig, cell, sample)
    # x = (x + 0) * (cell.z / sample[2]) + orig.z
    # y = (y + 0) * (cell.y / sample[1]) + orig.y
    # z = (z + 0) * (cell.x / sample[0]) + orig.x

    # generate non-gaussian labeles
    for i in range(len(atms)):
        progress(i, len(atms), desc="Sphere labelling maps "+str(name))

        mask = (x-x0[i])**2 + (y-y0[i])**2 + (z-z0[i])**2 <= ATOMS[atms[i]].waals**2
        amap[mask] = labs[i]

    return amap


def atm_to_map(mmap, x0, y0, z0, atms, labs, orig, sample, cell, occ=None, two_sigma=True, res=3, name='',
               clashes_info=False):
    """
    Align and annotate electron density map with ground-truth atomic model coordinates and values.*

    * It is much more accurate** to use two_sigma=True, however it is considerably slower. For better performance
    but less accurate maps set two_sigma=False.

    ** If two_sigma == False, the resulting map does not allow for increased boundary values as a result of
    Gaussian overlap.

    :param mmap: (np.ndarray[float]) electron density map intensities
    :param x0: (np.ndarray[float]) atomic x-coordinates
    :param y0: (np.ndarray[float]) atomic y-coordinates
    :param z0: (np.ndarray[float]) atomic z-coordinates
    :param atms: (np.ndarray[float]) atomic labels
    :param labs: (np.ndarray[int]) atomic labels to be mapped (e.g. residue names for each atom from coordinates)
    :param orig: (tuple(float, float, float)) cell origin
    :param sample: (tuple (int, int, int)) sampling rate along x, y and z axis
    :param cell: (tuple(float, float, float)) size of the cell in x, y and z dimensions
    :param occ: (np.ndarray[float]) atomic occupation, default 1 (not affected)
    :param two_sigma: (bool) True if two sigmas are to be used in Gaussian maps, False if one sigma, default True
    :param res: (float) resolution of the generated map, default 3
    :param name: (str) name to be displayed when genreating maps, useful when many are generated or in parllalel
                 processing, default to '' (empty string)
    :param clashes_info: True if print report of how many clashes, False otherwise, default False
    :return: gmap (np.ndarray[int]) 3D gaussian map used to generate annotations
             amap (np.ndarray[int]) 3D annotated map
    """

    amap = np.zeros(mmap.shape, dtype='f')
    gmap = np.zeros(mmap.shape, dtype='f')
    contribution = np.zeros(mmap.shape, dtype='f')
    clashes = 0

    if occ is None:
        occ = np.ones(len(atms))

    def label(thr, gm, lab):
        # CLASH RESOLUTION
        carea = contribution[thr]    # get gaussian area from contribution map
        garea = gm[thr]              # get gaussian area from gaussian map
        larea = amap[thr]            # get gaussian area from labelled map
        mx = garea > carea           # create boolean mask for voxels with larger contribution in gaussian
        carea[mx] = garea[mx]        # update the contribution vals with new highest contribution from gaussian
        larea[mx] = lab              # update only the values with larger contribution with new labs
        contribution[thr] = carea    # ovewrite the gaussian area in contribution map with new values
        return larea

    x, y, z = np.mgrid[0:amap.shape[0], 0:amap.shape[1], 0:amap.shape[2]].astype(np.float64)
    x, y, z = grid_to_real(x, y, z, orig, cell, sample)
    # x = (x + 0) * (cell.z / sample[2]) + orig.z
    # y = (y + 0) * (cell.y / sample[1]) + orig.y
    # z = (z + 0) * (cell.x / sample[0]) + orig.x

    sig = 0.225 * res  # 0.225 * res

    for i in range(len(atms)):
        progress(i, len(atms), desc="Generating Gaussian map "+str(name))

        n = 4.4     # gaussian cut-off sigma experimentally derived from chimera
        cutoff = gauss3d(x, y, z, x0[i], y0[i], z0[i], sig, scale=ATOMS[atms[i]].num * occ[i]) > \
            gauss3d(x0[i] + (n * sig), y0[i] + (n * sig), z0[i] + (n * sig), x0[i], y0[i], z0[i], sig,
                    scale=ATOMS[atms[i]].num*occ[i])
        gmap[cutoff] += (gauss3d(x, y, z, x0[i], y0[i], z0[i], sig, scale=ATOMS[atms[i]].num*occ[i])[cutoff])

        if not two_sigma:
            m = 1.5  # label threshold sigma
            g = gauss3d(x, y, z, x0[i], y0[i], z0[i], sig, scale=ATOMS[atms[i]].num * occ[i])
            s = gauss3d(x0[i] + (m * sig), y0[i] + (m * sig), z0[i] + (m * sig), x0[i], y0[i], z0[i], sig,
                        scale=ATOMS[atms[i]].num * occ[i])
            thresh = g > s

            clashes += np.sum(np.bitwise_not(np.bitwise_or(amap[thresh] == 0, amap[thresh] == labs[i])))

            amap[thresh] = label(thresh, g, labs[i])  # for debugging use ATOMS[atms[i]].num

    if two_sigma:
        for i in range(len(atms)):
            progress(i, len(atms), desc='Two gauss labelling '+str(name))

            n1 = 1.5  # gaussian cut-off value (label)
            n2 = 2.0  # gaussian cut-off area (mask)

            g = gauss3d(x, y, z, x0[i], y0[i], z0[i], sig, scale=ATOMS[atms[i]].num * occ[i])
            s1 = gauss3d(x0[i] + (n1 * sig), y0[i] + (n1 * sig), z0[i] + (n1 * sig), x0[i], y0[i], z0[i], sig,
                         scale=ATOMS[atms[i]].num * occ[i])
            s2 = gauss3d(x0[i] + (n2 * sig), y0[i] + (n2 * sig), z0[i] + (n2 * sig), x0[i], y0[i], z0[i], sig,
                         scale=ATOMS[atms[i]].num * occ[i])

            temp = np.copy(gmap)
            temp[g < s2] = 0        # extract area slightly larger than of interest to include gausian overlaps
            thresh = temp >= s1     # label the values of interest within that area

            clashes += np.sum(np.bitwise_not(np.bitwise_or(amap[thresh] == 0, amap[thresh] == labs[i])))
            amap[thresh] = label(thresh, g, labs[i])

    if clashes_info:
        print("--------- reported number of clashes:", clashes, ".")

    return gmap, amap


def amino6(data):
    """Function will group amino acids into 6 shape-based groups."""
    if len(np.unique(data)) < 6 or len(np.unique(data)) > 22:
        raise RuntimeError("Shape grouping function (amino6) can only be used with residue labels.")
    # ['ALA1', 'CYS2', 'ASP3', 'GLU4', 'PHE5', 'GLY6', 'HIS7', 'ILE8', 'LYS9', 'LEU10', 'MET11',
    #      'ASN12', 'PRO13', 'GLN14', 'ARG15',
    #  'SER16', 'THR17', 'VAL18', 'TRP19', 'TYR20']
    ndata = np.copy(data)
    # ndata[ndata == 1] = 1
    ndata[ndata == 6] = 1
    ndata[ndata == 16] = 1
    ndata[ndata == 2] = 1
    ndata[ndata == 18] = 1
    ndata[ndata == 17] = 1
    ndata[ndata == 8] = 1
    ndata[ndata == 13] = 2
    ndata[ndata == 10] = 3
    # ndata[ndata == 3] = 3
    ndata[ndata == 12] = 3
    ndata[ndata == 4] = 3
    ndata[ndata == 14] = 3
    ndata[ndata == 11] = 3
    ndata[ndata == 9] = 4
    ndata[ndata == 15] = 4
    ndata[ndata == 7] = 5
    # ndata[ndata == 5] = 5
    ndata[ndata == 20] = 5
    ndata[ndata == 19] = 6
    ndata[ndata==21] = 7
    return ndata


def save_map(data, orig, cell, path="gen_map.mrc", overwrite=True):
    """
    Save data as mrc file.

    :param data: (np.ndarray[float]) electron density map intensities
    :param orig: (tuple(float, float, float)) cell origin
    :param cell: (tuple(float, float, float)) size of the cell in x, y and z dimensions
    :param path: (str) filename and save location
    :param overwrite: (bool) if True, overwriting data if it exists
    """
    if data is None:
        raise RuntimeError("Cannot save None object.")
    if os.path.exists(path):
        if overwrite:
            print("WARNING: overwriting data at:\n        ", path)
        else:
            raise RuntimeError("Data path already exists at:\n        ", path)

    if len(np.unique(data)) < 100:
        data = data.astype('int8')
    else:
        data = data.astype('float32')

    mrcf = mrc.new(name=path, overwrite=overwrite)
    mrcf.set_data(data)
    mrcf.header.origin.x = orig.x
    mrcf.header.origin.y = orig.y
    mrcf.header.origin.z = orig.z
    mrcf.header.cella = cell
    mrcf.close()


def amino_map(data, orig, cell, path, overwrite=False):
    """
    Split and save the data into a number of maps equal to number of labels. Desgiend for use in Chimara and better
    visualisation of discrete labels with available visualisation software.

    :param data: (numpy.ndarray) 3D protein image
    :param orig: (np.recarray[float]) origin coordinates
    :param cell: (np.recarray[float]) size of the sampled cell
    :param path: (str) path to save data at
    :param overwrite: (bool) if True overwriting exiting labels, if False and the map exists, runtime error is raised
    """
    if len(np.unique(data)) == 1:
        print("\n\nWARNING: map has no unique values, so maps for each label won't be saved.\n")
    for i in np.unique(data):
        lab = int(i)
        if lab == 0:
            continue
        m = np.copy(data)
        m[m != lab] = 0
        save_map(m, orig, cell, path=path[:-4]+"_"+str(i)+".mrc", overwrite=overwrite)


def gen_lab(mpath, apath, gpath, lpath, labs, sigma, resolution, clabs=None, name=None):
    """
    Generate and save labelled map for specified resolution, label type (backbone vs amino acids) and sigma (1 vs 2).

    :param mpath: (str) path to load map (.mrc/.map) from
    :param apath: (str) path to load atomic model (.pdb/.cif) from
    :param gpath: (str) path to save Gaussian map at
    :param lpath: (str) path to save labelled map at
    :param labs: (str) one of ['backbone', 'residue', 'custom']
    :param sigma: (bool) True if generating with 2 sigmas (more accurate), False if with 1 sigma (faster)
    :param resolution: (float) resolution generating at
    :param name: (str) name to be displayed in tqdm, mostly used in parallel processing for tracking, default is None
    """

    if name is not None:
        print("\nEXECUTING", name)

    # load data
    mmap, orig, sample, cell = load_map(mpath)
    x, y, z, atm, res, bb, occ = load_model(apath)

    # annotate data
    if labs == 'backbone':
        labels = bb
    elif labs == 'amino':
        labels = res
    elif labs == 'custom':
        if clabs is None:
            raise RuntimeError("When providing custom labels parameter 'clabs' is expected with a list of labels"
                               "equal to the list of atoms in the pdb file.")
        labels = clabs
        if len(labels) != len(atm):
            raise RuntimeError("Length of the custom label file must be the same as the number of atoms in the .pdb"
                               "file (={}). Check for extra whitespace characters.".format(len(atm)))
        ids = defaultdict(lambda: len(ids))
        labels = [ids[i]+1 for i in labels]
    else:
        raise RuntimeError("Parameter 'labels' is expecting either one of ['backbone', 'amino', 'custom'].")
    gmap, amap = atm_to_map(mmap, x, y, z, atm, labels, orig, sample, cell, occ=occ, two_sigma=sigma, res=resolution,
                            name=name)

    # save data
    if not os.path.exists(gpath):
        save_map(gmap, orig, cell, path=gpath)
    save_map(amap, orig, cell, path=lpath)


def data_vox_norm(d, samp, cell, vox=0.7, vox_min=0.6, vox_max=1.0):
    """
    Resamples image to desired voxel size if outside of vox_sh_min and vox_sh_max. Isotropic.

    Uses spline interpolation.

    :param d: (np.ndarray[float]) 3D volume intensities
    :param samp: (tuple(int, int, int)) sampling rate
    :param cell: (np.recarray[float]) size of the sampled cell
    :param vox: (float) desired voxel size
    :param vox_min: (float) minimum allowed voxel size
    :param vox_max: (float) maximum allowed voxel size
    :return: d (np.ndarray[float]) rescaled 3D volume densities
             samp (tuple(int, int, int)) updated sampling rate
    """
    voxx, voxy, voxz = (cell.x / samp[0], cell.y / samp[1], cell.z / samp[2])
    # print("\n", voxx, voxy, voxz, samp, (d.shape[2], d.shape[1], d.shape[0]))
    samp = np.array(samp)
    if voxx > vox_max or voxx < vox_min:
        samp[0] = np.ceil(cell.x) / vox
    if voxy > vox_max or voxy < vox_min:
        samp[1] = np.ceil(cell.y) / vox
    if voxz > vox_max or voxz < vox_min:
        samp[2] = np.ceil(cell.z) / vox
    samp = tuple(samp)
    d = zoom(d, (samp[2] / d.shape[0], samp[1] / d.shape[1], samp[0] / d.shape[2]), order=0)
    # print(cell.x / samp[0], cell.y / samp[1], cell.z / samp[2], samp, (d.shape[2], d.shape[1], d.shape[0]), "\n\n")
    return d, samp


def data_crop(sh, d, orig=None, samp=None, cell=None, mid=None, move_o=False, ind=False):
    """
    Cut map to desired shape s and update the header data. Updated header data will only be returned if all of orig,
    sample and cell passed, otherwise all header data will be None.

    In case of uneven s, the extra voxel is added at the end of an image, thus eliminating the need to have two cases
    for origin.

    :param sh: (tuple[int]) desired shape in z, y and x
    :param d: (np.ndarray[float]) 3D volume intensities
    :param orig: (np.recarray[float]) origin coordinates
    :param samp: (tuple(int, int, int)) sampling rate
    :param cell: (np.recarray[float]) size of the sampled cell
    :param mid: (tuple[int]) custom point to center the crop on, if None crop is centered on the mid point of the map
    :param move_o: (bool) moves origin
    :param ind: (bool) returns indices for the start of the crop
    :return: img (np.ndarray[float]) 3D volume intensities
             orig (np.recarray[float]) updated origin coordinates, only if either of orig, sample and cell is passed
             sample (tuple(int, int, int)) updated sampling, only returned if either of orig, sample and cell is passed
             cell (np.recarray[float]) updated cell size, only returned if either of orig, sample and cell is passed
    """
    class Half:
        def __init__(self, num):
            self.x = num // 2
            self.y = num // 2 if num % 2 == 0 else (num // 2) + 1

    if mid is None:
        # center the crop
        halfz, halfy, halfx = (d.shape[0] // 2, d.shape[1] // 2, d.shape[2] // 2)
    else:
        # custom centre point
        halfz, halfy, halfx = mid   # this has now been swapped in borders function, so reversed order z,y,x

    if d.shape[0] <= sh[0] or d.shape[1] <= sh[1] or d.shape[2] <= sh[2]:  # z,y,x
        # raise RuntimeError("WARNING: map of size={} too small to be cut to desired size of {}.".format(d.shape, sh))
        # pad instead
        new_sh = [s if s > sh[i] else sh[i] for i, s in enumerate(d.shape)]
        new_d = np.zeros(new_sh)
        new_d[0:d.shape[0], 0:d.shape[1], 0:d.shape[2]] = d
        d = new_d
        halfz, halfy, halfx = (d.shape[0] // 2, d.shape[1] // 2, d.shape[2] // 2)
        if orig is not None and samp is not None and cell is not None:
            voxx, voxy, voxz = (cell.x / samp[0], cell.y / samp[1], cell.z / samp[2])
            cell.x, cell.y, cell.z = (new_sh[2] * voxx, new_sh[1] * voxy, new_sh[0] * voxz)
            samp = (new_sh[2], new_sh[1], new_sh[0])  # x,y,z

    x1, x2 = (halfx - Half(sh[2]).x, halfx + Half(sh[2]).y)
    y1, y2 = (halfy - Half(sh[1]).x, halfy + Half(sh[1]).y)
    z1, z2 = (halfz - Half(sh[0]).x, halfz + Half(sh[0]).y)

    if x1 < 0 or x2 > d.shape[2] or y1 < 0 or y2 > d.shape[1] or z1 < 0 or z1 > d.shape[0]:
        raise RuntimeError("WARNING: map of size={} too small to be cut to desired size of {}.".format(d.shape, sh))

    if orig is not None and samp is not None and cell is not None:
        voxx, voxy, voxz = (cell.x / samp[0], cell.y / samp[1], cell.z / samp[2])
        if move_o:
            orig.x, orig.y, orig.z = (orig.x + x1 * voxx,  # (Half(samp[0]-sh[2]).x * voxx),
                                      orig.y + y1 * voxy,  # (Half(samp[1]-sh[1]).x * voxy),
                                      orig.z + z1 * voxz)  # (Half(samp[2]-sh[0]).x * voxz))
        cell.x, cell.y, cell.z = (sh[2] * voxx, sh[1] * voxy, sh[0] * voxz)
        samp = (sh[2], sh[1], sh[0])
    d = d[z1:z2, y1:y2, x1:x2]

    if orig is not None and samp is not None and cell is not None:
        if ind:
            return d, orig, samp, cell, (z1, y1, x1)
        else:
            return d, orig, samp, cell
    else:
        if ind:
            return d, (z1, y1, x1)
        else:
            return d


def borders(d, margin=5, cshape=None):
    """
    Return crop box size along with its middle point.

    :param d: (np.ndarray[bool]) boolean mask for the protein
    :param margin: (int) additional margin to add at the limits of the data extent, default 5
    :param cshape: (int) optional - minimum shape for each dimension
    :return: sh, (halfx, halfy, halfz) (int), (tuple[int]) - shape of the box and middle of the box indices as tuple
    """
    ext = data_ext(d)   # find data extent in each direction
    # find a margin that prevents padding small maps beyond their original shape
    sh = (ext[5] - ext[4] + 2 * int(min(margin/2, (ext[4]+1)/2, (d.shape[0]-ext[5]-1)/2)),
          ext[3] - ext[2] + 2 * int(min(margin/2, (ext[2]+1)/2, (d.shape[1]-ext[3]-1)/2)),
          ext[1] - ext[0] + 2 * int(min(margin/2, (ext[0]+1)/2, (d.shape[2]-ext[1]-1)/2)))  # z,y,x
    # keep minimum of cshape or size of the map
    if cshape:
        sh = (max(cshape, sh[0]), max(cshape, sh[1]), max(cshape, sh[2]))
    sh = (min(d.shape[0], sh[0]), min(d.shape[1], sh[1]), min(d.shape[2], sh[2]))  # z,y,x
    halfx = ext[4] + ((ext[5] - ext[4]) // 2)
    halfy = ext[2] + ((ext[3] - ext[2]) // 2)
    halfz = ext[0] + ((ext[1] - ext[0]) // 2)   # z,y,x
    return sh, (halfx, halfy, halfz)


def data_ext(d):
    """
    Return minimum and maximum indices of of voxels containging data in each of the 3 dimensions.

    :param d: (np.ndarray[bool]) boolean mask for the protein
    :return: xmin, xmax, ymin ymax, zmin, zmax (int) minimum and maximum indices of the voxels containing data
    """
    zmask = np.nonzero(d.sum((1, 2)))[0]
    ymask = np.nonzero(d.sum((0, 2)))[0]
    xmask = np.nonzero(d.sum((0, 1)))[0]
    return xmask.min(), xmask.max(), ymask.min(), ymask.max(), zmask.min(), zmask.max()


def data_isopad(d, orig=None, samp=None, cell=None):
    """
    Padding data for isotropy. The new dimension is the highest of the three.

    :param d: (np.ndarray) 3D data (map or mask)
    :param orig: (np.recarray[float]) origin coordinates
    :param samp: (tuple(int, int, int)) sampling rate
    :param cell: (np.recarray[float]) size of the sampled cell
    :return: d (np.ndarray) new padded isotropic 3D data
             orig (np.recarray[float]) updated origin coordinates, only if either of orig, sample and cell is passed
             sample (tuple(int, int, int)) updated sampling, only returned if either of orig, sample and cell is passed
             cell (np.recarray[float]) updated cell size, only returned if either of orig, sample and cell is passed
    """
    m = max(d.shape)
    if orig is not None and samp is not None and cell is not None:
        voxx, voxy, voxz = (cell.x / samp[0], cell.y / samp[1], cell.z / samp[2])
        cell.x, cell.y, cell.z = (m * voxx, m * voxy, m * voxz)
        samp = (m, m, m)
    dnew = np.zeros((m, m, m), dtype=np.float32)
    dnew[:d.shape[0], :d.shape[1], :d.shape[2]] = d
    if orig is not None and samp is not None and cell is not None:
        return dnew, orig, samp, cell
    return dnew


def data_scale(d, sh, orig=None, samp=None, cell=None):
    """
    Scaling data to the desired shape sh.

    Uses spline interpolation.

    :param d: (np.ndarray) 3D data (map or mask)
    :param sh: (tuple [int, int, int[) shape for the data to be rescaled to
    :param orig: (np.recarray[float]) origin coordinates
    :param samp: (tuple(int, int, int)) sampling rate
    :param cell: (np.recarray[float]) size of the sampled cell
    :return: d (np.ndarray) new padded isotropic 3D data
             orig (np.recarray[float]) updated origin coordinates, only if either of orig, sample and cell is passed
             sample (tuple(int, int, int)) updated sampling, only returned if either of orig, sample and cell is passed
             cell (np.recarray[float]) updated cell size, only returned if either of orig, sample and cell is passed
    """
    if orig is not None and samp is not None and cell is not None:
        cell.x, cell.y, cell.z = (cell.x * (sh[0] / d.shape[0]), cell.y * (sh[1] / d.shape[1]),
                                  cell.z * (sh[2] / d.shape[2]))
    d = zoom(d, (sh[0] / d.shape[0], sh[1] / d.shape[1], sh[2] / d.shape[2]), order=0)
    if orig is not None and samp is not None and cell is not None:
        return d, orig, samp, cell
    return d
