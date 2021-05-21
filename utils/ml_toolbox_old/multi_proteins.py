import glob
import json
import os

from multiprocessing import Queue, Process, cpu_count, current_process
from tqdm import tqdm

import numpy as np
import pandas as pd

from proteins import load_map, load_model, atm_to_map, save_map, gen_lab, data_isopad, data_crop, borders, \
    data_scale, Sample
from utils import progress, divx, normalise


def load_data(path, res, n=None, predict=True, weights=False):   # TODO rename to load_data
    """
    Load data into object-oridented database from two directories: one containing electron density maps
    and one containing corresponding labelled maps (voxel-to-voxel corespondence). Function will only process
    images with associated labels, the rest will be ignored.

    Expected format of the passed directory containing data:
        path/:
            2.5/
            2.5label/
            3/
            3label/
            ...
    Each directory is to contain '.mrc' files with their PDB ID as a filename.

    :param path: (str) directory containing training 2 x res directories of samples and labels
    :param res: (list[float]) list of resolutions to be analysed
    :param n: (int) limit for the number of files to be read in each resolution, default=None
    :return: data (list[Sample]) list of Sample objects holidng info about loaded sample, including:
             id, resolution, paths for current and future files, raw data and corresponding headers, etc.
    :param predict: (bool) does not load labels if True (predict mode)
    :param weights: (bool) loads weights if True (custom weights)

    [REFACTOR: This function was refactored to remove the redundant for loops. For more info see git history.]
    [REFACTOR: This function was refactored to object-oriented database rahter than Pandas, as the introduced structure
               was incapable of processing multiple resolutions (dropping all). For more info see git history.]

    """
    data = []

    for r in res:

        if not predict and n is not None and n > len(os.listdir(os.path.join(path, "{}label/".format(r)))) and \
            n > len(os.listdir(os.path.join(path, "{}/".format(r)))):
            raise RuntimeError("Directory does not contain enough files for the requested {} maps.".format(n))

        if not predict:
            ys = sorted(glob.glob(path + "{}label/*.mrc".format(r)))[:n]
            if len(ys) == 0:
                ys = sorted(glob.glob(path + "{}label/*.map".format(r)))[:n]
            ids = [f.split('/')[-1].split('_')[0] for f in ys]
            xs = sorted([f for f in glob.glob(path + "{}/*.mrc".format(r)) if
                         f.split('/')[-1].split('.')[0] in ids])[:n]
            if len(xs) == 0:
                xs = sorted([f for f in glob.glob(path + "{}/*.map".format(r)) if
                             f.split('/')[-1].split('.')[0] in ids])[:n]
        else:
            xs = sorted(glob.glob(path + "{}/*.mrc".format(r)))[:n]
            if len(xs) == 0:
                xs = sorted(glob.glob(path + "{}/*.map".format(r)))[:n]
            ys = [None for x in xs]

        if len(xs) != len(ys):
            raise RuntimeError("Label mismatch exists between map and label directories.")

        if len(xs) == 0:
            raise RuntimeError("Size of the dataset is zero. Check if you provided corresponding labels, or if"
                               " your .mrc directory exists.")

        if weights:
            ws = sorted(glob.glob(path + "/{}weights/*.mrc".format(r)))[:n]
            if len(ws) == 0:
                ws = sorted(glob.glob(path + "/{}weights/*.map".format(r)))[:n]
            if len(ws) != len(xs):
                raise RuntimeError("Number of weights in {}weights is not the same as the number of maps in {}.".format(
                                    r, r))

        if weights:
            for i, (x, y, w) in enumerate(zip(xs, ys, ws)):
                progress(i, len(xs), desc="Loading data at {} resolution".format(r))
                data.append(Sample(r, x, y, w))
        else:
            for i, (x, y) in enumerate(zip(xs, ys)):
                progress(i, len(xs), desc="Loading data at {} resolution".format(r))
                data.append(Sample(r, x, y))

    return data


def preproces_data(data, mode='scale', cshape=None, res=None,  norm=True, crop_to_labels=True,
                   step=None, background=None, threshold=None, background_label=None):
    """
    Wrapper function around all of the data loading methods. Mode must be one of crop | scale | tile, defaults to scale.

    :param data (list[Sample]) list of Sample objects holidng info about loaded sample, including
             id, resolution, paths for current and future files, raw data and corresponding headers, etc.
    :param mode: (str) one of crop | scale | tile
    :param res: (list[float]) indicates which range of resolution to load, if res=None it loads all
    :param cshape: (tuple[int]) desired shape of the image, isotropic, default None
    :param norm: (bool) normalise data between 0 and 1
    :param crop_to_labels: (bool) crops map to the extent of the data if labels are present
    :param step: (int) use only with threshold, step between the start of each tile, for dense it's cshape-2*margin
    :param background: (float) sets limit for proportion of background in accepted extracted tiles,
                                 default None
    :param threshold: (float) if background_limit used, sets a threshold for a boolean map to extract tiles across
                          (values below are bg), default None; if not passed, executing background_label by default
    :param background_label: (float) if background_limit used, uses passed background value for a boolean map to
                                 extract tiles across
    :return: data (list[Sample]) list of samples with updated pre-processed data and header information
    """
    # TODO here
    # normalise
    if mode == 'crop':
        if cshape is None:
            raise RuntimeError("When loading data in crop mode, crop shape must be passed with cshape argument.")
        ret = preproces_data_crop(data, cshape, res=res, norm=norm)
    elif mode == 'scale':
        ret = preproces_data_scale(data, res=res, mx=cshape, norm=norm)
    elif mode == 'tile':
        ret = preproces_data_tile(data, res=res, cshape=cshape, norm=norm, crop=crop_to_labels,
                                  step=step, background=background,
                                  threshold=threshold, background_label=background_label)
    else:
        raise RuntimeError("The mode argument must be one of: crop | scale | tile .")

    if len(ret) == 0:
        raise RuntimeError("All data has been excluded - no data to process.")

    return ret


def preproces_data_crop(data, cshape=64, res=None, norm=True):
    """
    Loads and preprocesses the data the list of Sample object originating from load_data function. Returns
    an updated list of Sample object that is tensorflow-ready if data is extracted as described in Sample docstring.

    The preprocessing includes:
        1) cropping the data to the shape requested by cshape;
        2) dropping the data that has been too small to crop to the desired shape;
        *) header information is updated along the way for the ease of save and display in Chimera / Coot.

    :param data (list[Sample]) list of Sample objects holidng info about loaded sample including
             id, resolution, paths for current and future files, raw data and corresponding headers, etc.
    :param res: (list[float]) indicates which range of resolution to load, if res=None it loads all
    :param cshape: (int) desired shape of the image, isotropic
    :param norm: (bool) normalise data between 0 and 1
    :return: data (list[Sample]) list of samples with updated pre-processed data and header information

    [REFACTOR: This function is a refactor of a prototype tensorflow pre-processing make_np_dataset function.]
    """

    if res is None:
        res = np.unique([d.res for d in data])

    if isinstance(cshape, tuple) and len(cshape) == 3:
        pass
    elif isinstance(cshape, int):
        cshape = ((cshape,) * 3)
    else:
        raise RuntimeError("Parameter 'cshape' should be either a tuple or length 3 representing 3D shape, or a single"
                           "integer.")

    excl = []
    for r in res:
        samples = [d for d in data if d.res == r]

        for i, s in enumerate(samples):
            progress(i, len(samples), desc="Pre-processing data (mode=crop) at {} resolution".format(r))

            # 0) normalise
            if norm:
                s.map = normalise(s.map)

            # 1) crop data to custom size
            try:
                s.map, s.header_map.orig, s.header_map.samp, s.header_map.cell = \
                    data_crop(cshape, s.map, orig=s.header_map.orig, samp=s.header_map.samp, cell=s.header_map.cell,
                              move_o=True)
                if s.lab is not None:
                    s.lab, s.header_lab.orig, s.header_lab.samp, s.header_lab.cell = \
                        data_crop(cshape, s.lab, orig=s.header_lab.orig, samp=s.header_lab.samp, cell=s.header_lab.cell,
                                  move_o=True)
                if s.weight is not None:
                    s.weight, s.header_weight.orig, s.header_weight.samp, s.header_weight.cell = \
                        data_crop(cshape, s.weight, orig=s.header_weight.orig, samp=s.header_weight.samp,
                                  cell=s.header_weight.cell, move_o=True)
            except RuntimeError:
                excl.append((s.res, s.id))

    # 2) drop maps that were excluded
    if len(excl) != 0:
        data = [d for d in data if (d.res, d.id) not in excl]
        print("\nExcluded {} maps:".format(len(excl)))
        print(excl)

    return data


def preproces_data_scale(data, mx=None, res=None, norm=True):
    """
    Loads and preprocesses the data the list of Sample object originating from load_data function. Returns
    an updated list of Sample object that is tensorflow-ready if data is extracted as described in Sample docstring.

    The preprocessing includes:
        1) cropping the data to only contain the true values from the mask;
        2) padding the data to isotopy to prevent from deformation following resizing;
        3) resizing the data to required size;
        4) dropping the data that has been too small to crop to the desired shape;
        *) header information is updated along the way for the ease of save and display in Chimera / Coot.

    :param data (list[Sample]) list of Sample objects holidng info about loaded sample including
             id, resolution, paths for current and future files, raw data and corresponding headers, etc.
    :param mx (tuple[int]) desired image size, default None
    :param res: (list[float]) indicates which range of resolution to load, if res=None it loads all
    :param norm: (bool) normalise data between 0 and 1
    :return: data (list[Sample]) list of samples with updated pre-processed data and header information

    [REFACTOR: This function is a refactor of a prototype tensorflow pre-processing make_np_dataset function extended
               to allow for slicing.]
    """

    if res is None:
        res = np.unique([d.res for d in data])

    excl = []
    for r in res:
        samples = [d for d in data if d.res == r]

        for i, s in enumerate(samples):
            progress(i, len(samples), desc="Pre-processing data (mode=slice) at {} resolution".format(r))

            # 0) normalise
            if norm:
                s.map = normalise(s.map)

            # 1) crop data to the extent of the data
            if s.lab is None:
                # TODO refactor this to allow no label but just no cropping
                raise RuntimeError("Cannot crop data to contain signal, labels (mask) not available.")
            dshape, mid = borders(s.lab)
            try:
                s.map, s.header_map.orig, s.header_map.samp, s.header_map.cell = \
                    data_crop(dshape, s.map, orig=s.header_map.orig, samp=s.header_map.samp, cell=s.header_map.cell,
                              mid=mid)
                s.lab, s.header_lab.orig, s.header_lab.samp, s.header_lab.cell = \
                    data_crop(dshape, s.lab, orig=s.header_lab.orig, samp=s.header_lab.samp, cell=s.header_lab.cell,
                              mid=mid)
                if s.weight is not None:
                    s.weight, s.header_weight.orig, s.header_weight.samp, s.header_weight.cell = \
                        data_crop(dshape, s.weight, orig=s.header_weight.orig, samp=s.header_weight.samp,
                                  cell=s.header_weight.cell, mid=mid)
            except RuntimeError:
                excl.append((s.res, s.id))

            # 2) ensure data shape is isotropic before rescaling to avoid streching
            s.map, s.header_map.orig, s.header_map.samp, s.header_map.cell = \
                data_isopad(s.map, orig=s.header_map.orig, samp=s.header_map.samp, cell=s.header_map.cell)
            s.lab, s.header_lab.orig, s.header_lab.samp, s.header_lab.cell = \
                data_isopad(s.lab, orig=s.header_lab.orig, samp=s.header_lab.samp, cell=s.header_lab.cell)
            if s.weight is not None:
                s.weight, s.header_weight.orig, s.header_weight.samp, s.header_weight.cell = \
                    data_isopad(s.weight, orig=s.header_weight.orig, samp=s.header_weight.samp, cell=s.header_weight.cell)

        # 3) finally, zoom data (have to be in a separate loop so we can get the largest dimension)
        if mx is None:
            mx = divx(max(np.asarray([d.map.shape for d in data]).flatten()))

        for i, s in enumerate(samples):
            progress(i, len(samples), desc="Rescaling data at {} resolution".format(r))

            s.map, s.header_map.orig, s.header_map.samp, s.header_map.cell = \
                data_scale(s.map, (mx, mx, mx), orig=s.header_map.orig, samp=s.header_map.samp, cell=s.header_map.cell)
            s.lab, s.header_lab.orig, s.header_lab.samp, s.header_lab.cell = \
                data_scale(s.lab, (mx, mx, mx), orig=s.header_lab.orig, samp=s.header_lab.samp, cell=s.header_lab.cell)
            if s.weight is not None:
                s.weight, s.header_weight.orig, s.header_weight.samp, s.header_weight.cell = \
                    data_scale(s.weight, (mx, mx, mx), orig=s.header_weight.orig, samp=s.header_weight.samp,
                               cell=s.header_weight.cell)

    # 4) drop maps that were not used
    if len(excl) != 0:
        data = [d for d in data if (d.res, d.id) not in excl]
        print("\nExcluded {} maps:".format(len(excl)))
        print(excl)

    return data


def preproces_data_tile(data, res=None, cshape=64, margin=8, norm=True, crop=True,
                        step=None, background=None, threshold=None, background_label=None):
    """
    Wrapper around Sample class intrinsic tiling function 'decompose'.
    """
    # raise NotImplementedError("This function has not yet been implemented.")
    if res is None:
        res = np.unique([d.res for d in data])

    if not isinstance(cshape, int) or not isinstance(margin, int):
        raise RuntimeError("Parameter 'cshape' and 'margin' should be a single integer.")

    for r in res:
        samples = [d for d in data if d.res == r]

        for i_sample, s in enumerate(samples):
            progress(i_sample, len(samples), desc="Pre-processing data (mode=tile) at {} resolution".format(r))
            s.decompose(cshape=cshape, norm=norm, crop=crop, step=step,
                        background_limit=background,
                        threshold=threshold, background_label=background_label)

    print("Total number of tiles:", len([tile for d in data for tile in d.tiles]))

    return data


def postproces_data(data, tiles=False, res=None):
    """
    Wrapper function around all of the data loading methods. Mode must be one of [ tile ], defaults to scale.
    # TODO so far there is only one postprocess option but there might be more in future...

    :param data (list[Sample]) list of Sample objects holidng info about loaded sample, including
             id, resolution, paths for current and future files, raw data and corresponding headers, etc.
    :param tiles (bool) recomposes map rather than preds if True (and saves it under map_rec)
    :param res: (list[float]) indicates which range of resolution to load, if res=None it loads all
    """
    ret = postprocess_data_tile(data, tiles=tiles, res=res)

    if len(ret) == 0:
        raise RuntimeError("All data has been excluded - no data to process.")

    return ret


def postprocess_data_tile(data, tiles=False, res=None):
    """
    Wrapper around Sample class intrinsic untiling function 'recompose'.
    """
    if res is None:
        res = np.unique([d.res for d in data])

    for r in res:
        samples = [d for d in data if d.res == r]

        for i_sample, s in enumerate(samples):
            progress(i_sample, len(samples), desc="Re-stitching data (mode=tile) at {} resolution".format(r))
            s.recompose(tiles=tiles)

    return data


def query_data(data, res=None, verbose=True, save=False, savename="data_report.txt", ):     # TODO rename querry_data
    """
    Writes header data of the map files into text file for each entry in passed path_df data frame.

    :param data (list[Sample]) list of Sample objects holidng info about loaded sample including
             id, resolution, paths for current and future files, raw data and corresponding headers, etc.
    :param res: (list[float]) indicates which range of resolution to load, if res=None it loads all
    :param verbose: (bool) if True prints report to screen, default verbose=True
    :param save: (booL) if True saves report to file in savename parameter, default=False
    :param savename (str) filename to use for saved report, default="data_report.txt"
    :return: data (list[Sample]) list of samples with updated pre-processed data and header information

            headers:    PDB    RES    OX    OY    OZ    CX    CY    CZ    WX    WY    WZ    SX    SY    YZ
               0        pdb1    x     x     x     x     x     x     x     x     x     x     x     x     x
              ...

            where:
                PDB:            (str) ID of the map
                OX, OY, OZ:     (np.float64) origin coordinates in x, y and z in Angstroms
                CX, CY, CZ:     (np.float64) cell size in x, y and z in Angstroms
                WX, WY, WZ:     (np.float64) pixel size in x, y and z in Angstroms
                SX, SY, SZ:     (np.float64) sampling of the cell along the x, y and z axis (number of voxels)

    [REFACTOR: This function was refactored to remove redundant arrays definitions. For more info see git history.]
    """

    if res is None:
        report = np.zeros((14, len(data)), dtype=np.object)
    else:
        data = [d for d in data if d.res in res]
        if len(data) == 0:
            raise RuntimeError("Data does no contain any maps with specified resolutions of: {}".format(res))
        report = np.zeros((14, len(data)), dtype=np.object)

    for i, d in enumerate(data):
        progress(i, len(data), desc="Compiling data report")

        report[0, i] = d.id                                             # PDB ID
        report[1, i] = d.res                                            # resolution
        report[2, i] = d.header_map.orig.x                              # cell.x
        report[3, i] = d.header_map.orig.y                              # cell.y
        report[4, i] = d.header_map.orig.z                              # cell.x
        report[5, i] = d.header_map.cell.x                              # orig.x
        report[6, i] = d.header_map.cell.y                              # orig.y
        report[7, i] = d.header_map.cell.z                              # orig.z
        report[8, i] = d.header_map.cell.x / d.header_map.samp[0]     # voxl.x
        report[9, i] = d.header_map.cell.y / d.header_map.samp[1]     # voxl.y
        report[10, i] = d.header_map.cell.z / d.header_map.samp[2]    # voxl.z
        report[11, i] = d.header_map.samp[0]                          # samp.x
        report[12, i] = d.header_map.samp[1]                          # samp.y
        report[13, i] = d.header_map.samp[2]                          # samp.z

    info_df = pd.DataFrame(report.T, columns=['PDB', 'RES', 'orig.x', 'orig.y', 'orig.z', 'cell.x', 'cell.y', 'cell.z',
                                              'vox.x', 'vox.y', 'vox.z', 'samp.x', 'samp.y', 'samp.z'])
    if verbose:
        pd.set_option('display.max_rows', len(data))
        pd.set_option('display.max_columns', 14)
        print()
        print(info_df)
    if save:
        info_df.to_csv(savename)


def plot_data(data):
    """Plot all maps and labels."""
    for d in data:
        d.plot_data()


def gen_labs(gpath, res=None, custom_labs=False, two_sigma=False, gen_maps=True, jlabs=False, mpi=False, todoq=None):
    """
    Generates and saves labels for a set of resolutions, sigmas and label options (hardcoded).

    If want to use mpi, call mpi_gen_labs instead (see todoq parameter description for details).

    :param gpath: (str) global path to to generate all subdirectories at; at minimum it must contain:
                  - subdirectory called 'pdbs' with with .pdb/.cif files containing atomic models
                  - optionally* subdirectory called 'maps' with .mrc/.map images to be labelled
                  * if 'maps' does not exist, it will be generated (gen_maps will be set to True)
    :param res: (list[int]) which resolutions to be generated for
    :param custom_labs (bool) generate labels with a custom input, expecting a 'pdblabs/*.csv' file for to each '.pdb'.
    :param two_sigma (bool) generate labels based on two sigmas, for more info see atm_to_map function docstring
    :param gen_maps (bool) generates chimera maps along the labels
    :param jlabs (bool) take resolution from metadata.json file rather than directory name
    :param mpi: (bool) True enables multiprocessing, False proceeds sequentially
    :param todoq: (multiprocessing.Queue) if mpi is set to True, parameters are only put on Queue passed here rather
                  than executed

    [REFACTOR: this function is equivallent to previous make_save_labels->load_data,atm_to_map_make sequence, but
    executed one by one (rather then all the maps then all the labels) and saving straight to disk to save RAM.]
    """
    backbone = ['backbone', 'amino'] if not custom_labs else ['custom']
    two_sigma = [False, True] if two_sigma else [False]
    resolution = [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8] if res is None else res

    if not os.path.exists(gpath):
        raise RuntimeError("Path does not exist:", gpath)

    if jlabs and not os.path.exists(os.path.join(gpath, 'metadata.json')):
        raise RuntimeError("When selecting --jlabs/-jl parameter, you must provide an "
                "associated 'metadata.json' file containing resolutions.")
    if jlabs:
        with open(os.path.join(gpath, 'metadata.json'), 'r') as f:
            jfile = f.read()
        jres = json.loads(jfile)

    pdb_path = os.path.join(gpath, "pdbs")
    chim_path = os.path.join(gpath, "maps")
    gauss_path = os.path.join(gpath, "gauss-maps")
    lab_path = os.path.join(gpath, "label-maps")
    if custom_labs:
        clab_path = os.path.join(gpath, "pdblabs")

    if not os.path.exists(pdb_path):
        raise RuntimeError("Path containing '.pdb' model files does not exist:", pdb_path)
    pdbs = os.listdir(pdb_path)
    if not os.path.exists(chim_path) or len(os.listdir(chim_path)) == 0:
        print("\n\nWARNING: Generating synthetic maps as no maps directory was found for labeling in {}".format(gpath))
        gen_maps = True
        if not os.path.exists(chim_path):
            os.mkdir(chim_path)
    # else:
    #     if len(os.listdir(chim_path)) != len(res):
    #         raise RuntimeError("The number of maps to generate is not the same as the number of res.")
    if custom_labs:
        if not os.path.exists(clab_path):
            raise RuntimeError("When chosing parameter 'custom' for labels ('custom_labs') you need to provide a"
                               "directory called 'pdblabs' with .csv files containing labels for each .pdb file, "
                               "including value for each atom at separate line.")
        else:
            clabs = os.listdir(clab_path)
            if len(clabs) != len(pdbs):
                raise RuntimeError("The number of custom labels list is not the same as the number of the pdb files.")

    if not os.path.exists(gauss_path):
        os.mkdir(gauss_path)
    if not os.path.exists(lab_path):
        os.mkdir(lab_path)

    if mpi and todoq is None:
        raise RuntimeError("When using MPI, please pass a queue as a parameter.")

    if gen_maps:
        print("\n\n################################################ GENERATING CHIMERA MAPS\n")
        for r in resolution:
            p = os.path.join(chim_path, str(r))
            if not os.path.exists(p):
                os.mkdir(p)
            for pdb in pdbs:
                print("\n------------------------- RES:", r, "PDB:", pdb[:-4])
                if os.path.exists(os.path.join(p, pdb[:-4] + '.mrc')):
                    print("Map exists, skipping.")
                    continue
                print("\nPDB:", pdb[:-4])
                s = "chimera --nogui --script 'chimera_molmap.py %s %d %s.mrc' > /dev/null" % (
                    os.path.join(pdb_path, pdb), float(r),
                    os.path.join(p, pdb[:-4]))
                os.system(s)

    print("\n\n################################################ GENERATING LABELS\n")
    for j, bblabs in enumerate(backbone):
        lpath = os.path.join(lab_path, bblabs)
        if not os.path.exists(lpath):
            os.mkdir(lpath)

        for i, s in enumerate(two_sigma):
            l_path = os.path.join(lpath, str(i + 1) + "s")
            if not os.path.exists(l_path):
                os.mkdir(l_path)

            for r in resolution:
                print("\nRES:", r)
                print("TWO_SIGMA:", s)
                print("LABELS:", bblabs)
                gp = os.path.join(gauss_path, str(r))
                lp = os.path.join(l_path, str(r))
                cp = os.path.join(chim_path, str(r))
                if not os.path.exists(gp):
                    os.mkdir(gp)
                if not os.path.exists(lp):
                    os.mkdir(lp)
                if not os.path.exists(cp):
                    raise RuntimeError("Maps path does not exist;", cp)
                if len(os.listdir(cp)) != len(pdbs):
                    raise RuntimeError("Number of the maps to label in {} is not the same as the number of pdb files"
                                       .format(r))

                for i, pdb in enumerate(pdbs):
                    if jlabs:
                        for entry in jres:
                            if entry['entry'] == pdb[:-4]:
                                r = entry['resolution']

                    pname = "RES: " + str(r) + " | LABS? " + str(bblabs) + " | 2S? " + str(s) + " | PDB: " + pdb[:-4]
                    if mpi:
                        print("\n------------------------- SCHEDULING", pname)
                    else:
                        print("\n------------------------- EXECUTING", pname)
                    if 'water' in pdb or 'CMO' in pdb:      # EXCLUSIONS to be added here
                        continue
                    mrc = pdb[:-4] + ".mrc"
                    gmrc = pdb[:-4] + "_gauss.mrc"
                    lab = pdb[:-4] + "_{}.mrc".format(bblabs)

                    if os.path.exists(os.path.join(lp, lab)):
                        print("Labels exist, skipping.")
                        continue

                    clab = None
                    if custom_labs:
                        if pdb[:-4]+'.txt' not in clabs and pdb[:-4]+'.csv' not in clabs:
                            raise RuntimeError("PDB: {} does not have a corresponding custom label in {}.".format(
                                pdb[:-4], clab_path
                            ))
                        clab_file = open(os.path.join(clab_path, pdb[:-4]+".csv"), 'r')
                        clab = clab_file.readlines()
                        clab = [cl.strip() for cl in clab]

                    if mpi:
                        todoq.put([os.path.join(cp, mrc), os.path.join(pdb_path, pdb),
                                   os.path.join(gp, gmrc), os.path.join(lp, lab),
                                   bblabs, s, r, clab, pname])
                    else:
                        gen_lab(os.path.join(cp, mrc), os.path.join(pdb_path, pdb),
                                os.path.join(gp, gmrc), os.path.join(lp, lab),
                                bblabs, s, r, name=pdb[:-4], clabs=clab)


def mpi_gen_labs(gpath, res=None, custom_labs=False, two_sigma=False, gen_maps=True, jlabs=False):
    """
    Multiprocessing wrapper around gen_labs with mpi=True and todoq with all combination of parameters.
    Actual execution of gen_lab is performed here when processes are spawned.

    To generate sequentially, call gen_labs instead.

    :param gpath: (str) global path to to generate all subdirectories at; at minimum it must contain:
                  - subdirectory called 'pdbs' with with .pdb/.cif files containing atomic models
                  - optionally* subdirectory called 'maps' with .mrc/.map images to be labelled
                  * if 'maps' does not exist, it will be generated (gen_maps will be set to True)
    :param res: (list[int]) which resolutions to be generated for
    :param custom_labs (bool) generate labels with a custom input, expecting a 'pdblabs/*.csv' file for to each '.pdb'.
    :param gen_maps (bool) generates chimera maps along the labels
    """

    def worker(td: Queue):
        while not td.empty():
            try:
                args = td.get_nowait()
            except Exception as e:
                #if td.empty():
                #    break
                continue
            else:
                gen_lab(*args)
                print("------------------------- DONE " + str(args[-1]) + " | " + str(current_process().pid))

    procs = []
    todo = Queue()
    gen_labs(gpath, res=res, custom_labs=custom_labs, two_sigma=two_sigma, gen_maps=gen_maps, jlabs=jlabs,
            mpi=True, todoq=todo)

    for i in range(cpu_count() - 1):
        p = Process(target=worker, args=(todo,))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()


# LEGACY below #####################################################


def legacy_path_data(n, resolutions):
    """
    [REFACTOR: This function was replaced by path_data function, as loading models and difference maps was no
    longer needed. Also the function replaces the hardcoded paths by loading everything from desired directory.]

    Generate pandas dataframe containing file paths for models, maps, labelled maps and difference maps for
    as many resolutions as input, using pdb name as an index.
    Tested

    :param n: number of maps to keep in dataframe
    :param resolutions: resolutions to use
    """
    # REVIEW: remove hardcoded paths
    map_paths = []
    for i in resolutions:
        # map_paths_subset = glob.glob("/home/ccpem/jgreer/software/density_map_dataset/synthetic/{}/*.mrc".format(i))
        map_paths_subset = glob.glob("/home/ccpem/jgreer/software/density_map_dataset/rescaled/synthetic/{}/*.mrc"
                                     .format(i))
        # map_paths_subset = glob.glob("/Users/jg13994/Documents/RAL_Placement/joel_test/chim-maps/{}/*.mrc".format(i))
        map_paths.append(sorted(map_paths_subset))
    model_paths = glob.glob("/home/ccpem/jgreer/software/density_map_dataset/synthetic/pdbs/*.cif")
    # model_paths = glob.glob("/Users/jg13994/Documents/RAL_Placement/joel_test/pdbs/*.cif")
    model_paths = sorted(model_paths)
    # sorting should have ordered everything alphabetically which should be okay to match columns up

    # get paths of labelled maps (these will be the paths if they don't exist already)
    labelled_maps = []
    diff_maps = []
    # REVIEW: use enumerate for indices, zip for multiple lists
    for thisres, i in zip(resolutions, range(len(resolutions))):
        reslabelled_maps = []
        resdiff_maps = []
        for path in map_paths[i]:
            # print('path:{}'.format(path))
            lab_map = path.split('/')
            lab_map[-2] += 'label'
            lab_map = '/'.join(lab_map)
            reslabelled_maps.append(lab_map)

            # paths to save difference maps
            diff_map = path.split('/')
            diff_map[-2] += 'predictdiff'
            diff_map = '/'.join(diff_map)
            resdiff_maps.append(diff_map)
        labelled_maps.append(reslabelled_maps)
        diff_maps.append(resdiff_maps)

    # convert to np arrays for vectorized operations
    model_paths = np.array(model_paths)
    map_paths = np.array(map_paths)
    labelled_maps = np.array(labelled_maps)
    diff_maps = np.array(diff_maps)

    # we have one set of models for multiple sets of maps
    # lets make a df out of it all
    # row is model,
    # column is model path,resolution_1 path,....,etc
    # make label of df the mrc file prefix
    labels = []
    for path in model_paths:
        label = path.split('/')[-1]
        # print(label)
        label = label[:-4]
        # print(label)
        labels.append(label)
    # print(len(labels))
    print('Check first 5 labels look good: {}'.format(labels[0:5]))
    # labels are our df index!

    # shorten to use first n PDB entries only
    labels = labels[:n]
    model_paths = model_paths[:n]
    # for (path,labpath) in zip(map_paths,labelled_maps):
    #     print(path[:n])
    #     print(labpath[:n])
    #     path=path[:n]
    #     labpath=labpath[:n]
    map_paths = map_paths[:, :n]
    labelled_maps = labelled_maps[:, :n]
    diff_maps = diff_maps[:, :n]
    # print(map_paths)
    """
    print(len(map_paths))
    for resn in range(len(resolutions)):
        print(len(map_paths[resn]))
        print('\r')
    #print(model_paths)
    print(len(model_paths))
    """

    # make the lists to create a pd df out of
    # labels already is okay!
    cols = ['model']  # column names
    df_data = [list(model_paths)]
    # add to these lists for each resolution
    for resn in range(len(resolutions)):
        df_data.append(list(map_paths[resn]))  # add data for res
        df_data.append(list(labelled_maps[resn]))  # add lab_data for res
        df_data.append(list(diff_maps[resn]))
        cols.append('{}map'.format(resolutions[resn]))
        cols.append('{}labmap'.format(resolutions[resn]))
        cols.append('{}diffmap'.format(resolutions[resn]))

    """
    print('resolutions:{}'.format(resolutions))
    print('df_data shape:{}'.format(np.array(df_data).shape))
    for df_data_cntr in range(len(df_data)):
        print('df_data[{}]{}'.format(df_data_cntr,df_data[df_data_cntr]))
    print('labels:{}'.format(labels))
    print('col:{}'.format(cols))
    """

    """
    path_df=pd.DataFrame(list(zip(model_paths,map_paths[0],labelled_maps[0])),
        index=labels,
        columns =['model','{}map'.format(resolutions[0]),'{}labmap'.format(resolutions[0])]) 
    """
    path_df = pd.DataFrame(np.array(df_data).T,
                           index=labels,
                           columns=cols)

    # print('Check path df')
    # print(path_df)

    return path_df


def legacy_path_data_from_list(labmaplist, mrcmaplist, modellist, n, resolutions):
    """
    [REFACTOR: This function was replaced by path_data function, which loads all data from directory rather than
    loading only what is contained in the list.]

    Create a pandas dataframe with filepaths for each labelled maps, density maps and models
    Untested

    Could remove models from this. User inputs full filepath of .txt files which contain the full filepaths of the
    labelled maps, the density maps and the models.
    :param labmaplist: labelled map .txt file path
    :param mrcmaplist: density map .txt file path
    :param modellist: models .txt file path
    :param n: number of maps to put into dataframe
    :param resolutions: resolutions to use - each should correspond to a directory containing maps or labelled
     maps for a single resolution
    :return: pandas dataframe
    """
    # use this version if you have:
    # text file with labelled map paths
    # text file with mrc map paths
    # text file with model map paths
    # all should have the same pdbs in. They will be sorted before being used
    # assumed no resolution information is required

    map_paths = [line.rstrip('\n') for line in open(mrcmaplist)]
    labelled_maps = [line.rstrip('\n') for line in open(labmaplist)]
    model_paths = [line.rstrip('\n') for line in open(modellist)]

    # now should have a list of map,labmap and models
    # so rearrange all alphabetically to hopefully get pdb indexes in same order
    map_paths = sorted(map_paths)
    labelled_maps = sorted(labelled_maps)
    model_paths = sorted(model_paths)

    # assume no resolution information required - different path_data

    # generate fpaths to save difference maps (generated during classification with trained unet)
    diff_maps = []
    for path in map_paths:
        # paths to save difference maps
        d_map = path.split('/')
        d_map[-2] += 'predictdiff'
        d_map = '/'.join(d_map)
        diff_maps.append(d_map)

    # convert to np arrays for vectorized operations
    model_paths = np.array(model_paths)
    map_paths = np.array(map_paths)
    labelled_maps = np.array(labelled_maps)
    diff_maps = np.array(diff_maps)

    # we have one set of models for multiple sets of maps
    # lets make a df out of it all
    # row is model,
    # column is model path,resolution_1 path,....,etc
    # make label of df the mrc file prefix
    labels = []
    for path in model_paths:
        label = path.split('/')[-1]
        # print(label)
        label = label[:-4]
        # print(label)
        labels.append(label)
    # print(len(labels))
    print('Check first 5 labels look good: {}'.format(labels[0:5]))
    # labels are our df index!

    # shorten to use first n PDB entries only
    labels = labels[:n]
    model_paths = model_paths[:n]
    # for (path,labpath) in zip(map_paths,labelled_maps):
    # print(path[:n])
    # print(labpath[:n])
    # path=path[:n]
    # labpath=labpath[:n]
    map_paths = map_paths[:, :n]
    labelled_maps = labelled_maps[:, :n]
    diff_maps = diff_maps[:, :n]

    # make the lists to create a pd df out of
    # labels already is okay!
    cols = ['model']  # column names
    df_data = [list(model_paths), list(map_paths), list(labelled_maps), list(diff_maps)]
    cols.append('{}map'.format(resolutions))
    cols.append('{}labmap'.format(resolutions))
    cols.append('{}diffmap'.format(resolutions))

    path_df = pd.DataFrame(np.array(df_data).T,
                           index=labels,
                           columns=cols)
    return path_df


def legacy_make_save_labels(path_df, n, resolutions):  # plottype,definemodel,train,classifier,
    """
    [REFACTOR: To save RAM, this function was replaced with gen_labs function, that rather than loading all of the data
     and only then generating the labels, instead loads data and generates labels in place, sample by sample,
     saving it straight to the disk. Also, gen_labs function has been designed to work with multiprocessing.]

    [REFACTOR: This refactor also renders load_data and atm_to_map_make redundant, ans this is the only use
    of these functions, so they have also been moved to legacy.]

    Make density maps where densities are instead integer labels for different residues
    Tested

    :param path_df -- pandas df relating PDB label to filepaths
    :param n -- number of PDB entries to use (first n from the path_df dataframe)
    :param resolutions -- the list of resolutions you want to loop over and use

    There is no returned data as atm_to_map_make is used to save the labelled maps to disk (to the filepaths defined in
    path_df
    """
    # resolutions=np.array([4.2])

    mmap, origin, size, cells, x, y, z, atm, res = legacy_load_data(path_df,
                                                                    resolutions)  # contains load_model and load_map
    # labelled is list of [n][n_res][amap_here!]
    legacy_atm_to_map_make(mmap, x, y, z, atm, res, origin, size, cells, n, resolutions,
                           path_df)  # make the labelled maps for all the maps/models you loaded in


def legacy_load_data(path_df, resolutions):
    """
    [REFACTOR: This function became redundant when make_save_labels was replaced by multiprocessing-compatible
     gen_labs function, that generates and saves labels in place to avoid bulk loading large datasets.]

    Loads data from .mrc file
    Tested

    :param path_df:
    :param resolutions:
    :return:
    """
    # load in all (or a set amount of the maps/models)
    # put them in lists with the right shape
    # np.shape(n,100,100,100,1)
    # n is number of rows from path_df loaded in

    # for loading maps
    mmap = []
    origin = []
    size = []
    cells = []

    # for loading models
    x = []
    y = []
    z = []
    atm = []
    res = []

    for index, row in path_df.iterrows():
        # create a list to hold map for each resolution
        # eventual np array will be like [n][res]...

        mmap_res = []
        origin_res = []
        size_res = []
        cells_res = []
        for j in resolutions:
            reslabel = '{}map'.format(j)
            mmap_res_temp, origin_res_temp, cells_res_temp, size_res_temp = load_map(row[reslabel], True)
            mmap_res.append(mmap_res_temp)
            origin_res.append(origin_res_temp)
            size_res.append(size_res_temp)
            cells_res.append(cells_res_temp)
        # now load the model
        # np array will be like [n]...
        x_temp, y_temp, z_temp, atm_temp, res_temp = load_model(row['model'])

        # make into lists of length n
        mmap.append(mmap_res)
        origin.append(origin_res)
        size.append(size_res)
        cells.append(cells_res)

        x.append(x_temp)
        y.append(y_temp)
        z.append(z_temp)
        atm.append(atm_temp)
        res.append(res_temp)

    # don't make anything into a np arr as every mmap is a different shape!!!
    # convert all to np.arrays and pass them out
    """
    mmap=np.array(mmap)
    print('mmap shape:{}'.format(mmap.shape))
    origin=np.array(origin)
    size=np.array(size)
    cells=np.array(cells)
    """

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    atm = np.array(atm)
    res = np.array(res)
    # print('atm shape:{}'.format(atm.shape))

    """
    print('len start')
    print(len(mmap))
    print(len(mmap[0]))
    print(len(mmap[0][0]))
    print(len(mmap[0][0][0]))
    print(len(mmap[0][0][0][0]))
    print(len(atm))
    print(len(atm[0]))
    print(len(atm[0][0]))
    """

    """
    print('np.shape start')
    print(mmap.shape)
    print(mmap[0].shape)
    print(mmap[1][2].shape)
    print(mmap[0][0][0].shape)
    print(mmap[0][0][0][0].shape)
    print(atm.shape)
    print(atm[0].shape)
    print(atm[0][0].shape)
    """

    return mmap, origin, size, cells, x, y, z, atm, res


def legacy_atm_to_map_make(mmap, x, y, z, atm, res, origin, size, cells, n, resolutions, path_df):
    """
    [REFACTOR: This function became redundant when make_save_labels was replaced by multiprocessing-compatible
     gen_labs function, that generates and saves labels in place to avoid bulk loading large datasets.]

    Wrapper around atm_to_map looping over all the resolutions
    Tested

    :param mmap:
    :param x:
    :param y:
    :param z:
    :param atm:
    :param res:
    :param origin:
    :param size:
    :param cells:
    :param n: total number of maps in dataset
    :param resolutions:
    :param path_df:
    :return:
    """
    # list of amaps
    # we want one amap for each model we have
    # so shape will be [n][res]...
    # try to make the appropriate directory to save each res to:
    for thisres in resolutions:
        # get the right filepath to create
        thisreslabel = '{}labmap'.format(thisres)
        fp = path_df.iloc[0][thisreslabel]
        fp = fp.split('/')
        # print('fp:{}'.format(fp))
        fp = '/'.join(fp[:-1])
        try:
            print('fp:{}'.format(fp))
            os.mkdir(fp)
        except OSError:
            print('Directory for res={}A labelled maps already exists - continuing'.format(thisres))

    amap = []
    # same length/size arrays
    # for each model
    for i in tqdm(range(n), desc='Making labelled maps'):
        # print('n={}'.format(n))
        # create an array for atmmaps of each rs for 1 model
        amap_res = []
        # iterate over all resolutions can use [0][0] as all maps have same no of resolutions
        for j, thisres in zip(range(len(resolutions)), resolutions):
            print('Map {} of {}\nRes {} of {}'.format(i + 1, n, j + 1, len(resolutions)))
            # make the atm map (same model for each map)
            amap_res_temp = atm_to_map(mmap[i][j], x[i], y[i], z[i], atm[i], res[i], origin[i][j], size[i][j],
                                       cells[i][j], res=thisres)
            # save this map with the right name
            thisreslabel = '{}labmap'.format(thisres)
            save_map(amap_res_temp, origin[i][j], cells[i][j], path_df.iloc[i][thisreslabel], True)
            # add it to the list
            amap_res.append(amap_res_temp)
        # add list of amaps for each res to list of amaps
        amap.append(amap_res)

    # must be a list due to variable size
    """
    #return np array of amaps
    amap=np.array(amap)
    print('amap shape:{}'.format(amap.shape))
    """

    return amap


"""
def manipulate_dataset(dmPath, lmPath ):

    Early attempt at what is now make_np_dataset() - untested
    Maybe should be removed

    Take a density map and a labelled map as inputs and cut out the volume which we are interested in

    :param dmPaths: (str[]) density map path to generate glob list from
    :param lmPaths: (str[]) labelled density map path to generate glob list from
    :param

    #manipulate file paths to get a path for glob'ing
    dmPath=str(dmPath).split('/')
    dmPath=dmPath[:-1]
    dmPath='/'.join(dmPath)
    dmPath=dmPath+'/*'
    dmList=tf.data.Dataset.list_files(str(dmPath),shuffle=False)

    #manipulate file paths to get a path for glob'ing
    lmPath=str(lmPath).split('/')
    lmPath=lmPath[:-1]
    lmPath='/'.join(lmPath)
    lmPath=lmPath+'/*'
    lmList=tf.data.Dataset.list_files(str(lmPath),shuffle=False)

    dmPath=next(iter(dmList))
    lmPath=next(iter(lmList))
    print('Before:')
    print(dmPath)
    print(lmPath)

    print('Apply Mapping Function:')
    dmList=dmList.map(lambda x: tf.py_function(manipulate_maps_dataset, [x], [tf.string]))
    lmList=lmList.map(lambda x: tf.py_function(manipulate_maps_dataset, [x], [tf.string]))
    print('End of Mapping Function')

    #have tf lists of file paths in each dataset
    #need to check they are in same order
    print('Density Map List')
    print(dmList)
    print('\n')
    print('Labelled Density Map List')
    print(lmList)
    print('\n')

    #manipulate the dataset and get out a density map and its
    #associated labelled map

    #do for 1 map for now, later can apply to all if it works
    #repurpose lmPath and dmPath
    #dmPath=next(iter(dmList))
    #lmPath=next(iter(lmList))
    #manipulate_maps_dataset(dmPath,lmPath)

    #save these altered maps - look at in Chimera
    #is the correct section saved?
    return


def manipulate_maps_dataset(thisfile):

    Early attempt at what is now make_np_dataset() - untested
    Maybe should be removed 

    print(thisfile)
    #dmFile=tf.strings.as_string(dmFile)
    #lmFile=tf.strings.as_string(lmFile)
    print("fpath:")
    print(bytes.decode(thisfile,),type(bytes.decode(thisfile)))

    print('STR converts:')
    print(thisfile)
    print(thisfile)
    img,orig,cella,sample=load_map(thisfile,False)#false or true? needs checking
    #l_img,l_orig,l_cella,l_sample=load_map(lmFile,False)
    #now make the necessary cuts

    print('Whats in a map?')
    print('img')
    print(img.shape)
    print('\n')
    print('orig')
    print(orig.shape)
    print('\n')
    print('cella')
    print(cella.shape)
    print('\n')
    print('sample')
    print(sample.shape)
    print('\n')
    return thisfile
"""
