import os

import warnings
warnings.filterwarnings("ignore")

# TEST PATHS
test_pdb = "test_data/pdbs/"
test_mrc = "test_data/mrcs/"
test_lab = "test_data/labs/"
pdb_bb = os.path.join(test_pdb, "bb_aligned.pdb")
pdb_bb1 = os.path.join(test_pdb, "bb_amino_1.pdb")
pdb_bb2 = os.path.join(test_pdb, "bb_amino2.pdb")
pdb_bb7 = os.path.join(test_pdb, "bb_amino_7.pdb")
pdb_amino4 = os.path.join(test_pdb, "amino_4.pdb")
pdb_cmo = os.path.join(test_pdb, "cmo.pdb")
pdb_h2o = os.path.join(test_pdb, "h2o.pdb")
map_bb = os.path.join(test_mrc, "bb_aligned.mrc")
map_bb1 = os.path.join(test_mrc, "bb_amino_1.mrc")
map_bb2 = os.path.join(test_mrc, "bb_amino2.mrc")
map_bb7 = os.path.join(test_mrc, "bb_amino_7.mrc")
map_amino4 = os.path.join(test_mrc, "amino_4.mrc")
map_cmo = os.path.join(test_mrc, "cmo.mrc")
map_h2o = os.path.join(test_mrc, "h2o.mrc")

_laptop = "/home/jola/mydata/training_data/"
_workstation = "/home/sxy26921/data/"

global_path = _workstation if os.path.exists(_workstation) else _laptop

def custom_labgen_example():
    # python3 ml-protein-toolbox.py -d ~/mydata/testing_genlabs/ -r 2.5-u makelabels -c custom

    # from multi_proteins import gen_labs
    #
    # gen_labs("/home/jola/mydata/testing_genlabs", res=[2.5], custom_labs=True, gen_maps=False)
    from proteins import load_map, amino_map
    m, orig, sample, cell = load_map("/home/jola/mydata/training_data/6.5label/2xj7_bb.mrc")
    amino_map(m, orig, cell, "/home/jola/mydata/test.mrc")


def data_display_example(mrcpath=map_bb, pdbpath=pdb_bb):

    from proteins import load_map, load_model, atm_to_map, amino_map, save_map
    from viewer import plot_labels

    # load data and generate labels
    mmap, orig, sample, cell = load_map(mrcpath)    # load mrc file
    x, y, z, atm, res, bb, occ = load_model(pdbpath)    # load pdb model
    _, amap = atm_to_map(mmap, x, y, z, atm, res, orig, sample, cell, res=2.5)  # generate labels
    plot_labels(amap, x, y, z, atm, orig, sample, cell)     # plot labels

    # save labels
    pdbid = mrcpath.split('/')[-1]
    save_map(amap, orig, cell, path=os.path.join(test_lab, pdbid), overwrite=True)

    # save labels as individual mrc files for ease of display in Chimera / Coot
    aminopath = os.path.join(test_lab, mrcpath.split('/')[-1][:-4])
    if not os.path.exists(aminopath):
        os.mkdir(aminopath)
    amino_map(amap, orig, cell, path=aminopath, overwrite=True)


def data_processing_crop_example():

    from multi_proteins import load_data, preproces_data, query_data

    res = [2.5]

    data = load_data(global_path, res, n=30)
    [print(s.header_map.voxs) for s in data]
    data = preproces_data(data, mode='crop', cshape=(64, 64, 64))
    query_data(data)


def data_processing_tile_example():

    from multi_proteins import load_data, preproces_data, query_data, postproces_data

    res = [2.5]

    data = load_data(global_path, res, n=3)
    data = preproces_data(data, mode='tile', cshape=64)
    query_data(data)
    data = postproces_data(data, tiles=True)
    # [d.save_map(map_rec=True) for d in data]


def data_analysis_example():

    from multi_proteins import load_data, preproces_data
    from analysis import train_unet, predict_unet

    data = load_data(global_path, [2.5], n=10)
    # data = preproces_data(data, mode='crop', cshape=64)
    data = preproces_data(data) #, mode='crop', cshape=64)

    train_unet(data, [0, 1, 2], epochs=100, batch=1, loss_mode='scce')
    #predict_unet(data)

    

# labels generating example (needs main so uncomment to run)
# from multi_proteins import mpi_gen_labs
# if __name__ == "__main__":
#     mpi_gen_labs(global_path)

# custom_labgen_example()
# data_analysis_example()
data_processing_tile_example()

