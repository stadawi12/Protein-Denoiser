import os

import warnings
warnings.filterwarnings("ignore")

# TEST PATHS
test_pdb = "tests/data/label_data/pdbs/"
test_mrc = "tests/data/train_data/2.5/"
test_lab = "tests/data/train_data/2.5label/"
pdb_bb = os.path.join(test_pdb, "amino_4.pdb")
map_bb = os.path.join(test_mrc, "amino_4.mrc")

label_path = 'tests/data/label_data'
train_path = 'tests/data/train_data'


# MPI labels generating example (needs main so uncomment to run)
# from src.multi_proteins import mpi_gen_labs
# if __name__ == "__main__":
#     mpi_gen_labs(path)


def labgen_example():
    # python3 ml-protein-toolbox.py -d ~/mydata/genlabs/ -r 2.5-u makelabels -c custom

    from src.multi_proteins import gen_labs

    gen_labs(label_path, res=[2.5], gen_maps=False)


def data_load_process_save_example(mrcpath=map_bb, pdbpath=pdb_bb):

    from src.proteins import load_map, load_model, atm_to_map, amino_map, save_map

    # load data and generate labels
    mmap, orig, sample, cell = load_map(mrcpath)    # load mrc file
    x, y, z, atm, res, bb, occ = load_model(pdbpath)    # load pdb model
    _, amap = atm_to_map(mmap, x, y, z, atm, res, orig, sample, cell, res=2.5)  # generate labels

    # from src.viewer import plot_labels
    # plot_labels(amap, x, y, z, atm, orig, sample, cell)     # plot labels (only with mayavi)

    # save labels
    pdbid = mrcpath.split('/')[-1].split('.')
    pdbid = pdbid[0] + '_backbone.' + pdbid[1]
    save_map(amap, orig, cell, path=os.path.join(test_lab, pdbid), overwrite=True)

    # save labels as individual mrc files for ease of display in Chimera / Coot
    aminopath = os.path.join(test_lab, mrcpath.split('/')[-1])
    amino_map(amap, orig, cell, path=aminopath, overwrite=True)


def data_processing_scale_example():
    # python3 ml-protein-toolbox.py -d ~/mydata/training_data -r 2.5 -n 2 -p scale -s 64

    from src.multi_proteins import load_data, preproces_data, query_data

    res = [2.5]

    data = load_data(train_path, res, n=2)
    query_data(data)
    data = preproces_data(data, mode='scale', cshape=64)
    query_data(data)


def data_processing_tile_example():
    # python3 ml-protein-toolbox.py -d ~/mydata/training_data -r 2.5 -n 2 -p tile -s 64

    from src.multi_proteins import load_data, preproces_data, query_data, postproces_data

    res = [2.5]

    data = load_data(train_path, res, n=2)
    data = preproces_data(data, mode='tile', cshape=64)
    query_data(data)
    data = postproces_data(data, map=True)
    # [d.save_map(map_rec=True) for d in data]


def data_analysis_example():
    # python3 ml-protein-toolbox.py -d ~/mydata/training_data -r 2.5 -p crop -s 64 -u predict

    from src.multi_proteins import load_data, preproces_data
    from src.analysis import train_unet, predict_unet

    data = load_data(train_path, [2.5])
    data = preproces_data(data, mode='tile', cshape=64, background=0.5, background_label=0)

    # train_unet(data, lab_names=['bg','bb','sc'], epochs=100, batch=1, loss_mode='scce')
    predict_unet(data, model_name="tests/data/train_data/model_64x64x64_epoch0039_checkpoint.h5")


data_analysis_example()
