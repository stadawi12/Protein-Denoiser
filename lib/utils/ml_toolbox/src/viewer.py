import argparse
from datetime import datetime
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import mrcfile as mrc

from utilsml import real_to_grid


def plot_labels(amap, x, y, z, atm, orig, sample, cell):
    """
    Display map labels and corresponding atomic coordinates.

    :param amap: (numpy.ndarray[int]) labelled electron density map with integer labels, correspond to e.g. amino acids
    :param x: (np.ndarray[float]) atomic x-coordinates
    :param y: (np.ndarray[float]) atomic y-coordinates
    :param z: (np.ndarray[float]) atomic z-coordinates
    :param atm: (np.ndarray[float]) atomic labels
    :param orig: (tuple(float, float, float)) cell origin
    :param sample: (tuple (int, int, int)) sampling rate along x, y and z axis
    :param cell: (tuple(float, float, float)) size of the cell in x, y and z dimensions
    :return:
    """
    from mayavi import mlab

    atms = [list(set(atm)).index(i) + 1 for i in atm]
    x, y, z = real_to_grid(x, y, z, orig, cell, sample)
    mlab.figure()
    mlab.points3d(x, y, z, atms, scale_mode='none', scale_factor=0.6)

    x, y, z = np.mgrid[0:amap.shape[0], 0:amap.shape[1], 0:amap.shape[2]].astype(np.float64)
    points = amap[amap != 0]
    x = x[amap != 0]
    y = y[amap != 0]
    z = z[amap != 0]
    mlab.points3d(x, y, z, points, colormap='Set2', scale_mode='none', scale_factor=0.6, opacity='0.332')
    mlab.contour3d(amap, contours=2, colormap="Set2", opacity=0.4)

    mlab.show()


def plot_3d(imgs, ptype):
    """
    General function for display of 3D images with mlab and/or plt.

    :param imgs: (numpy.ndarray or list) list/ndarray of 3D ndarray images or a singl 3D ndarray image
    :param ptype: (str) one of 'plt'-2D, 'mlab'-3D or 'both specifies type of display
    """

    if str(ptype) not in ['plt', 'mlab', 'both']:
        raise RuntimeError("Unknown plot type:{}".format(ptype))

    if not isinstance(imgs, list) and not isinstance(imgs, np.ndarray):
        raise RuntimeError("Invalid data type."
                           "Only accepted formats are a list/np.ndarray of 3D numpy.ndarrays,"
                           "or a single 3D numpy.ndarray.")
    else:
        if isinstance(imgs, list):
            imgs = np.asarray(imgs)
        if len(imgs.shape) == 3:  # single np.ndarray image
            imgs = np.array([imgs])
        elif len(imgs.shape) != 4:
            raise RuntimeError("Unexpected dimensionallity. Expected: 3, got:", len(imgs[0].shape),
                               "dimensional image.")

    if ptype == 'plt' or ptype == 'both':

        for im in imgs:

            plt.figure()
            plt.imshow(im[int(im.shape[0]/2), :, :], interpolation='nearest')
            plt.figure()
            plt.imshow(im[:, int(im.shape[0]/2), :], interpolation='nearest')
            plt.figure()
            plt.imshow(im[:, :, int(im.shape[0]/2)], interpolation='nearest')

            if len(imgs) > 3:
                plt.show()

        if len(imgs) <= 3:
            plt.show()

    if ptype == 'mlab' or ptype == 'both':

        for im in imgs:
            continue

            #mlab.figure()
            #mlab.contour3d(im)

        #mlab.show()


def confusion_matrix(y_true, y_pred, labs, lab_names=None,
                     show=False, save='pred.png', title="Confusion Matrix", ts=False):
    if lab_names is not None and len(lab_names) != len(labs):
        raise RuntimeError("Number of unique labels is not the same as the number of label names.")

    if show and save is not None:
        raise RuntimeError("Function 'confusion_matrix' can only show or save, not both at once.")
    if save is not None:
        if len(save.split('/')) > 1:
            sdir = ''.join(save.split('/')[:-1])
            save = ''.join(save.split('/')[-1])
        else:
            sdir = 'logs/confusion'
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        if ts:
            save_t = os.path.join(sdir, save)
        save = os.path.join(sdir, save[-7:])


    y_true = np.squeeze(y_true).ravel()
    conf = tf.math.confusion_matrix(labels=y_true, predictions=y_pred, num_classes=len(labs)).numpy()  # TODO move this to np and remove import
    conf = np.around(conf.astype('float') / conf.sum(axis=-1)[:, np.newaxis], decimals=2)
    conf = np.nan_to_num(conf)
    plt.imshow(conf, cmap=plt.cm.Blues, aspect='equal', origin='lower',
               extent=(-0.5, len(labs) - 0.5, -0.5, len(labs) - 0.5), vmin=0.0, vmax=1.0)
    plt.colorbar()
    names = lab_names if lab_names is not None else labs
    tick_marks = np.arange(len(labs))
    plt.xticks(tick_marks, names)
    plt.yticks(tick_marks, names, rotation=45)
    threshold = np.max(conf) * 0.9
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        color = "white" if conf[i, j] > threshold else "black"
        plt.text(j, i, conf[i, j], horizontalalignment="center", color=color)

    plt.ylabel('true')
    plt.xlabel('predicted')
    plt.title(title)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(save, format='png')
        if ts:
            plt.savefig(save_t, format='png')

    plt.clf()
    plt.close()


def plot_metric(x, loss_t, loss_v, acc_t, acc_v,  # loss_tst=None, acc_tst=None,
                title="Training and Validation Loss and Accuracy",
                save='train.png', colour_scheme='blues', mname='Accuracy', ts=False):

    color_modes = {'blues': ['tab:blue', 'tab:red'],
                   'greens': ['tab:green', 'tab:orange']}

    if colour_scheme not in color_modes.keys():
        raise RuntimeError("Parameter 'color_scheme' must be one of:\n    {}". format(color_modes.keys()))
    else:
        cl_scheme = color_modes[colour_scheme]

    if save is not None:
        if len(save.split('/')) > 1:
            sdir = ''.join(save.split('/')[:-1])
            save = ''.join(save.split('/')[-1])
        else:
            sdir = 'logs/metrics'
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        if ts:
            save_t = os.path.join(sdir, save[:-4] + datetime.now().strftime("%y%m%d-%H%M%S") + save[-4:])
        save = os.path.join(sdir, save)

    fig, ax1 = plt.subplots()
    color = cl_scheme[0]
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(x, loss_t, color=color, label='training loss', marker='o')
    ax1.plot(x, loss_v, color=color, label='validation loss', marker='v', linestyle='--')
    # if loss_tst is not None:
    #     ax1.plot(x, loss_tst, color=color, label='testing_loss', marker='^', linestyle=':')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = cl_scheme[1]
    ax2.set_ylabel(mname, color=color)
    ax2.plot(x, acc_t, color=color, label='training accuracy', marker='o')
    ax2.plot(x, acc_v, color=color, label='validation accuracy', marker='v', linestyle='--')
    # if loss_tst is not None:
    #     ax2.plot(x, acc_tst, color=color, label='testing_accuracy', marker='^', linestyle=':')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.grid()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(title)
    fig.tight_layout()
    if ts:
        plt.savefig(save_t)
    plt.savefig(save)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Image name")
    parser.add_argument("-p", type=str, help="plt|mlab|both")
    args = parser.parse_args()

    f = args.file
    t = args.p

    img = None
    if f[-3:] == str('npy'):
        img = np.load(f)
    elif f[-3:] == str('mrc'):
        mrcf = mrc.open(f)
        img = mrcf.data
        mrcf.close()

    if img is None:
        raise RuntimeError("Unrecognized file type. Only accepted files are: '.npy' and '.nii'.")

    if t is None:
        t = 'mlab'

    plot_3d(img, str(t))
