import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def validate(test_data1, test_data2, device, unet):
    assert test_data1.shape == test_data2.shape
    
    # batchNorm = torch.nn.BatchNorm3d(1, affine=False)
    n_tiles = test_data1.shape[0]
    losses = []
    # run validation
    with torch.no_grad():
        for j in range(0, n_tiles, 3): 
            test_batch1 = test_data1[j:j+3,0:1,:,:,:]
            test_batch1 = test_batch1.to(device)
            test_batch2 = test_data2[j:j+3,0:1,:,:,:]
            # test_batch2 = batchNorm(test_batch2)
            test_batch2 = test_batch2.to(device)
            out = unet(test_batch1)
            loss_val = F.mse_loss(out, test_batch2)
            losses.append(loss_val.item())
    sum_of_losses = sum(losses)
    epoch_loss = sum_of_losses / len(losses)
    print(f"Validation Loss: {epoch_loss}")
    return epoch_loss


def create_directory(out_path, name_of_directory):
    original_length = len(name_of_directory)
    count = 0 
    old_directories = os.listdir(out_path)
    if name_of_directory in old_directories:
        while name_of_directory in old_directories:
            count += 1
            if len(name_of_directory) == original_length:
                name_of_directory += f"_{count}"
            else:
                name_of_directory = list(name_of_directory)
                name_of_directory[-1] = str(count)
                name_of_directory = "".join(name_of_directory)
        os.mkdir(os.path.join(out_path, name_of_directory))
    else:
        os.mkdir(os.path.join(out_path, name_of_directory))

    return name_of_directory


def save_model2(out_path, name_of_directory, epoch, unet):
    path_to_data = os.path.join(out_path, name_of_directory)
    if 'models' not in os.listdir(path_to_data):
        os.mkdir(os.path.join(path_to_data, 'models'))
    torch.save(unet.state_dict(),
            os.path.join(out_path, name_of_directory, 'models',
                f'epoch_{epoch}.pt'))


def plot_losses(plot_data : dict):
    """Constructs training vs validation loss curves.

    Parameter
    ---------
    plot_data : dict
        Dictionary containing the two datasets, training losses
        and validation losses.

    Returns
    -------
    fig : Figure
        Object Figure which can be used to     

    """
    validationLosses = plot_data["validationLosses"]
    trainingLosses = plot_data["trainingLosses"]
    fig, ax = plt.subplots()
    ax.plot(trainingLosses, label="Training Loss")
    ax.plot(validationLosses, label="Validation Loss")
    ax.legend(loc="upper right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    return fig
    

def save_plot(out_path, local_path, plot_data, epoch):
    """Saves plot to ../out/plots/e_{epoch}.png

    Parameters
    ----------
    local_path : str
        name of the training directory containing all output
        files of the trainig network
    plot_data  : dict
        Dictionary containing training and validation loss
        data
    epoch      : int
        The epoch that we are currently in.
            
    """
    # Global ../out/ directory for all different training runs
    path_to_plots = os.path.join(out_path,local_path,'plots')

    if 'plots' not in os.listdir(
            os.path.join(out_path,local_path)):
        os.mkdir(path_to_plots)

    fig = plot_losses(plot_data)
    path_of_plot = os.path.join(path_to_plots,f"e_{epoch+1}.png")
    fig.savefig(path_of_plot)
