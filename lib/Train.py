# Standard library imports
import sys
sys.path.insert(1, 'utils/ml_toolbox/src/')
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch
import yaml
import time
import os

# Custome module imports
from utils.DataLoader import Data
import utils.utils_train as ut
from net import unet
from parsers import train_parser

def Train(Network, input_data, data_path='data', out_path='out'):
    """Trains a noise2noise model on protein half-maps

    Parameters
    ----------
    data_path : str
        Path to the data folder
    input_data : dict
        A dictionary containing all necessary variables

    """
    # Paths to trainig and test data
    input_path = os.path.join(data_path, '1.0')
    trget_path = os.path.join(data_path, '2.0')
    test1_path = os.path.join(data_path, '1.5')
    test2_path = os.path.join(data_path, '2.5')

    # Objects for dealing with training data 
    input_maps = Data(input_path)
    trget_maps = Data(trget_path)

    # Objects for dealing with testing data 
    test1_data = Data(test1_path)
    test2_data = Data(test2_path)

    # Input data
    learning_rate = input_data["lr"]
    no_of_batches = input_data["no_of_batches"]
    norm_vox_lim  =(input_data["norm_vox_min"],
                    input_data["norm_vox_max"])
    loss_index    = input_data["loss_index"]
    norm_vox      = input_data["norm_vox"]
    cshape        = input_data["cshape"]
    device        = torch.device(input_data["device"])
    epochs        = input_data["epochs"]
    tail          = input_data["tail"]
    mbs           = input_data["mbs"] # Mini batch size

    # If no_of_batches=0, use all batches, else, use custom amount
    if no_of_batches == 0:
        no_of_batches = len(input_maps.ls)
    else:
        no_of_batches = no_of_batches

    # Model (UNET)
    unet = Network.UNet()
    unet = unet.to(device)

    # Optimiser
    optimiser = optim.Adam(unet.parameters(), lr=learning_rate)

    # start timer
    tic = time.perf_counter()

    # My implementation of a custom loss function
    def my_loss(output, target, power):
        loss = torch.mean((torch.abs(output - target))**power)
        return loss

    # CREATE FOLDER FOR OUTPUT FILES IN ../out/
    # Get number of maps being trained on
    number_of_maps = 0
    for i in range(no_of_batches):
        number_of_maps += input_maps.batch_sizes[i]
    # Construct name of directory
    dir_name = f"m{number_of_maps}_e{epochs}_mbs{mbs}"
    # Add tail to directory if specified in arguments
    dir_name = dir_name + tail
    # Generate output directory
    dir_name = ut.create_directory(out_path, dir_name)


    # Bins for storing computed losses and validation losses
    trainingLosses = []
    validationLosses = []

    # BEGIN TRAINING LOOP
    # For each epoch
    for e in range(epochs):
        
        # Create a bin for storing epoch training losses
        trainingLoss = []

        # For each batch
        for b in range(no_of_batches):

            # generate validation loss before starting each epoch
            if b == 0:
                # Load testing batch
                test1_data.load_batch(0, 
                        cshape=cshape,
                        norm_vox=norm_vox,
                        norm_vox_lim=norm_vox_lim)
                test2_data.load_batch(0, 
                        cshape=cshape,
                        norm_vox=norm_vox,
                        norm_vox_lim=norm_vox_lim)

                # Obtain validation loss
                validationLoss = ut.validate(test1_data.tiles, 
                        test2_data.tiles, device, unet)
                # Append validation loss to validation losses
                validationLosses.append(validationLoss)
            
            # Load correspodning batch of training maps
            # Data().tiles contains tiles of maps already
            input_maps.load_batch(b,
                        cshape=cshape,
                        norm_vox=norm_vox,
                        norm_vox_lim=norm_vox_lim)
            trget_maps.load_batch(b,
                        cshape=cshape,
                        norm_vox=norm_vox,
                        norm_vox_lim=norm_vox_lim)

            # Pass mbs tiles through network
            for i in range(0, input_maps.tiles.shape[0], mbs):
                
                # Grab mbs number of input tiles
                x = input_maps.tiles[i:i+mbs, 0:1, :, :, :]
                x = x.to(device)
                # Grab mbs number of target tiles 
                y = trget_maps.tiles[i:i+mbs, 0:1, :, :, :]
                y = y.to(device)
                # Pass input tiles through network
                unet.zero_grad()
                out = unet(x)

                # Pick loss function
                if loss_index == 0:
                    # Calculate torch loss
                    loss = F.mse_loss(out ,y)
                else:
                    # Calculate custom loss
                    loss = my_loss(out, y, loss_index)

                # Append training loss to bin
                trainingLoss.append(loss.item())
                # Print loss
                print(
                    f'epoch: {e+1}/{epochs}, ' +
                    f'batch: {b+1}/{no_of_batches}, ' +
                    f'tiles: {i}/{len(input_maps.tiles)}, ' + 
                    f'loss: {loss:.6}, '
                    )

                # Backward propagation
                loss.backward()
                optimiser.step()

        # OBTAIN AVERAGE OF TRAINING LOSS FOR THE EPOCH
        sum_of_trainingLosses = sum(trainingLoss)
        epoch_trainingLoss = sum_of_trainingLosses / len(trainingLoss)
        trainingLosses.append(epoch_trainingLoss)
        
        # At end of EPOCH, save model to ../out
        ut.save_model2(out_path, dir_name, e+1, unet)
        
        # For each EPOCH save plots of training vs validation
        # Create dictionary of loss data
        plot_data = {"validationLosses": validationLosses,
                     "trainingLosses"  : trainingLosses}
        # save plot in ../out/plots/
        ut.save_plot(out_path, dir_name, plot_data, e)


    # end timer
    toc = time.perf_counter()

    return print(f"time: {(toc-tic) / 60}m")

if __name__ == '__main__':

    from utils.Inputs import Read_Input
    from net import unet
    

    input_data = Read_Input('../inputs.yaml')
    Train(unet, input_data, out_path='../out', data_path='../data')

