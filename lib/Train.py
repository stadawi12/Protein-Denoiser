# Standard library imports
import sys
sys.path.insert(1, 'utils/ml_toolbox/src/')
sys.path.insert(1, 'utils')
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import yaml
import time
import os
import math
from shutil import copyfile

# Custome module imports
from Inputs import Read_Input
import utils.utils_train as ut
from net import unet
from parsers import train_parser
from loader import dataset, collate_fn

def Train(Network, inputs_path='inputs.yaml',
        data_path='data', out_path='out'):

    """Trains a noise2noise model on protein half-maps

    Parameters
    ----------
    data_path : str
        Path to the data folder
    input_data : dict
        A dictionary containing all necessary variables

    """
    input_data = Read_Input(inputs_path)

    # Input data
    learning_rate = input_data["lr"]
    loss_index    = input_data["loss_index"]
    batch_size    = input_data["batch_size"]
    validate      = input_data["validate"]
    shuffle       = input_data["shuffle"]
    device        = torch.device(input_data["device"])
    epochs        = input_data["epochs"]
    tail          = input_data["tail"]
    mbs           = input_data["mbs"] # Mini batch size

    #==============================================
    # TRAINING DATA -------------------------------
    #==============================================
    training_set = dataset(
            input_data, 
            path_global = data_path,
            data_set    = "Training")
    # Training data generator
    training_gen = DataLoader(
            training_set, 
            batch_size = batch_size,
            shuffle    = shuffle,
            collate_fn = collate_fn)

    #==============================================
    # VALIDATIONA DATA ----------------------------
    #==============================================
    validation_set = dataset(
            input_data, 
            path_global = data_path,
            data_set    = "Validation")
    # Training data generator
    validation_gen = DataLoader(
            validation_set, 
            batch_size = batch_size,
            shuffle    = shuffle,
            collate_fn = collate_fn)
    #===============================================

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
    number_of_maps = len(training_set)
    # Construct name of directory
    dir_name = f"m{number_of_maps}_e{epochs}_mbs{mbs}"
    # Add tail to directory if specified in arguments
    dir_name = dir_name + tail
    # Generate output directory
    dir_name = ut.create_directory(out_path, dir_name)

    # Copy input data into out/ directory
    copyfile(inputs_path, os.path.join(out_path, dir_name,
        'inputs.yaml'))


    # Bins for storing computed losses and validation losses
    trainingLosses = []
    validationLosses = []

    # BEGIN TRAINING LOOP
    for e in range(epochs):
        
        # Create a bin for storing epoch training losses
        trainingLoss = []

        #=================================================
        # TRAINING ---------------------------------------
        #=================================================
        b_count = 0 
        for inpt_tiles, trgt_tiles in training_gen:
            
            # Ensure input and target shapes are same
            assert inpt_tiles.shape == trgt_tiles.shape, \
                    "WARNING: Shapes do NOT match"
            
            # Pass mbs tiles through network
            for i in range(0, inpt_tiles.shape[0], mbs):
                
                # Grab mbs number of input tiles
                x = inpt_tiles[i:i+mbs, 0:1, :, :, :]
                x = x.to(device)
                # Grab mbs number of target tiles 
                y = trgt_tiles[i:i+mbs, 0:1, :, :, :]
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
                    f'batch: {b_count+1}/' +
                f'{math.ceil(len(training_set)/batch_size)}, ' +
                    f'tiles: {i}/{inpt_tiles.shape[0]}, ' + 
                    f'loss: {loss:.6}, '
                    )

                # Backward propagation
                loss.backward()
                optimiser.step()

            b_count += 1


        # ==================================================
        # VALIDATION----------------------------------------
        # ==================================================
        if validate:
            # Bin to hold validation losses
            valLosses = []
            with torch.set_grad_enabled(False):

                # loding validation data from generator
                for inpt_tiles, trgt_tiles in validation_gen:

                    # Ensure input and target shapes are same
                    assert inpt_tiles.shape == trgt_tiles.shape, \
                            "WARNING: Shapes do NOT match"

                    for i in range(0, inpt_tiles.shape[0], mbs):
                        x = inpt_tiles[i:i+mbs,:,:,:,:]
                        x = x.to(device)
                        y = trgt_tiles[i:i+mbs,:,:,:,:]
                        y = y.to(device)
                        out = unet(x)
                        # TODO make loss function global
                        # Pick loss function
                        if loss_index == 0:
                            # Calculate torch loss
                            loss = F.mse_loss(out ,y)
                        else:
                            # Calculate custom loss
                            loss = my_loss(out, y, loss_index)
                        valLosses.append(loss)

            # Calculate average of valLosses
            valLosses = sum(valLosses) / len(valLosses)

        # OBTAIN AVERAGE OF TRAINING LOSS FOR THE EPOCH
        sum_of_trainingLosses = sum(trainingLoss)
        epoch_trainingLoss = \
                sum_of_trainingLosses / len(trainingLoss)
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

