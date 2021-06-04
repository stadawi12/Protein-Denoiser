import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def validate(test_data1, test_data2, device, unet):
    assert test_data1.shape == test_data2.shape
    
    batchNorm = torch.nn.BatchNorm3d(1, affine=False)
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


def create_directory(name_of_directory):
    original_length = len(name_of_directory)
    count = 0 
    old_directories = os.listdir("../out/")
    if name_of_directory in old_directories:
        while name_of_directory in old_directories:
            count += 1
            if len(name_of_directory) == original_length:
                name_of_directory += f"_{count}"
            else:
                name_of_directory = list(name_of_directory)
                name_of_directory[-1] = str(count)
                name_of_directory = "".join(name_of_directory)
        os.mkdir(f"../out/{name_of_directory}")
    else:
        os.mkdir(f"../out/{name_of_directory}")

    return name_of_directory


def save_model2(name_of_directory, epoch, unet):
    torch.save(unet.state_dict(),
            f"trained_models/{name_of_directory}/epoch_{epoch}.pt")


def save_plots(filename, trainingLosses, validationLosses):
    og_length = len(filename)
    count = 0
    lo_contents = os.listdir("losses")
    if filename+".png" in lo_contents:
        while filename+".png" in lo_contents:
            count += 1
            if len(filename) == og_length:
                filename += f"_{count}"
            else:
                filename = list(filename)
                filename[-1] = str(count)
                filename = "".join(filename)
        plt.plot(trainingLosses, label="Training Loss")
        plt.plot(validationLosses, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("MSE-Loss")
        plt.savefig("losses/"+filename+".png")
    else:
        plt.plot(trainingLosses, label="Training Loss")
        plt.plot(validationLosses, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("MSE-Loss")
        plt.savefig("losses/"+filename+".png")
