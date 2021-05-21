import sys
sys.path.insert(1, 'utils/')
sys.path.insert(1, 'utils/ml_toolbox/')
sys.path.insert(1, 'utils/ml_toolbox/src/')

 
# utils imports 
from unet import UNet
from csv_filter import get_entries
import utils_train as ut
from load import Data

# python specific libraries
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from   torch import optim
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int,
        help="Number of epochs to run for")
parser.add_argument('-mbs', '--minibatchsize', type=int,
        help="Mini batch size")
parser.add_argument('-dirs', '--numberofdirs', default=None,
        type=int,
        help="Number of batches to include in training")
parser.add_argument('-t', '--tail', default='', type=str,
        help='Tail you want to add to the directory name')
args = parser.parse_args()

# Check if machine has a GPU
if torch.cuda.is_available():
    dev = "cuda:0"
    device = torch.device(dev)
else:
    dev = "cpu"
    device = torch.device(dev)

# NOTE if I am tiling the maps then the size of the map
# should not matter.

# Model (UNET)
unet = UNet()
unet = unet.to(device)
# Optimiser
lr = 0.001  # learning rate
optimiser = optim.Adam(unet.parameters(), lr=lr)

EPOCHS = args.epochs
mbs = args.minibatchsize   # mini-batch-size

globalPath = 'data/downloads/' # global path to maps


d1 = Data(globalPath+'1.0/')
d2 = Data(globalPath+'2.0/')

test1 = Data(globalPath+'1.5/')
test2 = Data(globalPath+'2.5/')


# start timer
tic = time.perf_counter()

trainingLosses = []
validationLosses = []

if args.numberofdirs == None:
    numberofdirs = len(d1.ls)
else:
    numberofdirs = args.numberofdirs

def my_loss(output, target, power):
    loss = torch.mean((torch.abs(output - target))**power)
    return loss


number_of_maps = 0
for i in range(numberofdirs):
    number_of_maps += d1.batch_sizes[i]

dir_name = f"m{number_of_maps}_e{EPOCHS}_mbs{mbs}"
dir_name = dir_name + args.tail

dir_name = ut.create_directory(dir_name)


for e in range(EPOCHS):
    # powers = np.arange(2,4+0.0001,(4-2)/(EPOCHS-1))
    
    trainingLoss = []

    for b in range(numberofdirs):

        d1.load_batch(b)
        d2.load_batch(b)

        # Training loop
        for i in range(0, d1.tiles.shape[0], mbs):
            # generate validation loss before starting each epoch
            # if i == 0:
            #     validationLoss = ut.validate(test1, test2, 
            #             device, unet)
            #     validationLosses.append(validationLoss)

            x = d1.tiles[i:i+mbs, 0:1, :, :, :]
            x = x.to(device)
            y = d2.tiles[i:i+mbs, 0:1, :, :, :]
            y = y.to(device)
            unet.zero_grad()
            out = unet(x)
            loss = F.mse_loss(out ,y)
            # loss = my_loss(out, y, 4)
            trainingLoss.append(loss.item())
            print(f'{i}: {loss}, {e}')
            loss.backward()
            optimiser.step()

    sum_of_trainingLosses = sum(trainingLoss)
    epoch_trainingLoss = sum_of_trainingLosses / len(trainingLoss)
    trainingLosses.append(epoch_trainingLoss)

    # OUTPUT MODELS EVERY 5TH EPOCH
    # After first epoch
    # And ensure to output model after all epochs have passed

    ut.save_model2(dir_name, e+1, unet)

# end timer
toc = time.perf_counter()

print(f"time: {(toc-tic) / 60}m")
