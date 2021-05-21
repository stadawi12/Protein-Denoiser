import sys
sys.path.insert(1, 'utils/')
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadTrainingData(globalPath, localPath, n_examples, 
                     df_ids):
    import loader as l

    # load all saved maps
    data = l.data_loader(globalPath, localPath, n_examples)
    # discard maps which are not in df_ids
    data_filtered = []
    for halfMap in data:
        if halfMap.id in df_ids:
            data_filtered.append(halfMap)
    # list of ids, we will return this for comparing 
    # to ensure maps are aligned
    data_ids = [d.id for d in data_filtered]
    # Tile filtered maps
    data_tiled = l.data_preprocess(data_filtered)
    # extract tiles of half maps
    tiles_np = [tile for d in data_tiled for tile in d.tiles]
    # convert tiles to tensors
    tiles_torch = [torch.tensor(t) for t in tiles_np]
    # unsqueeze so the shape of tiles are [bs,c,h,w,d]
    tiles_uncat = [t.unsqueeze(0).unsqueeze(0) \
                         for t in tiles_torch]
    # concatonate tiles into one array
    tiles_torch = torch.cat(tiles_uncat, 0)
    return tiles_torch, data_ids

def loadTestData(globalPath, localPath, n_examples):
    import loader as l

    # load all saved maps
    data = l.data_loader(globalPath, localPath, n_examples)
    # list of ids, we will return this for comparing 
    # to ensure maps are aligned
    data_ids = [d.id for d in data]
    # Tile filtered maps
    data_tiled = l.data_preprocess(data)
    # extract tiles of half maps
    tiles_np = [tile for d in data_tiled for tile in d.tiles]
    # convert tiles to tensors
    tiles_torch = [torch.tensor(t) for t in tiles_np]
    # unsqueeze so the shape of tiles are [bs,c,h,w,d]
    tiles_uncat = [t.unsqueeze(0).unsqueeze(0) \
                         for t in tiles_torch]
    # concatonate tiles into one array
    tiles_torch = torch.cat(tiles_uncat, 0)
    return tiles_torch, data_ids

def get_ids(globalPath, min_dim, max_dim, min_res, max_res):
    from csv_filter import get_entries

    full_path = globalPath + "map_tally.csv"
    # load tally of saved maps from csv file
    df_saved = pd.read_csv(full_path)
    df_saved = pd.DataFrame(df_saved)

    # filter saved maps
    df_filtered = get_entries(df_saved, min_dim, 
                              max_dim,  min_res,
                              max_res)

    df_filtered = df_filtered.loc[df_filtered.index%2 == 0]
    df_entries = df_filtered["Entry"]
    df_entries = df_entries.tolist() # changed from to_numpy to tolist
    df_ids = [el[4:] for el in df_entries]
    return df_ids

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

def save_model(filename, unet):
    og_length = len(filename)
    count = 0
    tm_contents = os.listdir("trained_models")
    if filename+".pt" in tm_contents:
        while filename+".pt" in tm_contents:
            count += 1
            if len(filename) == og_length:
                filename += f"_{count}"
            else:
                filename = list(filename)
                filename[-1] = str(count)
                filename = "".join(filename)
        torch.save(unet.state_dict(), 
                "trained_models/"+filename+".pt")
    else:
        torch.save(unet.state_dict(), 
                "trained_models/"+filename+".pt")

def create_directory(name_of_directory):
    original_length = len(name_of_directory)
    count = 0 
    old_directories = os.listdir("trained_models")
    if name_of_directory in old_directories:
        while name_of_directory in old_directories:
            count += 1
            if len(name_of_directory) == original_length:
                name_of_directory += f"_{count}"
            else:
                name_of_directory = list(name_of_directory)
                name_of_directory[-1] = str(count)
                name_of_directory = "".join(name_of_directory)
        os.mkdir(f"trained_models/{name_of_directory}")
    else:
        os.mkdir(f"trained_models/{name_of_directory}")

    return name_of_directory

def save_model2(name_of_directory, epoch, unet):
    torch.save(unet.state_dict(),
            f"trained_models/{name_of_directory}/epoch_{epoch}.pt")


def save_model(folderName, epoch, unet):
    og_length = len(folderName)
    count = 0
    tm_contents = os.listdir("trained_models")
    if folderName in tm_contents:
        while folderName in tm_contents:
            count += 1
            if len(folderName) == og_length:
                folderName += f"_{count}"
            else:
                folderName = list(folderName)
                folderName[-1] = str(count)
                folderName = "".join(folderName)
        torch.save(unet.state_dict(), 
                f"trained_models/{folderName}/epoch_{epoch}.pt")
    else:
        torch.save(unet.state_dict(), 
                f"trained_models/{folderName}/epoch_{epoch}.pt")

def save_losses(filename, losses):
    og_length = len(filename)
    count = 0
    lo_contents = os.listdir("losses")
    df = pd.DataFrame()
    df["losses"] = losses
    if filename+".csv" in lo_contents:
        while filename+".csv" in lo_contents:
            count += 1
            if len(filename) == og_length:
                filename += f"_{count}"
            else:
                filename = list(filename)
                filename[-1] = str(count)
                filename = "".join(filename)
        df.to_csv(f"losses/"+filename+".csv", index=False)
    else:
        df.to_csv(f"losses/"+filename+".csv", index=False)

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
