from ftplib import FTP
from progressbar import ProgressBar
from csv_filter import get_entries
import pandas as pd
import os
"""
    This module downloads half-maps from the ftp
    server. By default, it saves them to the 
    data/downloads/1.0/ or 2.0/ directory
    depending on whether it's half_map_1 or 2 but
    we can also specify the path where we want to
    save the file.
    It downloads all maps that fit the filtering
    criteria. For example, all maps of dimension
    0-128 and resolution 3-4 Angstroms. Those
    parameters can be altered.
"""

def get_names(input_data, csv_path):
    # Load csv file with all half-maps data
    filepath = csv_path
    data     = pd.read_csv(filepath)
    df       = pd.DataFrame(data)

    # specify filter parameters
    min_dim = input_data["min_dim"]
    max_dim = input_data["max_dim"]
    min_res = input_data["min_res"]
    max_res = input_data["max_res"]

    # filter the csv file for the wanted entries
    df_fltrd = get_entries(df, min_dim, max_dim, min_res, max_res)

    # Estimate size of download
    sizes      = df_fltrd[" Size"].tolist()
    size_total = sum(sizes) / (1024 ** 2)
    print("Total download size = {:.2f} MB".format(size_total))

    entries = df_fltrd["Entry"].tolist()
    tails   = df_fltrd[" Tail"].tolist()

    return entries, tails

def file_write(data):
    file.write(data)
    global pbar
    pbar += len(data)

def download(entry, tail, global_path='data', path=None):
    """
        This function downloads the half map specified
        by the inputs entry and tail. For example
            entry = 'EMD-0552'
            tail  = 'emd_0552_half_map_1.map.gz'
            path  = None
        it will download the following half-map and 
        save it to the dedicated directory 
        (data/downloads/1.0/).
        You can also specify the path by setting
        path equal to your desired download destination.
    """
    # Is it half_map_1 or 2?
    half_map_x = tail[-8]
    assert half_map_x.isdigit(), \
    "half_map_x (%r) must be a digit" % half_map_x

    # extracting ID of file to be downloaded
    id_dwnld = tail[4:-18] 
    assert id_dwnld.isdigit(), \
    "id_download (%r) must be a number" % id_dwnl

    # name of file to be downloaded id + '.map.gz'
    name_dwnld = id_dwnld + '.map.gz'
    name_bare  = id_dwnld + '.map'

    # path to half_map_x (x = 1 or 2)
    local_path = os.path.join(global_path, half_map_x + '.0')
    if path != None:
        local_path = path

    # Check if map already stored
    stored_maps = os.listdir(local_path) #list of stored maps

    if name_dwnld in stored_maps or name_bare in stored_maps:
        if path != None:
            print(f"Map {id_dwnld} already stored in {path}")
            return 'Map stored'
        else:
            print(f"Map {id_dwnld} already stored in {half_map_x}.0/")

    else:
        # access ftp server where half-maps are stored
        ftp = FTP('ftp.ebi.ac.uk')
        ftp.login()
        # go to global directory where all structures are stored
        ftp.cwd('/pub/databases/emdb/structures/')
        # enter specific directory where half-maps can be found
        ftp.cwd(entry+'/other/')

        file_path = os.path.join(local_path, name_dwnld)
        global file
        file = open(file_path, 'wb')
        size = ftp.size(tail)
        global pbar
        pbar = ProgressBar(maxval=size)
        pbar.start()
        ftp.retrbinary("RETR " + tail, file_write)
        file = file.close()
        ftp.quit()

if __name__ == '__main__':


    answer = input("Do you want to download these maps? y/n: ")
    if answer == 'y':
        from sort import Maps
        """
            Before downloading data need to ensure that 
            directories containin maps are prepared 
            correctly in order to prevent downloading
            copies of maps.
        """
        
        # Create object to deal with sorting data
        PATH_1 = '../../data/1.0/'
        PATH_2 = '../../data/2.0/'
        m1 = Maps(PATH_1)
        m2 = Maps(PATH_2)

        # Ensure maps have been flattened
        m1.flatten()
        m2.flatten()
        print("!Flattened directories containing batches")

        # delete residual directories
        m1.del_rest()
        m2.del_rest()
        print("!Deleted residual sub-directories")

        # Ensure bad maps are empty
        m1.move_all_from_bad()
        m2.move_all_from_bad()
        print("!Moved everything from badMaps")

        # Finally, download maps
        for i in range(len(entries)):
            download(entries[i], tails[i])


