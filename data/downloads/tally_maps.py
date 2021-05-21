# imports
import pandas as pd
import os

# list all maps in directory 1.0/
map_ids = os.listdir('1.0')
# select map IDs only (remove .map from end of file)
map_ids = [ID[:-4] for ID in map_ids]
# Add 'EMD-' to all IDs 
map_ids = ['EMD-' + ID for ID in map_ids]

# load the csv file containing all maps
df_all = pd.read_csv('../halfMaps.csv')
df_all = pd.DataFrame(df_all)
# Ensure I only have two half maps of each structure
searchfor = ['half_map_1', 'half_map_2']
df_all = df_all.loc[
        df_all[" Tail"].str.contains('|'.join(searchfor))]
# search entries of saved maps only
df_saved = df_all.loc[df_all["Entry"].isin(map_ids)]

# save tallied maps to csv file
df_saved.to_csv('map_tally.csv', index=False)
