The layout of this directory is as follows:
The 1.0/ directory holds half_maps_1 and
the 2.0/ directory holds half_maps_2.
The tally_maps.py script creates a .csv file
which keeps track of the downloaded maps stored
in 1.0/ and 2.0/.
The map_tally.csv is the file that contains a tally of 
all stored maps. You can run $ python tally_maps.py to
re-tally the maps if you have downloaded or removed
any half maps.
The 1.0preds/ contains reconstructed half_maps_1, these
are maps which have been passed through the denoising
network and reconstructed.
Similarly for 2.0preds/.
The directories 1.5/ and 2.5/ are for storing test
maps. These are maps that will be used during the
training of a neural network to see when the network
starts to overfit. These maps will not be used to train
the network so they will serve as unseen test examples.
