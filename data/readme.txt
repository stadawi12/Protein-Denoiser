The layout of the data directory is as follows:

The 1.0/ directory holds half_maps_1 and
the 2.0/ directory holds half_maps_2.

The 1.0preds/ contains reconstructed half-maps, these
are maps which have been passed through the denoising
network and reconstructed.

The directories 1.5/ and 2.5/ are for storing test
maps. These are maps that will be used during the
training of a neural network to see when the network
starts to overfit. These maps will not be used to train
the network so they will serve as unseen test examples.

The (1.0 and 2.0)badMaps/ directories are used for 
storing half-maps whose cross-correlation is below
our desired threshold.

Inside of directories [1.0, 2.0, 1.5, 2.5] we will find
a sub-directory structure. Half-maps will be stored in
sub-directories named [1.0, 2.0, 3.0,..., n.0] each
sub-directory will be populated with half-maps, half-maps
will be equally divided between those sub-directories
with the exception the the last sub-directory (n.0) might
contain fewer maps.
This sub-directory structure is used for loading maps into
memory, only one sub-directory of maps will be loaded into
memory at once, this is to prevent memory overflow.
