# Using Noise2Noise models for denoising electron density maps

In this project, I am using various neural network 
architectures based on noise-to-noise learning
to denoise electron density maps of proteins,
[noise2noise paper](https://arxiv.org/abs/1803.04189).

## Half Maps

My training data consists of protein half-maps. Electron 
density maps are generated by processing 2D x-ray images 
containing many protein crystals in different orientations. 
A series of steps, of locating particles in an image,
sorting them according to their orientation and averaging
allows for building a reliable representation of the protein
from different viewing angles. A 3D map can be built based
on the 2D "shots" of the protein from different angles.

Half-maps are produced by splitting the particles in 
the initial 2D x-ray image into two groups and 
building two seperate 3D maps in same fashion as outlined 
above, giving two electron density maps called 
half-maps. Constructing two half-maps - each by using half
of that data - reduces statistical bias when compared
to generating a single 3D map which uses all data points.

## Noise2Noise

Through many years a vast library of highly detailed 
protein maps has been accrued and is still growing every day
to this day. Given that many models used for constructing 
the protein maps generate half-maps, there exists a 
vast amount of data needed for noise2noise learning.

As the two half-maps both contain a similar but unique 
noise profile we are able to use them for noise2noise 
learning. Where we set one of the two half-maps as an
input to the network and the other half-map as a target.
This allows the network to learn the noise profile of the
protein maps and effectively reduce the amount of noise
in the 3D map.

## Architectures

When constructing a noise2noise network we have a lot
of freedom and choices to make as to the hyperparameters
of the network. In this project we are testing and
comparing different network architectures. It is natural
that there will exist more favourabel configurations. 

This project proves that half-maps can be used for
noise2noise training as we have observed significant
noise reduction in the maps after using 30 half-map
pairs to train our noise2noise network.

The drawback of this method is blurring. The maps - although
to a certain extent denoised - become blurred, losing 
the high frequency signal. This is the main issue that
is being tackled in this project. We are testing 
different architectures and progressively 
providing more maps for the network to learn in order
to find the best way of preserving the high-frequency signal.

# Getting started with the code

First thing when you download the code is to ensure
that you have all the necessary python packages installed.
The necessary packages can be found in the 
`environment.txt` file in the main directory.

Next you should familiarise yourself with the layout of the
directories and purpose of each module.

## `main.py`
This module controls most actions that you would like to
perform; download data, train on data, denoise a map using
a trained model and some other actions that we will get in to
a bit more detail. Firstly, this module takes in one argument,
that is the action you would like to perform. The flag for
this argument is `-a` or `--action` and the allowed choices
are: `['train', 'proc', 'moveback', 'download']`.

Before running any of these actions you want to take a look 
at the `inputs.yaml` file. It should contain all the inputs
and required parameters for each action. Take a look at the 
type of parameters you can manipulate.

### `python main.py -a train`
This command should train a model based on the parameters
specified in the `inputs.yaml` file. The training will find
maps stored in the `data/` directory. We will get into the 
structure of the `data/` directory later on. 

The training module can be found in `lib/Train.py` this is
the backend of the training action. Any training outputs such
as the trained model, plots and other model related outputs
will go to the `out/` directory.

Everytime a train action is performed successfully, first a 
directory will be created in the `out/` directory, it will
be named according to the parameters specified in the
`inputs.yaml` file and the number of training examples used.
For example, if we have ten training examples, we plan to run
the training for ten epochs and we choose a mini-batch-size of 
three, a directory `out/m10_e10_mbs3/` will be created. The 
next thing in the pipeline is to copy the `inputs.yaml` file
into that directory so that we know what parameters we have set
before the training began, this makes it easier to later look
back at our training outputs and determine the parameters used
for that training session, this is important when you will run 
many different training sessions. It will also create two
other directories, `models/` and `plots/` inside. After each
epoch it will output the trained model to `models/` and a 
plot of the loss + validation curves for that epoch. At the end
of our training we should have ten models stored in `models/`
and ten plots stored inside `plots/`, one for each epoch. They
will be labeled accordingly.

In case we run another training session and we choose to use the
same parameters, the pipeline should recognise that a directory
with the same filename already exists and instead of overwriting
that directory it will create another directory as follows
`out/m10_e10_mbs3_1`, appending a `'_1'` to the directory
name. It will do that for up to ten directories, so the largest
number it can go to is `'_9'` copies, it has trouble with 
appending double digits to the end of the filename but I hope
that you will not be running a training session over ten times 
using the same parameters.


### `python main.py -a proc`

This action process or denoises a map specified in the inputs
file and uses one of the trained models stored in the `out/` 
directory. In the `inputs.yaml` file you can specify which
model to use and which epoch of that model you would like to
use to denoise the specified map stored in the 
`data/<res>/` directory where `res` can be specified in the 
`inputs.yaml` file.

When denoising a map, ensure you use the same DataLoader 
parameters that were used to train the model, these include
the following `[norm, norm_vox, norm_vox_min(max)]` rest of
the parameters you can mostly get away with, do not play with
`cshape` and `margin` unless you know what you are doing.

This action will load the map, decompose
it into tiles, pass the tiles through the trained 
network i.e. denoise them, one-by-one and at the end it will
recompose the map and save it into the 
`out/<training_session_name>/denoised/` directory with an
appropriate filename corresponding to the map ID and epoch
used.

### `python main.py -a moveback`

This is a simple action that moves all the maps from the 
`badMaps` directories back in with the rest of the maps

### `python main.py - download`

This action downloads the maps based on criteria specified in the
inputs file 
`[min_dim, max_dim, min_res, max_res]`. All the downloaded maps
will land in either `data/1.0/'` or `data/2.0/` directory, in 
general this action will download pairs of half-maps, one
will go into one directory and another in the other. If it finds
that this half-map is already stored in that directory it will not
download it again.

Before the download begins this action first ensures that 
the equivalent of `moveback` command takes place in order
to ensure that we do not download maps that we already have 
on our system. However, if we have maps stored in the
`data/1.5(2.5)` directories, it will still download these maps
again so, be careful. This needs to be fixed in the future.

The downloaded maps will be zipped you would want to unzip them
using `gunzip *.gz`.

## `xcorrSort.py`

This module is used for cleaning our data, the training maps in
directories `data/1.0` and `data/2.0` will be cross-correlated
with each other and a `threshold` can be set in the `inputs.yaml` file.
The cross-correlation between two maps is a measure of similarity
between the two half-maps, it ranges from -1 to 1 where -1 is 
negatively correlated and 1 means they are exactly the same.
Any maps whose correlation is below the `threshold` will be moved
to their corresponding `badMaps/` directories which can be found in
the `data/` directory. Any maps found in the `badMaps/` directories
will by default not be used in training. If you want to move maps
back out of the `badMaps` directory, you should use the action
`moveback` (`$ ccpem-python main.py -a moveback`). You can then rerun the
`xcorrSort.py` module with a different `threshold` value.
IMPORTANT: To run this module you will need to use `ccpem-python`.

If need any help email me at:
mr.dawidstasiak@outlook.com
