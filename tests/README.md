# Test Package

There are two test modules with unit tests in them here so far.
`tests.py` and `tests2.py`. 

## `tests.py`

This module tests the download of half maps and ensures it is working
as intended, similarly for res scraper which is the resolution web 
scraper and size scraper module, it also ensures that the PyTorch 
unet network works as intended, takes in a tensor of shape 
`[n,1,64,64,64]`
and outputs a tensor of same shape with no errors in between.
This module also tests the crop function used for constructing the 
neural network.

## `tests2.py`

This module tests the training and processing pipelines. Ensures that
training works and produces all the correct outputs in all the 
correct directories.
I train a unet on a single tile (shape `[1,1,64,64,64]`) so that it 
doesn't take a lot of time and then I denoise the tile using the 
model that has been trained and checking if the outputs are where
they should be and whether they contain all the correct files and
filenames.
The `inputs.yaml` file is tailored for testing, please do not alter 
unless you know what you are doing.
