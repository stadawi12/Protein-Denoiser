# ML Protein Toolbox for 3D cryo-EM data segmentation

The toolbox contains a set of tools for labelling, preprocessing, analysing and postprocessing 3D cryo-EM volumes and corresponding models. All preprocessing and analysis tools are optimised for cryo-EM data specifically.

Notable features:
 * command line tool
 * API
 * set of custom loss functions to handle volume background imbalance
 * 3 modes of pre- and post-processing including .mrc headers
 * customisable architecture
 * loading and saving data
 * data structure for holding maps and models
 * 8 different metrics and visualisations for performance tracking

### Contributors:
Before submitting push request, please make sure your changes pass the unit tests by running:
```python
python3 -m unitttest
```
If the new changes provide new functionality or API, please document the changes by providing appropriate doc strings and write a unit test for your change.

Thank you for your interest in developing the toolbox!

### Usage:

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -u [querymaps|train|predict|makelabels]
```

    Expected format of the passed directory containing data:
        path/:
            2.5/
            2.5label/
            3/
            3label/
            ...

    - querymaps - load data and display dataset table along wiht header information

    - train - train a neural network for segmentation based on the label directories

    - predict - inference based on existing model

    - makelabels - create Gaussian-based, individually thresholded labelled maps from corresponding models 

### Preprocessing

 * -p/--preprocess [crop|scale|tile] - preprocess data

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -p tile
```

    - crop - crops data and headers to desired shape, use along with -s/--cshape parameter to specify shape (64 isotropic by default)
    - sclae - scales data and headers to desired shape, use along with -s/--cshape parameter to specify shape (64 isotropic by default)
    - tile - sliding 3D tiles across the image, data is saved in sample.tiles for maps and sample.ltiles for labels, use sample.recompose() to stitch back together

    For tiles, use parameter -th/--threshold (float) and -bg/--background (float) to reject tiles with less background proportion than -bg given threshold -th.

 * -n/--normalise - flag to normalise the data between 0 and 1

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -p tile --normalise
```

### Training

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -p tile -u train
```

 * -e/--epoch - number of epochs to train for

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -p tile -u train -e 100
```

 * -b/--batch - number of data points per batch

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -p tile -u train -e 100 -b 10
```

 * -a/--arch [large|medium|small] - choice of architecture depth

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -p tile -u train -e 100 -b 10 -a large
```
    - large - depth 3
    - medium - depth 2
    - small - depth 1

 * -l/--loss [scce|weighted_scce|focal|custom_weighted_scce|prob_scce|weighted_prob_scee] - choice of loss function

```python
python3 ml-protein-toolbox.py -d path -r 2.5 -r 3 -p tile -u train -e 100 -b 10 -a large - l weighted_scce
```

   For imbalanced data use weighted or focal. For custom, provide an extra directory (e.g. 2.5weights) with mrc file containing per-voxel weights.

### Metric tracking and visualising

 * -cb/--callback [checkpoints|metrics|confusion|maps|stopping|tests|trains|timestamps]

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -p tile -u train -e 100 -b 10 -a large - l weighted_scce -cb metrics -cb confusion -cb maps
```

    - checkpoints - saves model checkpoints in pwd/checkpoints

    - metrics - saves performance tracking metrics (accuracy, non-zero true positives, loss) to pwd/metrics

    - confusion - saves confusion matrices to pwd/confusion

    - maps - saves per-epoch map predictions as mrc file for validation in datapath/[RES]preds; if -cb test or -cb train specified, it also saves predictions on tests and trains sets

    - stopping - enable early stopping (patience=30 since best epoch)

    - tests - all chosen metrics are typically saved for validation set, with -cb tests, they are also saved for the test set

    - trains - all chosen metrics are typically saved for validation set, with -cb trains, they are also saved for the train set

    - timestamp - timestamps statistics instead of overwritting

### Inference / prediction

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -p tile -u predict
```

 * -h/--model - path to pre-trained model

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -p tile -u predict --model path/to/model
```

 * -v/--evaluate - evaluate model (data must have labels directory)

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -p tile -u predict --evaluate
```

### Generate labels

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -u makelabels
```
    Expected format of the passed directory containing data:
        path/:
            pdbs/
            maps/ (optional, if not passed, it will generate synthetic maps)

 * -m/--mpilabs - use MPI

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -u makelabels --mpilabs
```

 * -g/--genmaps - generates synthetic maps from models and labels these maps

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -u makelabels --genmaps
```

 * -ts/--two_sigma - generates data with thresholding based on two sigmas - computationally heavy, but produces very accurate labels

```python
python3 ml-protein-toolbox.py -d path -r [2.5, 3, ...] -u makelabels --mpilabs -ts
```

 * -jl/--json_labs - pulls resolution to generate the labels at from an associated json file:
        ```json
        sample-id {
            resolution: 3.4
        }
        ```

