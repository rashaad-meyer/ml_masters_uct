# ml_masters_uct

## Deconvolution Folder

### Custom Layers

#### DeconvDft2dLayer.py
This file contains the Deconvolution 2D custom layer class. This layer is only compatible with grayscale images or 2D data.

#### DeconvDft2dRgbLayer.py
This file contains the Deconvolution 2D rgb custom layer class. This layer is only compatible with RGB images or 2D multi-channel data. 

### Test files

#### DeconvMultiDatasetTest.py
This file contains code that mostly trains deconvolution, convolutional, and dense layers on the MNIST and cifar10 dataset. It does include some methods that will plots output images or graphs of the accuracy/loss of the NN while it is trained.

#### DeconvConvDtdTest.py
This file contains a method used to train a simple CNN or DNN (deconvolutional neural network) on the DTD dataset. The link to this dataset can be found in one of the comments in the file

#### DeconvConvFmdTest.py
This file contains a method used to train a simple CNN or DNN (deconvolutional neural network) on the FMD dataset. The link to this dataset can be found in one of the comments in the file

### Miscellaneous Files

#### DeconvFFT.py
This file was used to develop the methods for thd Deconvolution DFT classes

#### ConvertRgbDirectoryToGrayScale.py
This file contains methods that can convert a directory with a dataset filled with rgb images to one with grayscale images. Also has methods that can resize images in directory
