# PyTorch Experiment Notebook

## 01_deconv_pytorch

This [notebook](https://github.com/viibrem/ml_masters_uct/blob/master/PyTorch/Notebooks/01_deconv_pytorch.ipynb)
contains a simple test to show how the deconvolution
layer learns its inverse operation which is the 4 factor convolutional
blur. The Deconv Layer is able to learn the correct kernel and optimize
the MSE to 0 showing that it is learning and functioning correctly

## 03_deconv_gopro_small_test

This [notebook](https://github.com/viibrem/ml_masters_uct/blob/master/PyTorch/Notebooks/03_deconv_gopro_small_test.ipynb)
 carries out an experiment on the
[GOPRO Dataset](https://seungjunnah.github.io/Datasets/gopro.html).
This dataset contains both sharp and blurred images and was used to
test the deconv layer on a real world example. The deconv layer didn't
perform well in this example and optimized the MSE to 0.01 but the
output image as blurry as the input image

## 04_deconv_pytorch_cifar

In this [notebook](https://github.com/viibrem/ml_masters_uct/blob/master/PyTorch/Notebooks/04_deconv_pytorch_cifar.ipynb)
, 2 models were trained on the cifar10 dataset. Both models
were using the ConvNet Tiny architecture as a backbone. The difference between
the 2 models were that the first was using a deconv layer as its first layer and
the second was using a conv layer. The deconv layer managed to score an accuracy
that was 1% higher than the conv layer after 5 epochs

## 05_deconv_avg_blur_test

In this [notebook](https://github.com/viibrem/ml_masters_uct/blob/master/PyTorch/Notebooks/05_deconv_avg_blur_test.ipynb)
, the deconv layer was tested on how well it could learn to
deblur images that have been blurred with an average blur filter. When adding
padding to the images, the MSE managed to get lower than 0.01 but this could 
be due to a large portion of the pixels being 0(from padding). An experiment
was also run without the padding and the MSE got to 0.025 instead. The deconv
layer didn't do much deblurring and also decreased the image quality as well.
A scaling factor (also a learnable param) was added after the deconv layer to
help the deconv learn the inverse operation better


## 06_deconv_gaussian_blur_test

In this [notebook](https://github.com/viibrem/ml_masters_uct/blob/master/PyTorch/Notebooks/06_deconv_gaussian_blur_test.ipynb)
, the deconv layer was tested on how well it could learn to
deblur images that have been blurred with a gaussian blur filter. When adding
padding to the images, the MSE managed to get to 0.027. Looking at the output,
the deconv layer seemed to blur the image instead of deblurring it.

### Future Experiment ideas:

- Take out padding
- Add learnable scaling factor

## 07_deconv_pytorch_cifar100

In this [notebook](https://github.com/viibrem/ml_masters_uct/blob/master/PyTorch/Notebooks/07_deconv_pytorch_cifar100.ipynb)
, 2 models were trained on the cifar100 dataset. Both models
were using the ResNet18 architecture as a backbone. The difference between
the 2 models were that the first was using a deconv layer as its first layer and
the second was using a conv layer. The deconv layer managed to score an accuracy
of 86.2% whereas the conv layer scored an accuracy of 83.6% after 10 epochs

## 08_deconv_SIR_kaggle_ds

In this [notebook](https://github.com/viibrem/ml_masters_uct/blob/master/PyTorch/Notebooks/08_deconv_SIR_kaggle_ds.ipynb)
, the deconv layer was tested on a Super Image resolution dataset
found on [Kaggle](https://www.kaggle.com/datasets/akhileshdkapse/super-image-resolution?resource=download).
The deconv layer managed to get the MSE down to around 0.008 after 200 epochs,
but it did degrade the quality of the image. The same experiment was done using a
convolution layer. The convolution layer managed to reduce the loss to 0.01 after 200
epochs and blurred the image further showing that the deconv did in fact outperform the conv layer

### Future work:
- Read through super-image resolution papers and apply what was done there in deconv
experiments

## 09_deconv_SIR_L1

In this [notebook](https://github.com/viibrem/ml_masters_uct/blob/master/PyTorch/Notebooks/09_deconv_SIR_L1.ipynb)
, the deconv layer was tested on a Super Image resolution dataset
found on [Kaggle](https://www.kaggle.com/datasets/akhileshdkapse/super-image-resolution?resource=download).
This notebook is almost identical to notebook 8 except that an L1 loss was used in this notebook
whereas L2 (MSE) was used in notebook 8. The deconv layer (L1 loss = 0.08) still manages to beat the conv layer
(L1 loss = 0.09). The picture quality after being inputted in both of these layers do not reduce the quality in
the image or at least it is less noticeable!