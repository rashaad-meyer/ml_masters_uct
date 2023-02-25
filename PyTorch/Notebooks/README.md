# PyTorch Experiment Notebook

## 01_deconv_pytorch

This notebook contains a simple test to show how the deconvolution
layer learns its inverse operation which is the 4 factor convolutional
blur. The Deconv Layer is able to learn the correct kernel and optimize
the MSE to 0 showing that it is learning and functioning correctly

## 03_deconv_gopro_small_test

This notebook carries out an experiment on the
[GOPRO Dataset](https://seungjunnah.github.io/Datasets/gopro.html).
This dataset contains both sharp and blurred images and was used to
test the deconv layer on a real world example. The deconv layer didn't
perform well in this example and optimized the MSE to 0.01 but the
output image as blurry as the input image

## 04_deconv_pytorch_cifar

In this experiment, 2 models were trained on the cifar dataset. Both models
were using the ConvNet Tiny architecture as a backbone. The difference between
the 2 models were that the first was using a deconv layer as its first layer and
the second was using a conv layer. The deconv layer managed to score an accuracy
that was 1% higher than the conv layer after 5 epochs
