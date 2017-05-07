# Text analysis examples using Apache MXNet R package

This repo contains the code and data of two simple examples you can play with.

## Installation

For Windows/OSX users, we provide a CPU-only pre-built binary package. You can install a weekly updated package directly from the R console:

```r
install.packages("drat", repos="https://cran.rstudio.com")
drat::addRepo("dmlc")
install.packages("mxnet")
```

To install the mxnet R package on Linux or enable the GPU backend, please follow the instruction below:

http://mxnet.io/get_started/install.html

Besides, we highly recommend the [blogs on MXNet from Azure team](https://blogs.technet.microsoft.com/machinelearning/tag/mxnet/). It includes detailed tutorials on installation and various examples on distributed training using MXNet.

## CNN example

The CNN example is modified from [one wonderful blog from Azure team](https://blogs.technet.microsoft.com/machinelearning/2017/02/13/cloud-scale-text-classification-with-convolutional-neural-networks-on-microsoft-azure/).

However, they used 4 K80 Tesla GPUs for training the netwrok on over 2 million samples. The model and data have been tailored to get satisfactory performance using CPU on a laptop. This can be used as a starting point for people interested in MXNet.

## RNN example

The RNN example is done as a GSOC project in last year (I am the mentor :laughing: ). Just as the CNN example, you should be able to get good performance on your laptop.
