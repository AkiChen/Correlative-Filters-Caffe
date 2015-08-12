# Multiview test in Caffe

Caffe is among the best presentations of convolutional nerual networks. Its competiters might be cuda-convnet, torch, theano and so on. I choose caffe to train my models because its elegent design of 'solver-net-layer-blob-syncedmem' structure and the excellent speed of trainning is also impressive.

But it seems that the multiview(or 10 view) test of network trained with transformed data is currently not available in caffe at all, even though such method of model testing would always achieve a better score as decribed by Alex Krizhevsky in the training strategy of [cuda-convnet](https://code.google.com/p/cuda-convnet/wiki/TrainingNet).

So I wrote a few lines of code to add this function to caffe.

## What's multiview test

Training a CNN could be very tricky. For example, in the task of image classification, an effective trick is training with transformed data --- in every epoch, each training image only 



