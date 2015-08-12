# Multiview test in Caffe

Caffe is among the best presentations of convolutional nerual networks. Its competiters might be cuda-convnet, torch, theano and so on. I choose caffe to train my models because its elegent design of 'solver-net-layer-blob-syncedmem' structure and the excellent speed of trainning is also impressive.

But it seems that the multiview(or 10 view) test of network trained with transformed data is currently not available in caffe at all, even though such method of model testing would always achieve a better score as decribed by Alex Krizhevsky in the training strategy of [cuda-convnet](https://code.google.com/p/cuda-convnet/wiki/TrainingNet).

So a few lines of codes are added to realize such function.

## What's multiview test

Training a CNN could be very tricky. For example, in the task of object classification, an effective trick is training with transformed data --- in every epoch, each training image only feeds the net with a small patch of itself(which is called cropping) and the patch might also be transformed as its horizontal reflection since objects in the mirror can be easily recogonized by human. Various transformed data helps relieve overfitting during training.
<img src="./pictures/mult_view_pic1.png" style="width:500px;">

Like 



Unfinished





