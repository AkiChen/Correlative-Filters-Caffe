# Multiview test in Caffe

Caffe is among the best presentations of convolutional nerual networks. Its competiters might be cuda-convnet, torch, theano and so on. I choose caffe to train my models because its elegent design of 'solver-net-layer-blob-syncedmem' structure and the excellent speed of trainning is also impressive.

But it seems that the multiview(or 10 view) test of network trained with transformed data is currently not available in caffe at all, even though such method of model testing would always achieve a better score as decribed by Alex Krizhevsky in the training strategy of [cuda-convnet](https://code.google.com/p/cuda-convnet/wiki/TrainingNet).

So a few lines of codes are added to realize such function.

## What's multiview test

Training a CNN could be very tricky. For example, in the task of object classification, an effective trick is training with transformed data --- in every epoch, each training image only feeds the net with a small patch of itself(which is called cropping) and the patch might also be transformed as its horizontal reflection since objects in the mirror can be easily recogonized by human. Various transformed data helps relieve overfitting of tradition BP training.

<img src="https://raw.githubusercontent.com/AkiChen/Multiview-Caffe/master/pictures/multiview_origin_pic0.png">

The input transformed data of CNN in train&test phase is shown in the graph above. Once the size of training patch is decided, transformed patch is randomly generated (both the Offset_Y and Offset_X are random numbers, the random patch is mirrored by a probability of 0.5). 

The training phase of caffe and cuda-convnet are almost the same when applied with cropping&mirror and both of them take the center patch of image in testing by default. 

Nevertheless an alternate testing method is also adopted in cuda-convnet: each sample would generate 10 different patches that its final prediction of label is acuqired by averaging the testing result of ten transformed patches as below.

<img src="https://raw.githubusercontent.com/AkiChen/Multiview-Caffe/master/pictures/multiview_origin_pic1.png">

In most cases, the multiview test performs better than only testing on the center patch and the improvement depends mainly on the cropping size that the more you crop on the corner, the more significant enhancement you will gain with multiview testing.

## How to use

You need to edit your model definition at two places as below, you can check the `cifar10_quick_train_test.prototxt` in `examples\cifar10` for an example. It is also recommended to save your labels of test dataset in a file called `label_test_file`.

1.Add one line in the `transform_param` of your data input layer in TEST phase: 

<pre><code>
transform_param { 
    mean_file: "mean.binaryproto" 
    crop_size: 30 
    <strong>multi_view:true</strong>
}
</code></pre>

2.Add a layer that takes the label propability distribution as output. Its name should be *softmax*.

<pre><code>
layer {
    name: "softmax" 
    type: "Softmax" 
    bottom: "ip2" 
    top: "softmax" 
    include { phase: TEST } 
}
</code></pre>

3.'label_test_file' is a file which saves all the labels in one line with type of int. Check the 'label_test_file' in `examples\cifar10` which saves 10,000 labels of all the testing samples.

<pre><code>
3 8 8 0 6 6 1 6 3 1 0 9 5 7 9 8 5 ...
</code></pre>

4.Run a command like below : 

<pre><code>
TOOLS=../../build/tools

    $TOOLS/caffe <strong>multi_view_test</strong> \
    --model=cifar10_quick_train_test.prototxt \
    --weights=cifar10_quick_iter_140000.caffemodel \
    --class_num=10 \
    --iterations=100 \
    --outfile_name=quick \
    --gpu=1 \
    --use_mirror=true
</code></pre>
In which: `--model` means the model definition, `--weights` means the trained net, `--class_num` means the total kinds of samples, `--iterations` means the number of iterations needed to test all the samples(for cifar10 with test batch 100,it's 100), `--outfile_name` is up to your choice, `--gpu` is the gpu id you want to test on, `--use_mirror` is whether you would like to test on mirror patches( so you might test on 5 or 10 patches)

5.Result

If you have the `label_test_file` in the executing path, the averaged accuracy would be printed, or you have to compute the final average label with scores saved in files like outfile_'outfile_name'_left_top_corner.
<pre><code>
I0818 13:59:46.009418 32112 caffe.cpp:346] Each run is testing for 100 iterations.
I0818 13:59:46.013913 32112 caffe.cpp:379] Right now, the input sample is transformed as left_top_corner
I0818 13:59:53.431879 32112 caffe.cpp:379] Right now, the input sample is transformed as right_top_corner
I0818 14:00:00.617496 32112 caffe.cpp:379] Right now, the input sample is transformed as center
I0818 14:00:07.809696 32112 caffe.cpp:379] Right now, the input sample is transformed as left_bot_corner
I0818 14:00:15.003214 32112 caffe.cpp:379] Right now, the input sample is transformed as right_bot_corner
I0818 14:00:22.489519 32112 caffe.cpp:379] Right now, the input sample is transformed as left_top_corner_m
I0818 14:00:30.241361 32112 caffe.cpp:379] Right now, the input sample is transformed as right_top_corner_m
I0818 14:00:38.206393 32112 caffe.cpp:379] Right now, the input sample is transformed as center_m
I0818 14:00:46.335206 32112 caffe.cpp:379] Right now, the input sample is transformed as left_bot_corner_m
I0818 14:00:54.460909 32112 caffe.cpp:379] Right now, the input sample is transformed as right_bot_corner_m
I0818 14:01:02.601949 32112 caffe.cpp:430] For each saved loss file, there are 1000000 scores
I0818 14:01:07.750748 32112 caffe.cpp:462] Load in origin labels.
I0818 14:01:07.753285 32112 caffe.cpp:481] The averaged accuracy is 0.661
</code></pre>

6.Testing with scripts

The testing procedure above is too complex for me, so I write `edit_model.py` and `test_multi_view.sh` in `examples\cifar10` to simply test with 
<pre><code>
./test_multi_viwe.sh quick 140000
</code></pre>
But you still need to add one line of code in the input layer of test phase
<pre><code>
transform_param { 
    mean_file: "mean.binaryproto" 
    crop_size: 30 
    <strong>multi_view:false</strong>
}
</code></pre>
The setting up above won't affect normal training&testing and the rest work is done by the scripts.






