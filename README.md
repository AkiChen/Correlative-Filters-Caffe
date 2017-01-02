
#  Correlative Filters in Caffe
> Building Correlations Between Filters in Convolutional Neural Networks
> **Hanli Wang, Peiqiu Chen, Sam Kwong**



This project  provides implementation for the  method called **correlative filters** on the caffe framework. It's currently merged in the up-to-date version of  caffe published on 2016.12.14.

Generally speaking, CF introduces a series of revised **2D convolutional layers**  , in which filters are initiated and trained jointly in accordance with predefined correlation (**correlation**, denotes a certain kind of linear transformation here). Compared with the conventional CNN, CFs are efficient to work cooperatively and finally make a more generalized optical system.

The primitive version of CF, including opposite CF and translational CF, has been published on the conference [SMC 2015](http://ieeexplore.ieee.org/document/7379661/). 
The revised version has been accepted by the journal of [IEEE Transcation on Cybernetics](http://ieeexplore.ieee.org/document/7782341/), in which SCF(Static Correlative Filters) and PCF(Parametric Correlative Filters) are introduced.

*Noted that the ten-view-test is also included in this repo. For documentation of the **Multiview test in Caffe**, please refer to the branch named **old_multiview_caffe**. *

----------

[TOC]

## Introducing Correlative Filters
> In this part, we mainly talks about the **motivation** and **workflow** of the proposed CF method. 

Deep learning has swept across almost every field of machine learning like a hurricane. 

To my knowledge, deep neural networks are mostly implemented in an end-to-end fashion, which leads to a  trainable feature representation for the given training data. Hence, using the DNN model, few task specific knowledge is needed to build an acceptable recognition system. 

Nevertheless, the outstanding performance achieved by CNN as compared with other kinds of deep neural networks partly depends on CNN’s special structure of connections in small neighborhood, which is a kind of particular priori knowledge that guides each unit to just focus on its presupposed patch of view. 

Based on this thought, it is reasonable to design a more optimized architecture which brings in more priori information that contributes to optical representation while retaining the flexibility and adaptability of trainable feature extractors. 

### Motivation
> Story behind correlative filters.

#### Collaboration in Biological Visual Systems
In the very early stage of primate subcortical vision systems, there exist cells with center-surround receptive fields which come into two types: one is sensitive to bright spot on dark background whereas the other focuses on the inverse pattern. They are  believed to help extract visual patterns under variant luminance, as shown in figure. 1.

<img src="https://github.com/AkiChen/Correlative-Filters-Caffe/raw/Correlative-Filters/doc_images/1.jpg" style="width:400px;">
**Figure.1**  Center-surround receptive fields sensitive to opposite patterns

#### Collaboration in CNN
As multiple filters have always been recognized as receptive fields of CNN, we visualize the filter banks of a **normally trained network** to examine whether the similar phenomenon occurs, as illustrated in figure. 2.

<img src="https://github.com/AkiChen/Multiview-Caffe/raw/Correlative-Filters/doc_images/2.png" style="width:600px;">
**Figure.2** Illustration of the observed relations between normally trained filters.

We found four kinds of relationship. Note that all the weights of filters are randomly initialized with Gaussian distribution and trained freely with stochastic gradient descent, hence these observed relations indicate the cooperation of correlated filters benefits for extracting visual features.

According to the observation above, we came up with the idea to realize those relationship before training.

### Work Flow
> Brief introduction of SCF and PCF, for more details, please refer to our paper on TC.

#### Static Correlative Filters
To simulate the collaboration discovered, we designed four kinds of static correlative filters, in which each pair of master filter and dependent filter are predefined to have a static relationship, as explained in figure. 3.

<img src="https://github.com/AkiChen/Multiview-Caffe/raw/Correlative-Filters/doc_images/3.png" style="width:600px;">
**Figure 3**. Illustration of the proposed four kinds of SCFs.

The forward pass of SCF is exact the same with normal convolutional layers and the flow chart of back-propagation is as followed.

```sequence
Dependent->Dependent: Compute own weights' diff
Master->Master: Compute own weights' diff
Dependent->Master: Feed diff to its master
Note right of Master: Compute diff of the whole pair
Master->Dependent: Refine the dependent's diff
Note right of Master: Update master weight
Note right of Dependent: Update dependent weight
```
**Figure 4**. Back-propagation of SCFs.

####  Parametric Correlative Filters
Besides the proposed SCFs, there might exist other linear correlations that have not been observed intuitively. As an extension to SCF, we come up with the idea to construct trainable correlations by making the correlation matrix learnable during the network training, which leads to the proposed parametric correlative filter (PCF). An illustration about how to train the proposed PCF is presented in figure below.

<img src="https://github.com/AkiChen/Multiview-Caffe/raw/Correlative-Filters/doc_images/5.jpg" style="width:600px;">
**Figure 5**. Illustration of training parametric correlative filters.

## Instructions for use
> Manual of where and how to use convolutional layers applied with CF.

In my implementation, 2D convolutional layers applied with different kinds of CF are realized as separated types of layers. 

### ORfusionConvolutionLayer
>This layer supports opposite CF and rotary CF. This layer has better performance when placed near input data layer.

A sample prototxt for ORfusionConvolutionLayer is as followed.
```protobuf
layer {
  name: "conv1_0"
  type: "ORfusionConvolution"
  bottom: "data"
  top: "conv1_0"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output:96 
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.08
    }
    bias_filler {
      type: "constant"
    }
  }
  correlative_filters_param {
     opposite_num: 25
     rotate_num: 12 
  }
}
```
Compared with a normal covolutional layer, we only made two modifications:
- The field of **`type`** is changed as `ORfusionConvolution`.
- We add the parameter field of **`correlative_filters_param`** in which numbers of opposite CF and rotary CF in each **input channel** are listed.

In the sample code, this layer takes the data( three channels for CIFAR-10 ) as input layer, so that 75 pairs of opposite and 36 pairs of rotary correlations are defined.

If you want to use only opposite/rotary CF in this layer, just set the rotate/opposite_num as zero.


### ScaleConvolutionLayer
>This layer supports the scaling CF.

Just like the sample code in ORfusionConvolutionLayer, we only needs to edit the  **`type`** and **`correlative_filters_param`**.
```protobuf
layer {
  name: "conv2_0"
  type: "ScaleConvolution"
  bottom: "pool1"
  top: "conv2_0"
  convolution_param{
    num_output:160
    ...
  }
  correlative_filters_param{
    scale_num:5
    scale_trans_source:"scale_transform.rawmatrix"
  }
}
```

As shown in the sample code, scaling CF use parameter named `scale_num` to notify that each **output feature map** has 5 pairs of scaling CF filters, so that 800 pairs of scaling CF are defined in the sample code. `scale_trans_source` denotes the file that saves the transform matrix of scaling CF.

### TranslationConvolutionLayer
>This layer supports the translational CF.

Similarly, to activate translational CF, modifications of **`type`** and **`correlative_filters_param`** are needed.
```protobuf
layer {
  name: "conv3_0"
  type: "TranslationConvolution"
  bottom: "pool2"
  top: "conv3_0"
  convolution_param{
    num_output:256
    ...
  }
  correlative_filters_param{
    translation_horizon_num: 10
    translation_vertical_num: 10
  }
}
```
Parameter `translation_horizon_num` means the number of groups where horizontal translation is applied.
Parameter `translation_vertical_num` means the number of groups where vertical translation is applied.

In the given sample code, both horizontal translation and vertical translation have 2560 groups of translational CF.

### ParamCfConvolutionLayer
>This layer supports the parametric CF.

Sample code for PCF is shown below.
```protobuf
layer {
  name: "conv2_0"
  type: "ParamCfConvolution"
  bottom: "pool1"
  top: "conv2_0"
  convolution_param{
    num_output:160
    ...
  }
  correlative_filters_param{
    param_cf_num: 10 
  }
}
```

In the given sample code, 1600 pairs of parametric CF are defined.

## Download & Citation
Clone this repo with git.
```bash
$ git pull https://github.com/AkiChen/Correlative-Filters-Caffe.git
```

Please cite our paper in your publications if this project helps your research:
```
@article{wang2016building,
  title={Building Correlations Between Filters in Convolutional Neural Networks.},
  author={Wang, H and Chen, P and Kwong, S},
  journal={IEEE transactions on cybernetics},
  year={2016}
}
```
## Feedback & Bug Report
- Email: <payenjoe@126.com>
