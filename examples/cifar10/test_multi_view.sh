#!/usr/bin/env sh
PREFIX="cifar10_"
TOOLS=../../build/tools
MODEL=${PREFIX}$1"_train_test.prototxt"
WEIGHT=${PREFIX}$1"_iter_"$2".caffemodel"
                                                                                                                                                                               
./edit_model.py append $MODEL
                                                                                                                                                                               
$TOOLS/caffe multi_view_test \
  --model=$MODEL \
  --weights=$WEIGHT \
  --class_num=10 \
  --iterations=100 \
  --outfile_name=quick \
  --gpu=1 \
  --use_mirror=true
                                                                                                                                                                               
./edit_model.py remove $MODEL