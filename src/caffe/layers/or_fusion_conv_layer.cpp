#include <vector>

#include "caffe/layers/or_fusion_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <time.h>
#include <fstream>
#include <algorithm>
#include <set>

namespace caffe {

template <typename Dtype>
void ORfusionConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ORfusionConvolutionLayer<Dtype>::MakeOppositeTable(){
  //For opposite CF, the Cross-Map connection is used.
  //serial number of master filters are saved in the opposite_table_
  //each dependent filter is 1*channels after its master.
  int list_length = this->channels_ * opposite_num_;
  opposite_table_.resize(list_length);
  
  for(int i = 0; i < this->channels_; ++i)
    for(int j = 0; j < opposite_num_; ++j)
      opposite_table_[i*opposite_num_ + j] = i + 2*j*this->channels_;
}

template <typename Dtype>
void ORfusionConvolutionLayer<Dtype>::MakeRotateTable(){
  //For rotary CF, the Cross-Map connection is used.
  //serial number of master filters are saved in the rotate_table_
  //each dependent filter is just right after its master 
  int list_length = this->channels_ * rotate_num_;
  rotate_table_.resize(list_length);
  
  for(int i = 0; i < this->channels_; ++i)
    for(int j = 0; j < rotate_num_; ++j)
      rotate_table_[i*rotate_num_ + j] = i + 2*(j+opposite_num_)*this->channels_;
}

template <typename Dtype>
void ORfusionConvolutionLayer<Dtype>::SpecialSetup() {
  
  opposite_num_ = this->layer_param_.correlative_filters_param().opposite_num();
  rotate_num_ = this->layer_param_.correlative_filters_param().rotate_num();
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  kernel_size_ = kernel_shape_data[0];

  std::string layer_name_ = this->layer_param_.name();

  LOG(INFO) << "Layer named '" <<layer_name_<< "' is equiped with ORfusion CF.";
  LOG(INFO) << "The opposite_num is :"<<opposite_num_;
  LOG(INFO) << "The rotate_num is :"<<rotate_num_;

  MakeOppositeTable();
  MakeRotateTable();

  Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  int filter_offset = kernel_size_ * kernel_size_;
  // Initiate the original raw weights

  //#1 initiate the opposite CF , note that the dependent filter is just right after it's master
  for(int g = 0; g < opposite_num_*this->channels_; ++g){
    //change the value of the dependent opposite filters
    caffe_copy(filter_offset, weight + opposite_table_[g]*filter_offset, weight + (1+opposite_table_[g])*filter_offset);
    caffe_scal(filter_offset, (Dtype)(-1.), weight + (1+opposite_table_[g])*filter_offset);
  }
  //#2 intiate the rotary CF , note that the dependent filter is just right after it's master 
  for(int g = 0; g < rotate_num_*this->channels_; ++g){
    //change the value of the dependent rotary filters
    for(int g2=1; g2 <= kernel_size_; ++g2)
      caffe_stride_copy(kernel_size_, weight + rotate_table_[g]*filter_offset+(g2-1)*kernel_size_, weight + (1+rotate_table_[g])*filter_offset+(kernel_size_ - g2), 1, kernel_size_);
  }
}

template <typename Dtype>
void ORfusionConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  //The forward pass of opposite CF and rotary CF is exactly the same with noraml conv layer.

  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ORfusionConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }

  // Diff estimated by cost function has been calculated,
  // further steps are needed to refine correlated weights.
  int filter_offset = kernel_size_*kernel_size_;

  //#1 For opposite CF
  //each dependent filter feed its master with its own diff * (-1), 
  int channels_tmp = this->channels_;

  for(int g = 0; g < opposite_num_*this->channels_; ++g){
    //Feed the master filters
    caffe_axpy(filter_offset, (Dtype)(-1.), weight_diff + (channels_tmp+opposite_table_[g])*filter_offset, weight_diff + opposite_table_[g]*filter_offset);

    //Update diff of dependent filters
    caffe_copy(filter_offset, weight_diff + opposite_table_[g]*filter_offset, weight_diff + (channels_tmp+opposite_table_[g])*filter_offset);
    caffe_scal(filter_offset, (Dtype)(-1.), weight_diff + (channels_tmp+opposite_table_[g])*filter_offset);
  }

  //#2 For rotary CF
  //similar with opposite CF, firstly, we refine diff of masters, then,
  //update the weight_diff of dependents according to the rotary correlation

  for(int g=0; g < rotate_num_*this->channels_; ++g){
    Dtype* original_pointer = weight_diff + rotate_table_[g]*filter_offset;
    Dtype* right_rotate_pointer = weight_diff + (rotate_table_[g]+channels_tmp)*filter_offset;

    //first step: Feed the original master filter
    for(int g2=1; g2 <= kernel_size_; ++g2)
      caffe_stride_axpy(kernel_size_, (Dtype)(1.), right_rotate_pointer +(kernel_size_ - g2), original_pointer+(g2-1)*kernel_size_, kernel_size_, 1);

    //second step: Broadcast the updated weight_diff to the dependent filters
    for(int g2=1; g2 <= kernel_size_; ++g2)
      caffe_stride_copy(kernel_size_, original_pointer+(g2-1)*kernel_size_, right_rotate_pointer+(kernel_size_ - g2), 1, kernel_size_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ORfusionConvolutionLayer);
#endif

INSTANTIATE_CLASS(ORfusionConvolutionLayer);
REGISTER_LAYER_CLASS(ORfusionConvolution);
}  // namespace caffe
