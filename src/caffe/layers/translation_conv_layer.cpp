#include <vector>

#include "caffe/layers/translation_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <time.h>
#include <fstream>
#include <algorithm>
#include <set>

namespace caffe {

template <typename Dtype>
void TranslationConvolutionLayer<Dtype>::compute_output_shape() {
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
void TranslationConvolutionLayer<Dtype>::MakeShiftTable() {
  int filters_related_per_output = 3*(this->horizon_num_ + this->vertical_num_);
  CHECK_GT(this->channels_, filters_related_per_output) << "Translation_Conv layer can't take too much translation filters";
  
  int ver_length = this->vertical_num_*this->num_output_;
  int hor_length = this->horizon_num_*this->num_output_;
  vertical_table_.resize(ver_length);
  horizon_table_.resize(hor_length);
  //As a simple version, we make the vertical filters always at the head of filters while the horizon is just right behind it
  for(int k = 0; k < this->num_output_; ++k){
    for(int l = 0; l < this->vertical_num_; ++l)
      vertical_table_[this->vertical_num_*k + l] = this->channels_*k + 3*l;
    for(int l = 0; l < this->horizon_num_;  ++l)
      horizon_table_[this->horizon_num_*k + l] = this->channels_*k + 3*this->vertical_num_ + 3*l;
  }

}

template <typename Dtype>
void TranslationConvolutionLayer<Dtype>::SpecialSetup() {
  
  this->horizon_num_ = this->layer_param_.correlative_filters_param().translation_horizon_num();
  this->vertical_num_ = this->layer_param_.correlative_filters_param().translation_vertical_num();
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  kernel_size_ = kernel_shape_data[0];

  std::string layer_name_ = this->layer_param_.name();

  LOG(INFO) << "The horizon_num_ is :"<<horizon_num_;
  LOG(INFO) << "The vertical_num_ is :"<<vertical_num_;

  MakeShiftTable();
  
  //make the related weight equal
  Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  int filter_offset = kernel_size_ * kernel_size_;
  int half_kernel = kernel_size_/2 ;

  //first step: we make the vertical related filters equal
  for(int g=0; g < vertical_num_ * this->num_output_; ++g){
    Dtype* original_pointer = weight + vertical_table_[g]*filter_offset;
    caffe_copy(filter_offset, original_pointer, original_pointer + 1*filter_offset + half_kernel*kernel_size_);
  }
  //second step: we make the the horizon related filters equal
  for(int g=0; g < horizon_num_  * this->num_output_; ++g){
    Dtype* original_pointer = weight + horizon_table_[g]*filter_offset;
    //right_shift (kernel_size_ - half_kernel) pixels
    for(int g2 = 0; g2 < half_kernel; ++g2)
      caffe_stride_copy(kernel_size_, original_pointer+g2, original_pointer + 1*filter_offset+(kernel_size_ - half_kernel)+g2, kernel_size_, kernel_size_);
    //left_shift half_kernel pixels
    for(int g3 = 0; g3 < (kernel_size_ - half_kernel); ++g3)
      caffe_stride_copy(kernel_size_, original_pointer+half_kernel+g3, original_pointer + 2*filter_offset+g3, kernel_size_, kernel_size_);
  }
}

template <typename Dtype>
void TranslationConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
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
void TranslationConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

  //for the weight_diff,the dependent filters feed the master filters with its own diff

  int filter_offset = kernel_size_ * kernel_size_;
  int half_kernel = kernel_size_/2 ;

  //first step: let the vertical related filters feed the original filters
  for(int g=0; g < vertical_num_ * this->num_output_; ++g){
    Dtype* original_pointer = weight_diff + vertical_table_[g]*filter_offset;
    //down_shift half_kernel AND up_shift (kernel_size_ - half_kernel)
    caffe_axpy(filter_offset, (Dtype)(1.), original_pointer + 1*filter_offset + half_kernel*kernel_size_, original_pointer);
  }

  //second step: let the horizon related filters feed the original filters
  for(int g=0; g < horizon_num_  * this->num_output_; ++g){
    Dtype* original_pointer = weight_diff + horizon_table_[g]*filter_offset;

    //right_shift (kernel_size_ - half_kernel)
    for(int g2 = 0; g2 < half_kernel; ++g2)
      caffe_stride_axpy(kernel_size_, (Dtype)(1.), original_pointer+1*filter_offset+(kernel_size_ - half_kernel)+g2, original_pointer, kernel_size_, kernel_size_);
    //left_shift half_kernel
    for(int g3 = 0; g3 < (kernel_size_ - half_kernel); ++g3)
      caffe_stride_axpy(kernel_size_, (Dtype)(1.), original_pointer+2*filter_offset+g3, original_pointer, kernel_size_, kernel_size_);
  }

  //third step: broadcast the updated weight_diff to the vertical filters
  for(int g=0; g < vertical_num_ * this->num_output_; ++g){
    Dtype* original_pointer = weight_diff + vertical_table_[g]*filter_offset;
    //down_shift half_kernel AND up_shift (kernel_size_ - half_kernel)
    caffe_copy(filter_offset, original_pointer, original_pointer + 1*filter_offset + half_kernel*kernel_size_);
  }

  //fourth step: broadcast the updated weight_diff to the horizon filters
  for(int g=0; g < horizon_num_  * this->num_output_; ++g){
      Dtype* original_pointer = weight_diff + horizon_table_[g]*filter_offset;

     //right_shift (kernel_size_ - half_kernel)
      for(int g2 = 0; g2 < half_kernel; ++g2)
        caffe_stride_copy(kernel_size_, original_pointer+g2, original_pointer+1*filter_offset+(kernel_size_ - half_kernel)+g2, kernel_size_, kernel_size_);
     //left_shift half_kernel
      for(int g3 = 0; g3 < (kernel_size_ - half_kernel); ++g3)
        caffe_stride_copy(kernel_size_, original_pointer+half_kernel+g3, original_pointer+2*filter_offset+g3, kernel_size_, kernel_size_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(TranslationConvolutionLayer);
#endif

INSTANTIATE_CLASS(TranslationConvolutionLayer);
REGISTER_LAYER_CLASS(TranslationConvolution);
}  // namespace caffe
