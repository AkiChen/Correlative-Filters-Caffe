#include <vector>

#include "caffe/layers/scale_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <time.h>
#include <fstream>
#include <algorithm>
#include <set>

namespace caffe {

template <typename Dtype>
void ScaleConvolutionLayer<Dtype>::compute_output_shape() {
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
void ScaleConvolutionLayer<Dtype>::MakeScaleTable() {
  int filters_related_per_output = 2*(this->scale_num_);
  CHECK_GT(this->channels_, filters_related_per_output) << "Too much pairs of scale CF for this layer, maxium: " << (this->channels_)/2 <<" pairs";

  int list_length = scale_num_ * this->num_output_;
  scale_table_.resize(list_length);
  //The scale correlation use the with-map connection strategy
  //scale_table_ only saves the serial number of master filters,
  //each master's dependent is just right after.
  for(int k = 0; k < this->num_output_; ++k)
    for(int l = 0; l < scale_num_; ++l)
      scale_table_[scale_num_*k + l] = this->channels_*k + 2*l;

}

template <typename Dtype>
void ScaleConvolutionLayer<Dtype>::SpecialSetup() {
  
  scale_num_ = this->layer_param_.correlative_filters_param().scale_num();
  std::string transform_matrix_src = this->layer_param_.correlative_filters_param().scale_trans_source();
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  kernel_size_ = kernel_shape_data[0];

  std::string layer_name_ = this->layer_param_.name();

  LOG(INFO) << "The scale_num_ is :"<<scale_num_;

  MakeScaleTable();
  //set the transform matrix
  int kernel_length = kernel_size_ * kernel_size_;
  transform_matrix_.reset(new SyncedMemory(kernel_length * kernel_length * sizeof(Dtype)));
  
  LOG(INFO) << "Reading transform matrix from file :"<<transform_matrix_src;
  Dtype* trans_mat = (Dtype*)transform_matrix_->mutable_cpu_data();

  std::ifstream in_stream(transform_matrix_src.c_str());
  /*CHECK_NOTNULL(in_stream)<< "Scale transform matrix file not detected";*/
  if(in_stream) {
    LOG(INFO)<<"File Detected";
    Dtype a = 0;
    for(int g = 0; g < kernel_length*kernel_length ; ++g )
    {
      in_stream >> a;
      trans_mat[g]= a;
    }
    in_stream.close();
  }

  //make the weights related
  Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  const Dtype* const_trans_mat = (const Dtype*)transform_matrix_->cpu_data();

  for(int g=0; g < scale_num_ * this->num_output_; ++g)
    caffe_cpu_gemv<Dtype>(CblasTrans, kernel_length, kernel_length,
    (Dtype)1., const_trans_mat, weight + scale_table_[g]*kernel_length, (Dtype)0., weight + (scale_table_[g]+1)*kernel_length);
}

template <typename Dtype>
void ScaleConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void ScaleConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

  //update the correlative filters

  int kernel_length = kernel_size_ * kernel_size_;
  Dtype* const_trans_mat = (Dtype*)transform_matrix_->cpu_data();
  //first step, feed the mater filters with its dependent filters
  for(int g=0; g < scale_num_ * this->num_output_; ++g)
    caffe_cpu_gemv<Dtype>(CblasNoTrans, kernel_length, kernel_length,
    (Dtype)1., const_trans_mat, weight_diff + (scale_table_[g]+1)*kernel_length, (Dtype)1., weight_diff + scale_table_[g]*kernel_length);
  //Second step, broadcast the updated weight_diff 
  for(int g=0; g < scale_num_ * this->num_output_; ++g)
    caffe_cpu_gemv<Dtype>(CblasTrans, kernel_length, kernel_length,
    (Dtype)1., const_trans_mat, weight_diff + scale_table_[g]*kernel_length, (Dtype)0., weight_diff + (scale_table_[g]+1)*kernel_length);
}

#ifdef CPU_ONLY
STUB_GPU(ScaleConvolutionLayer);
#endif

INSTANTIATE_CLASS(ScaleConvolutionLayer);
REGISTER_LAYER_CLASS(ScaleConvolution);
}  // namespace caffe
