#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/param_cf_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ParamCfConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* weight = this->blobs_[0]->mutable_gpu_data();

  //make the weights follow the preseted relation
  const Dtype* cf_weight = this->blobs_.back()->gpu_data();
  int kernel_length = kernel_size_ * kernel_size_;
  for(int g=0; g < param_cf_num_ * this->num_output_; ++g)
    caffe_gpu_gemv<Dtype>(CblasTrans, kernel_length, kernel_length,
    (Dtype)1., cf_weight, weight + param_cf_table_[g]*kernel_length, (Dtype)0., weight + (param_cf_table_[g]+1)*kernel_length);


  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ParamCfConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }

  //update the correlative filters

  int kernel_length = kernel_size_ * kernel_size_;
  const Dtype* cf_weight = (Dtype*)this->blobs_.back()->gpu_data();
  Dtype* cf_weight_diff = (Dtype*)this->blobs_.back()->mutable_gpu_diff();

  //set the cf_weight_diff
  for(int i=0; i < param_cf_num_*this->num_output_; i++)
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasTrans, kernel_length, kernel_length, 1,
        (Dtype)1., weight + param_cf_table_[i]*kernel_length, weight_diff + (param_cf_table_[i]+1)*kernel_length, (Dtype)1., cf_weight_diff);
//    caffe_gpu_gemv<Dtype>(CblasNoTrans, kernel_length, 1,
//    (Dtype)1., weight + param_cf_table_[i]*kernel_length, weight_diff + (param_cf_table_[i]+1)*kernel_length, (Dtype)1., cf_weight_diff);
  //trans the diff to origin weight
  for(int i=0; i < param_cf_num_*this->num_output_; i++)
    caffe_gpu_gemv<Dtype>(CblasNoTrans, kernel_length, kernel_length,
    (Dtype)1., cf_weight, weight_diff + (param_cf_table_[i]+1)*kernel_length, (Dtype)1., weight_diff + param_cf_table_[i]*kernel_length);

  
}

INSTANTIATE_LAYER_GPU_FUNCS(ParamCfConvolutionLayer);

}  // namespace caffe
