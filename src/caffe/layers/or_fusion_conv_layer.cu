#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/or_fusion_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ORfusionConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
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
void ORfusionConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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

  // Diff estimated by cost function has been calculated,
  // further steps are needed to refine correlated weights.
  int filter_offset = kernel_size_*kernel_size_;

  //#1 For opposite CF
  //each dependent filter feed its master with its own diff * (-1), 

  for(int g = 0; g < opposite_num_*this->channels_; ++g){
    //Feed the master filters
    caffe_gpu_axpy(filter_offset, (Dtype)(-1.), weight_diff + (1+opposite_table_[g])*filter_offset, weight_diff + opposite_table_[g]*filter_offset);

    //Update diff of dependent filters
    caffe_copy(filter_offset, weight_diff + opposite_table_[g]*filter_offset, weight_diff + (1+opposite_table_[g])*filter_offset);
    caffe_gpu_scal(filter_offset, (Dtype)(-1.), weight_diff + (1+opposite_table_[g])*filter_offset);
  }

  //#2 For rotary CF
  //similar with opposite CF, firstly, we refine diff of masters, then,
  //update the weight_diff of dependents according to the rotary correlation


  for(int g=0; g < rotate_num_*this->channels_; ++g){
    Dtype* original_pointer = weight_diff + rotate_table_[g]*filter_offset;
    Dtype* right_rotate_pointer = weight_diff + (rotate_table_[g]+1)*filter_offset;

    //first step: Feed the original master filter
    for(int g2=1; g2 <= kernel_size_; ++g2)
      caffe_gpu_stride_axpy(kernel_size_, (Dtype)(1.), right_rotate_pointer +(kernel_size_ - g2), original_pointer+(g2-1)*kernel_size_, kernel_size_, 1);

    //second step: Broadcast the updated weight_diff to the dependent filters
    for(int g2=1; g2 <= kernel_size_; ++g2)
      caffe_gpu_stride_copy(kernel_size_, original_pointer+(g2-1)*kernel_size_, right_rotate_pointer+(kernel_size_ - g2), 1, kernel_size_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ORfusionConvolutionLayer);

}  // namespace caffe
