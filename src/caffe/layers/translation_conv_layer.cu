#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/translation_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void TranslationConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
void TranslationConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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

  //for the weight_diff,the dependent shift filters feed the original filters with its own diff

  int filter_offset = kernel_size_ * kernel_size_;
  int half_kernel = kernel_size_/2 ;

  //first step: let the vertical related filters feed the original filters
  for(int g=0; g < vertical_num_ * this->num_output_; ++g){
    Dtype* original_pointer = weight_diff + vertical_table_[g]*filter_offset;
    //down_shift half_kernel AND up_shift (kernel_size_ - half_kernel)
    caffe_gpu_axpy(filter_offset, (Dtype)(1.0), original_pointer + 1*filter_offset + half_kernel*kernel_size_, original_pointer);
  }

  //second step: let the horizon related filters feed the original filters
  for(int g=0; g < horizon_num_  * this->num_output_; ++g){
    Dtype* original_pointer = weight_diff + horizon_table_[g]*filter_offset;

    //right_shift (kernel_size_ - half_kernel)
    for(int g2 = 0; g2 < half_kernel; ++g2)
      caffe_gpu_stride_axpy(kernel_size_, (Dtype)(1.), original_pointer+1*filter_offset+(kernel_size_ - half_kernel)+g2, original_pointer, kernel_size_, kernel_size_);
    //left_shift half_kernel
    for(int g3 = 0; g3 < (kernel_size_ - half_kernel); ++g3)
      caffe_gpu_stride_axpy(kernel_size_, (Dtype)(1.), original_pointer+2*filter_offset+g3, original_pointer, kernel_size_, kernel_size_);

   // caffe_gpu_scal(filter_offset, (Dtype)(1.75), original_pointer);
  }

  //third step: broadcast the updated weight_diff to the vertical shifted filters
  for(int g=0; g < vertical_num_ * this->num_output_; ++g){
    Dtype* original_pointer = weight_diff + vertical_table_[g]*filter_offset;
    //down_shift half_kernel AND up_shift (kernel_size_ - half_kernel)
    caffe_copy(filter_offset, original_pointer, original_pointer + 1*filter_offset + half_kernel*kernel_size_);
  }

  //fourth step: broadcast the updated weight_diff to the horizon shifted filters
  for(int g=0; g < horizon_num_  * this->num_output_; ++g){
      Dtype* original_pointer = weight_diff + horizon_table_[g]*filter_offset;

     //right_shift (kernel_size_ - half_kernel)
      for(int g2 = 0; g2 < half_kernel; ++g2)
        caffe_gpu_stride_copy(kernel_size_, original_pointer+g2, original_pointer+1*filter_offset+(kernel_size_ - half_kernel)+g2, kernel_size_, kernel_size_);
     //left_shift half_kernel
      for(int g3 = 0; g3 < (kernel_size_ - half_kernel); ++g3)
        caffe_gpu_stride_copy(kernel_size_, original_pointer+half_kernel+g3, original_pointer+2*filter_offset+g3, kernel_size_, kernel_size_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TranslationConvolutionLayer);

}  // namespace caffe
