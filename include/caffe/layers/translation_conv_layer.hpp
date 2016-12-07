#ifndef CAFFE_TRANSALTION_CONV_LAYER_HPP_
#define CAFFE_TRANSALTION_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Mostly the same with normal Convolutional Layer
 *  
 *
 *   Several pair of opposite corrlations or roatation correlations are constructed among 
 *   the filters of conv layer.
 *
 *   For the opposite correlation: through the whole trainning procedure, the master
 *   filter is assigned to have opposite values compared with its dependetn one.
 *   
 *   For the rotate correlation: master filter and its dependent is presented as rotated relation
 */
template <typename Dtype>
class TranslationConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @Method&Param there are a few appended parameters and methods :
   *  - horizon_num_ The number of horizontal shifted filter.
   *  - vertical_num_   The number of vertical shifted filter.
   *  - kernel_size_ The deprecaded parameter of convoltional layer
   *  - horizon_table_ For horizon relation
   *  - vertical_table_   For vertical relation
   */
  explicit TranslationConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "TranslationConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  virtual void SpecialSetup();
  void MakeShiftTable();
 
 private:
  int horizon_num_;
  int vertical_num_;
  int kernel_size_;
  vector<int> horizon_table_;
  vector<int> vertical_table_;
};

}  // namespace caffe

#endif  // CAFFE_TRANSALTION_CONV_LAYER_HPP_
