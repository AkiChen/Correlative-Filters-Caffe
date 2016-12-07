#ifndef CAFFE_SCALE_CONV_LAYER_HPP_
#define CAFFE_SCALE_CONV_LAYER_HPP_

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
 *   Several pair of scale corrlations are constructed among 
 *   the filters of conv layer.
 *   
 *   For the scale correlation: the center patch of dependent filter is assigned to be the zoomed whole part of master
 */
template <typename Dtype>
class ScaleConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @Method&Param there are a few appended parameters and methods :
   *  - opposite_num_ The number of opposite correlations constructed, hence there are 2*opposite_num_ filters involved in correlation.
   *  - rotate_num_   The number of rotate   correlations constructed, hence there are 2*rotate_num_   filters involved in correlation.
   *  - kernel_size_ The deprecaded parameter of convoltional layer
   *  - opposite_table_ For opposite relation, the mapping relation of master and dependnet is stored in this table.
   *  - rotate_table_   For rotate   relation, the mapping relation of master and dependnet is stored in this table.
   */
  explicit ScaleConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "ScaleConvolution"; }

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
  void MakeScaleTable();
 
 private:
  int scale_num_;
  int kernel_size_;
  vector<int> scale_table_;
  shared_ptr<SyncedMemory> transform_matrix_;
};

}  // namespace caffe

#endif  // CAFFE_SCALE_CONV_LAYER_HPP_
