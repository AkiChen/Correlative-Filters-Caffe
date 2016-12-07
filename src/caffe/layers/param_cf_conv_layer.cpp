#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/param_cf_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <time.h>
#include <fstream>
#include <algorithm>
#include <set>

namespace caffe {

template <typename Dtype>
void ParamCfConvolutionLayer<Dtype>::compute_output_shape() {
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
void ParamCfConvolutionLayer<Dtype>::MakeParamCfTable() {
  int filters_related_per_output = 2*(this->param_cf_num_);
  CHECK_GT(this->channels_, filters_related_per_output) << "Param_CF_Conv layer can't take too much correlative filters";

  int list_length = param_cf_num_ * this->num_output_;
  param_cf_table_.resize(list_length);
  //As a simple version, we make the scale filters always at the head of filters per output feature
  //what's more, the dependent filter is just right after its master.
  for(int k = 0; k < this->num_output_; ++k)
    for(int l = 0; l < param_cf_num_; ++l)
      param_cf_table_[param_cf_num_*k + l] = this->channels_*k + 2*l;

}

template <typename Dtype>
void ParamCfConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  this->force_nd_im2col_ = conv_param.force_nd_im2col();
  this->channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = this->channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  this->num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(this->num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(this->num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  this->kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == this->num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < this->num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  this->stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = this->stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == this->num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  this->pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = this->pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == this->num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  this->dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = this->dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == this->num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << this->num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = true;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    this->is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!this->is_1x1_) { break; }
  }
  // Configure output channels and groups.
  this->channels_ = bottom[0]->shape(this->channel_axis_);
  this->num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(this->num_output_, 0);
  this->group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(this->channels_ % this->group_, 0);
  CHECK_EQ(this->num_output_ % this->group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    this->conv_out_channels_ = this->channels_;
    this->conv_in_channels_ = this->num_output_;
  } else {
    this->conv_out_channels_ = this->num_output_;
    this->conv_in_channels_ = this->channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  param_cf_num_ = this->layer_param_.correlative_filters_param().param_cf_num();  
  kernel_size_ = kernel_shape_data[0];
  std::string layer_name_ = this->layer_param_.name();

  LOG(INFO) << "The param_correlative_num_ is :"<<param_cf_num_;

  MakeParamCfTable();
  //set the transform matrix
  int kernel_length = kernel_size_ * kernel_size_;


  vector<int> weight_shape(2);
  weight_shape[0] = this->conv_out_channels_;
  weight_shape[1] = this->conv_in_channels_ / this->group_;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  this->bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(this->bias_term_, this->num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + this->bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (this->bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(3);
    } else {
      this->blobs_.resize(2);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    this->blobs_.back().reset(new Blob<Dtype>(
        1, 1, kernel_length, kernel_length));
    shared_ptr<Filler<Dtype> > tmp_weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    tmp_weight_filler->Fill(this->blobs_.back().get());
    LOG(INFO) << "Param Correlative Filters has been setted";

    // If necessary, initialize and fill the biases.
    if (this->bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  this->kernel_dim_ = this->blobs_[0]->count(1);
  this->weight_offset_ = this->conv_out_channels_ * this->kernel_dim_ / this->group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
void ParamCfConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  //make the weights follow the preseted relation
  const Dtype* cf_weight = this->blobs_.back()->cpu_data();
  int kernel_length = kernel_size_ * kernel_size_;
  for(int g=0; g < param_cf_num_ * this->num_output_; ++g)
    caffe_cpu_gemv<Dtype>(CblasTrans, (int)kernel_length, (int)kernel_length,
    (Dtype)1., (const Dtype*)cf_weight, weight + param_cf_table_[g]*kernel_length, (Dtype)0.,(Dtype* )(weight + (param_cf_table_[g]+1)*kernel_length));


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
void ParamCfConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
  const Dtype* cf_weight = (Dtype*)this->blobs_.back()->cpu_data();
  Dtype* cf_weight_diff = (Dtype*)this->blobs_.back()->mutable_cpu_diff();

  //set the cf_weight_diff
  for(int i=0; i < param_cf_num_*this->num_output_; i++)
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasTrans, kernel_length, kernel_length, 1,
        (Dtype)1., weight + param_cf_table_[i]*kernel_length, weight_diff + (param_cf_table_[i]+1)*kernel_length, (Dtype)1., cf_weight_diff);
  //trans the diff to origin weight
  for(int i=0; i < param_cf_num_*this->num_output_; i++)
    caffe_cpu_gemv<Dtype>(CblasNoTrans, kernel_length, kernel_length,
    (Dtype)1., cf_weight, weight_diff + (param_cf_table_[i]+1)*kernel_length, (Dtype)1., weight_diff + param_cf_table_[i]*kernel_length);
}

#ifdef CPU_ONLY
STUB_GPU(ParamCfConvolutionLayer);
#endif

INSTANTIATE_CLASS(ParamCfConvolutionLayer);
REGISTER_LAYER_CLASS(ParamCfConvolution);
}  // namespace caffe
