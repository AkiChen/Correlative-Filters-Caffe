#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>//aki_update

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"//aki_update

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;


DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
//aki_update_start
//these are flags which are needed in multi-view test
DEFINE_int32(class_num, 0,
    "The number of class to write to outfile");
DEFINE_string(outfile_name, "",
    "The name output file");
DEFINE_string(use_mirror, "",
    "use mirror or not in multiview");
//aki_update_start

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  caffe::Caffe::SetDevice(FLAGS_gpu);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  LOG(INFO) << "Starting Optimization";
  shared_ptr<caffe::Solver<float> >
    solver(caffe::GetSolver<float>(solver_param));

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Solve(FLAGS_snapshot);
  } else if (FLAGS_weights.size()) {
    CopyLayers(&*solver, FLAGS_weights);
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

// Multi View Test: score a model like cuda-convnet
int multi_view_test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a _train_test.prototxt file to define model. use --model= ";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need a _iter_XXX.caffemodel to set weight. use --weights= ";
  CHECK_GT(FLAGS_class_num, 0) << "Need the number of classes. use --class_num= ";
  CHECK_GT(FLAGS_outfile_name.size(), 0) << "Need the prefix of ouput loss files. use --outfile_name= ";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    LOG(INFO) << "It is strongly recommended to use GPU right here.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Each run is testing for " << FLAGS_iterations << " iterations.";


  //set the view which is needed
  std::map<int, std::string> view_table;
  std::map<int ,std::string >::iterator current_view;

  view_table.insert(std::pair<int,std::string>(1,"left_top_corner"));
  view_table.insert(std::pair<int,std::string>(2,"right_top_corner"));
  view_table.insert(std::pair<int,std::string>(3,"left_bot_corner"));
  view_table.insert(std::pair<int,std::string>(4,"right_bot_corner"));
  view_table.insert(std::pair<int,std::string>(5,"center"));
  if(FLAGS_use_mirror == "true") {
    view_table.insert(std::pair<int,std::string>(6,"left_top_corner_m"));
    view_table.insert(std::pair<int,std::string>(7,"right_top_corner_m"));
    view_table.insert(std::pair<int,std::string>(8,"left_bot_corner_m"));
    view_table.insert(std::pair<int,std::string>(9,"right_bot_corner_m"));
    view_table.insert(std::pair<int,std::string>(10,"center_m"));
  }
  
  int total_score_num = 0;

  for(current_view=view_table.begin(); current_view!=view_table.end(); current_view++ ){

    vector<Blob<float>* > bottom_vec;
    vector<int> test_score_output_id;
    vector<float> test_score;
    float loss = 0;

    std::ofstream outfile((std::string("outfile_")+FLAGS_outfile_name+"_"+current_view->second).c_str());
    int index = 0;
    std::ofstream tmp_cache(std::string("multiview_cache").c_str());
    tmp_cache << current_view->first;
    LOG(INFO) <<"Right now, the input sample is transformed as "<<current_view->second;
    tmp_cache.close();

    for (int i = 0; i < FLAGS_iterations; ++i) {
      //for each iteration of one batch
      float iter_loss;
      const vector<Blob<float>*>& result =
          caffe_net.Forward(bottom_vec, &iter_loss);
      loss += iter_loss;
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        //for each input testing picture
        const float* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k, ++idx) {
          //for each score on one class
          const float score = result_vec[k];
          if (i == 0) {
            test_score.push_back(score);
            test_score_output_id.push_back(j);
          } else {
            test_score[idx] += score;
          }
          const std::string& output_name = caffe_net.blob_names()[caffe_net.output_blob_indices()[j]];
          //The last layer should output the probability distribution of input picture
          //Mostly, we use the softmax layer.
          if(output_name == "softmax"){
            outfile << score << " ";
            if(current_view==view_table.begin())
                total_score_num++;
          }
          if(output_name == "sigmoid"){
              outfile << score << " ";
            if(current_view==view_table.begin())
                total_score_num++;
          }
        }
      }
    }

    outfile.close();
  }

  //Now that all the loss has been saved on the disk, we take them out to caculate the average accuracy
  float* sum_result;
  float* sub_result;

  LOG(INFO) <<"For each saved loss file, there are "<<total_score_num<<" scores"<<std::endl;
  sum_result = (float*)malloc(sizeof(float)*total_score_num);
  sub_result = (float*)malloc(sizeof(float)*total_score_num);
  
  //read and add the scores
  for(current_view=view_table.begin(); current_view!=view_table.end(); current_view++ )
  {
    std::ifstream infile((std::string("outfile_")+FLAGS_outfile_name+"_"+current_view->second).c_str());
    if(current_view==view_table.begin()){
      //As the first file, we directly read&write the scores to the float array which saves the sum of score
      for(int i=0; i < total_score_num; ++i)
        infile >> sum_result[i];
    }
    else{
      //As the rest files, their scores is read&write to sub_result and added to the sum_result
      for(int i=0; i < total_score_num; ++i)
        infile >> sub_result[i];
      caffe::caffe_axpy(total_score_num, (float)1.0,(const float*)sub_result,(float*)sum_result);
    }
  }
  delete(sub_result);
  //now we need to caculate the accuracy, noted that the correct label is save in label_test_file
  int* correct_labels;
  int test_sample_num = total_score_num / FLAGS_class_num;
  int correct_count = 0;

  //read in the origin labels
  correct_labels = (int *)malloc(sizeof(int)*test_sample_num);
  std::ifstream infile(std::string("label_test_file").c_str());
  for( int i=0; i< test_sample_num; ++i){
    infile >> correct_labels[i];
  }
  LOG(INFO)<<"Load in origin labels.";
  std::pair<float, int> tmp_decision;
  for(int i=0; i< test_sample_num; ++i)
    for(int j=0; j< FLAGS_class_num; ++j){
      if (j==0){
        tmp_decision.first = sum_result[i*FLAGS_class_num+j];
        tmp_decision.second = 0;
      }
      if (sum_result[i*FLAGS_class_num+j] > tmp_decision.first) {
        tmp_decision.first = sum_result[i*FLAGS_class_num+j];
        tmp_decision.second = j;
      }
      if ( j == FLAGS_class_num-1 ){
        if( tmp_decision.second == correct_labels[i] )
          correct_count++;
      }
    }

  float accuracy = correct_count/(0.0+test_sample_num);
  LOG(INFO) <<"The averaged accuracy is "<<accuracy;

  delete(sum_result);
  delete(correct_labels);
  

  return 0;
}
RegisterBrewFunction(multi_view_test);


int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
