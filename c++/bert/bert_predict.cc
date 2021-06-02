#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <cmath>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(warmup, 0, "warmup.");
DEFINE_int32(repeats, 1, "repeats.");
DEFINE_bool(use_gpu, false, "use gpu.");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(100, 0);
  } else {
    config.EnableMKLDNN();
  }

  // Open the memory optim.
  config.EnableMemoryOptim();
  return CreatePredictor(config);
}

void run(Predictor *predictor, std::vector<int64_t> *input,
         const std::vector<int> &input_shape, std::vector<float> *out_data) {
  int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                  std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_t = predictor->GetInputHandle(input_names[i]);
    input_t->Reshape(input_shape);
    input_t->CopyFromCpu(input[i].data());
  }
  
  for (size_t i = 0; i < FLAGS_warmup; ++i)
    CHECK(predictor->Run());

  auto st = time();
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    CHECK(predictor->Run());
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    out_data->resize(out_num);
    output_t->CopyToCpu(out_data->data());
  }
  LOG(INFO) << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms";
}

void softmax(std::vector<float>& data){
  float sum = exp(data[0]) + std::exp(data[1]);
  for(size_t i = 0; i < data.size(); ++i) {
    data[i] = exp(data[i])/sum;
  }
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();

  std::vector<int> input_shape = {1, 19}; // batch size = 1
  std::vector<int64_t> inputs_data[2] = {
    {101, 2114, 22349, 16434, 2008, 18496, 2015, 1996, 4292, 1996, 28855, 15879, 5053, 1997, 2019, 4004, 5957, 4169, 102},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  }; // 'against shimmering cinematography that lends the setting the ethereal beauty of an asian landscape painting'

  std::vector<float> out_data;
  run(predictor.get(), inputs_data, input_shape, &out_data);

  softmax(out_data);

  std::string label = out_data[0] > out_data[1] ? "Negative" : "Positive";
  LOG(INFO) << "Data: against shimmering cinematography that lends the setting the ethereal beauty of an asian landscape painting";
  LOG(INFO) << "  Label: " << label;
  LOG(INFO) << "  Negative prob: " << out_data[0];
  LOG(INFO) << "  Positive prob: " << out_data[1];

  return 0;
}
