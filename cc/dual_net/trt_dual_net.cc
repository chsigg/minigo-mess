// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cc/dual_net/trt_dual_net.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "cc/check.h"
#include "cc/constants.h"
#include "cc/thread_safe_queue.h"
#include "cuda/cuda_runtime_api.h"
#include "tensorrt/NvInfer.h"
#include "tensorrt/NvUffParser.h"

namespace minigo {

namespace {
class TrtWorker {
 public:
  explicit TrtWorker(nvinfer1::ICudaEngine* engine) {
    context_ = engine->createExecutionContext();
    MG_CHECK(context_);

    void* host_ptr;
    size_t input_size =
        FLAGS_batch_size * (kN * kN * DualNet::kNumStoneFeatures);
    MG_CHECK(cudaHostAlloc(&host_ptr, input_size, cudaHostAllocWriteCombined) ==
             cudaSuccess);
    pos_tensor_ = static_cast<float*>(host_ptr);
    size_t output_size = FLAGS_batch_size * (kNumMoves + 1);
    MG_CHECK(cudaHostAlloc(&host_ptr, output_size, cudaHostAllocDefault) ==
             cudaSuccess);
    value_output_ = static_cast<float*>(host_ptr);
    policy_output_ = value_output_ + FLAGS_batch_size;
  }

  ~TrtWorker() {
    cudaFreeHost(value_output_);
    cudaFreeHost(pos_tensor_);

    context_->destroy();
  }

  void RunMany(absl::Span<const DualNet::BoardFeatures*> features,
               absl::Span<DualNet::Output*> outputs) {
    MG_DCHECK(features.size() == outputs.size());

    // Copy the features into the input tensor.
    auto* feature_data = pos_tensor_;
    for (const auto* feature : features) {
      // Copy the features into the input tensor.
      feature_data = std::copy(feature->begin(), feature->end(), feature_data);
    }

    // Run the model.
    void* buffers[] = {pos_tensor_, policy_output_, value_output_};
    context_->execute(FLAGS_batch_size, buffers);

    // Copy the policy and value out of the output tensors.
    const auto* policy_data = policy_output_;
    const auto* value_data = value_output_;
    for (auto* output : outputs) {
      std::copy_n(policy_data, kNumMoves, output->policy.begin());
      policy_data += kNumMoves;
      output->value = *value_data++;
    }
  }

 private:
  nvinfer1::IExecutionContext* context_;

  float* pos_tensor_;
  float* policy_output_;
  float* value_output_;
};

class Logger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "TensorRT internal error: " << msg;
        break;
      case Severity::kERROR:
        std::cerr << "TensorRT error: " << msg;
        break;
      case Severity::kWARNING:
        std::cerr << "TensorRT warning: " << msg;
        break;
      default:
        break;
    }
  }
};
}  // namespace

namespace internal {
class TrtService {
  struct InferenceData {
    DualNet::Task task;
    absl::Span<const DualNet::BoardFeatures*> features;
    absl::Span<DualNet::Output*> outputs;
    std::string* model;
  };

 public:
  TrtService(std::string model_path) : running_(true) {
    runtime_ = nvinfer1::createInferRuntime(logger_);
    MG_CHECK(runtime_);

    auto* parser = nvuffparser::createUffParser();

    parser->registerInput("pos_tensor",
                          nvinfer1::DimsCHW(DualNet::kNumStoneFeatures, kN, kN),
                          nvuffparser::UffInputOrder::kNCHW);

    parser->registerOutput("policy_output");
    parser->registerOutput("value_output");

    auto* builder = nvinfer1::createInferBuilder(logger_);
    auto* network = builder->createNetwork();

    MG_CHECK(parser->parse(model_path.c_str(), *network,
                           nvinfer1::DataType::kFLOAT));

    builder->setMaxBatchSize(FLAGS_batch_size);
    builder->setMaxWorkspaceSize(1 << 30);

    cudaSetDevice(0);
    auto* engine = builder->buildCudaEngine(*network);
    MG_CHECK(engine);

    network->destroy();
    builder->destroy();
    parser->destroy();

    std::vector<std::future<std::pair<int, nvinfer1::ICudaEngine*>>> futures;
    futures.emplace_back(std::async([&] { return std::make_pair(0, engine); }));

    auto* blob = engine->serialize();
    MG_CHECK(blob);

    for (int device_id = 1; device_id < FLAGS_num_gpus; ++device_id) {
      futures.emplace_back(std::async(std::launch::async, [&, device_id]() {
        cudaSetDevice(device_id);
        auto* engine = runtime_->deserializeCudaEngine(blob->data(),
                                                       blob->size(), nullptr);
        MG_CHECK(engine);
        return std::make_pair(device_id, engine);
      }));
    }

    auto functor =
        [this, model_path](const std::pair<int, nvinfer1::ICudaEngine*>& pair) {
          pthread_setname_np(pthread_self(), "TrtWorker");
          cudaSetDevice(pair.first);
          TrtWorker worker(pair.second);
          while (running_) {
            InferenceData inference;
            if (queue_.PopWithTimeout(&inference, absl::Seconds(1))) {
              worker.RunMany(inference.features, inference.outputs);
              if (inference.model) {
                *inference.model = model_path;
              }
              inference.task();
            }
          }
        };

    for (auto& future : futures) {
      auto pair = future.get();
      engines_.push_back(pair.second);
      worker_threads_.emplace_back(functor, pair);
      worker_threads_.emplace_back(functor, pair);
    }
    blob->destroy();
  }

  ~TrtService() {
    running_ = false;
    for (auto& thread : worker_threads_) {
      thread.join();
    }
    for (auto* engine : engines_) {
      engine->destroy();
    }
    runtime_->destroy();
  }

  void RunMany(DualNet::Task&& task,
               absl::Span<const DualNet::BoardFeatures*> features,
               absl::Span<DualNet::Output*> outputs, std::string* model) {
    queue_.Push({std::move(task), features, outputs, model});
  }

 private:
  Logger logger_;
  nvinfer1::IRuntime* runtime_;
  std::vector<nvinfer1::ICudaEngine*> engines_;

  ThreadSafeQueue<InferenceData> queue_;
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> running_;
};
}  // namespace internal

namespace {
class TrtDualNet : public DualNet {
 public:
  explicit TrtDualNet(internal::TrtService* service) : service_(service) {}

  void RunMany(DualNet::Task&& task, absl::Span<const BoardFeatures*> features,
               absl::Span<Output*> outputs, std::string* model) override {
    service_->RunMany(std::move(task), features, outputs, model);
  }

 private:
  internal::TrtService* service_;
};

}  // namespace

TrtDualNetFactory::TrtDualNetFactory(std::string model_path)
    : DualNetFactory(model_path),
      service_(new internal::TrtService(model_path)) {}

TrtDualNetFactory::~TrtDualNetFactory() = default;

std::unique_ptr<DualNet> TrtDualNetFactory::New() {
  return absl::make_unique<TrtDualNet>(service_.get());
}

}  // namespace minigo
