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

#include "cc/dual_net/tf_dual_net.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "cc/check.h"
#include "cc/constants.h"
#include "cc/thread_safe_queue.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

using tensorflow::DT_FLOAT;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace minigo {

namespace {
class TfWorker {
 public:
  explicit TfWorker(const tensorflow::GraphDef& graph_def) {
    tensorflow::SessionOptions options;
    options.config.mutable_gpu_options()->set_allow_growth(true);
    session_.reset(tensorflow::NewSession(options));
    TF_CHECK_OK(session_->Create(graph_def));

    inputs_.emplace_back("pos_tensor",
                         tensorflow::Tensor(tensorflow::DT_FLOAT,
                                            tensorflow::TensorShape(
                                                {FLAGS_batch_size, kN, kN,
                                                 DualNet::kNumStoneFeatures})));

    output_names_.emplace_back("policy_output");
    output_names_.emplace_back("value_output");
  }

  ~TfWorker() {
    if (session_ != nullptr) {
      TF_CHECK_OK(session_->Close());
    }
  }

  void RunMany(const std::vector<const DualNet::BoardFeatures*>& features,
               const std::vector<DualNet::Output*>& outputs) {
    // Copy the features into the input tensor.
    auto* feature_data = inputs_.front().second.flat<float>().data();
    for (const auto* feature : features) {
      // Copy the features into the input tensor.
      feature_data = std::copy(feature->begin(), feature->end(), feature_data);
    }

    // Run the model.
    TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

    // Copy the policy and value out of the output tensors.
    const auto* policy_data = outputs_[0].flat<float>().data();
    const auto* value_data = outputs_[1].flat<float>().data();
    for (auto* output : outputs) {
      std::copy_n(policy_data, kNumMoves, output->policy.begin());
      policy_data += kNumMoves;
      output->value = *value_data++;
    }
  }

 private:
  std::unique_ptr<tensorflow::Session> session_;
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_;
  std::vector<std::string> output_names_;
  std::vector<tensorflow::Tensor> outputs_;
};

class TfDualNet : public DualNet {
  struct InferenceData {
    std::vector<const BoardFeatures*> features;
    std::vector<Output*> outputs;
    Continuation continuation;
  };

 public:
  explicit TfDualNet(std::string model_path)
      : DualNet(model_path), running_(true) {
    auto functor = [this](const tensorflow::GraphDef& graph_def) {
      pthread_setname_np(pthread_self(), "TfWorker");
      TfWorker worker(graph_def);
      while (running_) {
        InferenceData inference;
        if (queue_.PopWithTimeout(&inference, absl::Seconds(1))) {
          worker.RunMany(inference.features, inference.outputs);
          inference.continuation(model_path_);
        }
      }
    };

    // If we can't find the specified graph, try adding a .pb extension.
    auto* env = tensorflow::Env::Default();
    if (!env->FileExists(model_path).ok()) {
      model_path = absl::StrCat(model_path, ".pb");
    }

    tensorflow::GraphDef graph_def;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(env, model_path, &graph_def));

    for (int device_id = 0; device_id < FLAGS_num_gpus; ++device_id) {
      auto device = std::to_string(device_id);
      PlaceOnDevice(&graph_def, "/gpu:" + device);
      // Two threads per device.
      worker_threads_.emplace_back(functor, graph_def);
      worker_threads_.emplace_back(functor, graph_def);
    }
  }

  ~TfDualNet() override {
    running_ = false;
    for (auto& thread : worker_threads_) {
      thread.join();
    }
  }

  void RunManyAsync(std::vector<const BoardFeatures*>&& features,
                    std::vector<Output*>&& outputs,
                    Continuation continuation) override {
    queue_.Push(
        {std::move(features), std::move(outputs), std::move(continuation)});
  }

 private:
  static void PlaceOnDevice(tensorflow::GraphDef* graph_def,
                            const std::string& device) {
    for (auto& node : *graph_def->mutable_node()) {
      if ([&] {
            if (node.op() != "Const") {
              return true;
            }
            auto it = node.attr().find("dtype");
            return it == node.attr().end() ||
                   it->second.type() != tensorflow::DT_INT32;
          }()) {
        node.set_device(device);
      }
    }
  }

  ThreadSafeQueue<InferenceData> queue_;
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> running_;
};
}  // namespace

std::unique_ptr<DualNet> NewTfDualNet(const std::string& model_path) {
  return absl::make_unique<TfDualNet>(model_path);
}

}  // namespace minigo
