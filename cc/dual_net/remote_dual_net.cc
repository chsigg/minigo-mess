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

#include "cc/dual_net/remote_dual_net.h"

#include <atomic>
#include <functional>
#include <future>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "cc/check.h"
#include "cc/thread_safe_queue.h"
#include "gflags/gflags.h"
#include "grpc++/grpc++.h"
#include "grpc++/server.h"
#include "proto/inference_service.grpc.pb.h"

// Worker flags.
DEFINE_string(checkpoint_dir, "",
              "Path to a directory containing TensorFlow model checkpoints. "
              "The inference worker will monitor this when a new checkpoint "
              "is found, load the model and use it for futher inferences. "
              "Only valid when remote inference is enabled.");
DEFINE_bool(use_tpu, true,
            "If true, the remote inference will be run on a TPU. Ignored when "
            "remote_inference=false.");
DEFINE_string(tpu_name, "", "Cloud TPU name, e.g. grpc://10.240.2.2:8470.");
DEFINE_int32(conv_width, 256, "Width of the model's convolution filters.");
DEFINE_int32(parallel_tpus, 8,
             "If model=remote, the number of TPU cores to run on in parallel.");

// Server flags.
DEFINE_int32(port, 50051, "The port opened by the InferenceService server.");

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

namespace minigo {
namespace {

// The RemoteDualNet client pushes inference requests onto an instance of this
// InferenceService.
class RemoteDualNet : public DualNet, InferenceService::Service {
  struct InferenceData {
    std::vector<const BoardFeatures*> features;
    std::vector<Output*> outputs;
    Continuation continuation;
  };

 public:
  explicit RemoteDualNet(const std::string& model_path)
      : DualNet(model_path), batch_id_(1) {
    worker_thread_ = std::thread([=]() {
      std::vector<std::string> cmd_parts = {
          absl::StrCat("BOARD_SIZE=", kN),
          "python",
          "inference_worker.py",
          absl::StrCat("--model=", model_path),
          absl::StrCat("--checkpoint_dir=", FLAGS_checkpoint_dir),
          absl::StrCat("--use_tpu=", FLAGS_use_tpu),
          absl::StrCat("--tpu_name=", FLAGS_tpu_name),
          absl::StrCat("--conv_width=", FLAGS_conv_width),
          absl::StrCat("--parallel_tpus=", FLAGS_parallel_tpus),
      };
      auto cmd = absl::StrJoin(cmd_parts, " ");
      FILE* f = popen(cmd.c_str(), "r");
      for (;;) {
        int c = fgetc(f);
        if (c == EOF) {
          break;
        }
        fputc(c, stderr);
      }
      fputc('\n', stderr);
    });

    server_ = [&] {
      ServerBuilder builder;
      builder.AddListeningPort(absl::StrCat("0.0.0.0:", FLAGS_port),
                               grpc::InsecureServerCredentials());
      builder.RegisterService(this);
      return builder.BuildAndStart();
    }();
    std::cerr << "Inference server listening on port " << FLAGS_port
              << std::endl;
    server_thread_ = std::thread([this]() { server_->Wait(); });
  }

  ~RemoteDualNet() override {
    server_->Shutdown(gpr_inf_past(GPR_CLOCK_REALTIME));
    server_thread_.join();
    worker_thread_.join();
  }

  void RunManyAsync(std::vector<const BoardFeatures*>&& features,
                    std::vector<Output*>&& outputs,
                    Continuation continuation) override {
    queue_.Push(
        {std::move(features), std::move(outputs), std::move(continuation)});
  }

 private:
  Status GetConfig(ServerContext* context, const GetConfigRequest* request,
                   GetConfigResponse* response) override {
    response->set_board_size(kN);
    response->set_batch_size(FLAGS_batch_size);
    return Status::OK;
  }

  Status GetFeatures(ServerContext* context, const GetFeaturesRequest* request,
                     GetFeaturesResponse* response) override {
    InferenceData inference;
    while (!queue_.PopWithTimeout(&inference, absl::Seconds(1))) {
      if (context->IsCancelled()) {
        return Status(StatusCode::CANCELLED, "connection terminated");
      }
    }

    std::string byte_features(FLAGS_batch_size * DualNet::kNumBoardFeatures, 0);
    auto it = byte_features.begin();
    for (const auto* features : inference.features) {
      it = std::transform(features->begin(), features->end(), it,
                          [](float x) { return x != 0.0f ? 1 : 0; });
    }
    response->set_batch_id(batch_id_++);
    response->set_features(std::move(byte_features));

    {
      absl::MutexLock lock(&pending_mutex_);
      pending_map_[response->batch_id()] = std::move(inference);
    }

    return Status::OK;
  }

  Status PutOutputs(ServerContext* context, const PutOutputsRequest* request,
                    PutOutputsResponse* /*response*/) override {
    InferenceData inference;
    {
      absl::MutexLock lock(&pending_mutex_);
      auto it = pending_map_.find(request->batch_id());
      MG_CHECK(it != pending_map_.end());
      inference = std::move(it->second);
      pending_map_.erase(it);
    }

    // Check we got the expected number of values. (Note that because request
    // may be padded, inference.features.size() <= FLAGS_batch_size).
    MG_CHECK(request->value().size() == FLAGS_batch_size)
        << "Expected response with " << FLAGS_batch_size << " values, got "
        << request->value().size();

    // There should be kNumMoves policy values for each inference.
    MG_CHECK(request->policy().size() == request->value().size() * kNumMoves);

    auto policy_it = request->policy().begin();
    auto value_it = request->value().begin();
    for (auto* outputs : inference.outputs) {
      std::copy_n(policy_it, outputs->policy.size(), outputs->policy.begin());
      policy_it += outputs->policy.size();
      outputs->value = *value_it++;
    }

    inference.continuation(request->model_path());

    return Status::OK;
  }

 private:
  std::thread worker_thread_;
  std::thread server_thread_;

  std::unique_ptr<grpc::Server> server_;

  std::atomic<int32_t> batch_id_;

  ThreadSafeQueue<InferenceData> queue_;

  // Mutex that protects access to pending_map_.
  absl::Mutex pending_mutex_;

  // Map from batch ID to list of remote inference requests in that batch.
  std::unordered_map<int32_t, InferenceData> pending_map_
      GUARDED_BY(&pending_mutex_);
};

}  // namespace

std::unique_ptr<DualNet> NewRemoteDualNet(const std::string& model_path) {
  return absl::make_unique<RemoteDualNet>(model_path);
}

}  // namespace minigo
