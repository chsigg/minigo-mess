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

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "cc/constants.h"
#include "cc/dual_net/fake_dual_net.h"
#include "cc/random.h"
#include "gmock/gmock.h"
#include "grpc++/create_channel.h"
#include "grpc/status.h"
#include "gtest/gtest.h"
#include "proto/inference_service.grpc.pb.h"

namespace minigo {
namespace {

class InferenceServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::generate_n(std::back_inserter(priors_), kNumMoves, Random());
    value_ = 0.1;
    fake_dual_net_ = absl::make_unique<FakeDualNet>(priors_, value_);
    remove_dual_net_ = NewRemoteDualNet("RemoteDualNet");
  }

  std::vector<float> priors_;
  float value_;

  std::unique_ptr<DualNet> fake_dual_net_;
  std::unique_ptr<DualNet> remove_dual_net_;
};

TEST_F(InferenceServerTest, Test) {
  // Run a fake inference worker on a separate thread.
  // Unlike the real inference worker, this fake worker doesn't loop, and
  // doesn't add any RPC ops to the TensorFlow graph. Instead, the RPCs and
  // proto marshalling is performed manually.
  std::thread server_thread([this]() {
    int port = 50051;
    InferenceService::Stub stub(grpc::CreateChannel(
        absl::StrCat("localhost:", port), grpc::InsecureChannelCredentials()));

    grpc::Status status;

    // Get the server config.
    GetConfigRequest get_config_request;
    GetConfigResponse get_config_response;
    {
      grpc::ClientContext context;
      status =
          stub.GetConfig(&context, get_config_request, &get_config_response);
      ASSERT_TRUE(status.ok()) << "RPC failed: " << status.error_message()
                               << ": " << status.error_details();
    }

    int board_size = get_config_response.board_size();
    int batch_size = get_config_response.batch_size();

    ASSERT_EQ(kN, board_size);
    ASSERT_LT(0, batch_size);

    // Get the features.
    GetFeaturesRequest get_features_request;
    GetFeaturesResponse get_features_response;
    {
      grpc::ClientContext context;
      status = stub.GetFeatures(&context, get_features_request,
                                &get_features_response);
      ASSERT_TRUE(status.ok()) << "RPC failed: " << status.error_message()
                               << ": " << status.error_details();
    }
    ASSERT_EQ(batch_size * DualNet::kNumBoardFeatures,
              get_features_response.features().size());

    // Run the model.
    const std::string& src = get_features_response.features();
    std::vector<DualNet::BoardFeatures> features(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < DualNet::kNumBoardFeatures; ++j) {
        features[i][j] =
            static_cast<float>(src[i * DualNet::kNumBoardFeatures + j]);
      }
    }
    std::vector<DualNet::Output> outputs =
        std::move(fake_dual_net_->RunMany(std::move(features)).outputs);

    // Put the outputs.
    PutOutputsRequest put_outputs_request;
    PutOutputsResponse put_outputs_response;
    for (const auto& output : outputs) {
      for (int i = 0; i < kNumMoves; ++i) {
        put_outputs_request.add_policy(output.policy[i]);
      }
      put_outputs_request.add_value(output.value);
    }
    put_outputs_request.set_batch_id(get_features_response.batch_id());
    {
      grpc::ClientContext context;
      status =
          stub.PutOutputs(&context, put_outputs_request, &put_outputs_response);
      ASSERT_TRUE(status.ok()) << "RPC failed: " << status.error_message()
                               << ": " << status.error_details();
    }
  });

  std::vector<DualNet::BoardFeatures> features(16);
  auto result = remove_dual_net_->RunMany(std::move(features));
  for (const auto& output : result.outputs) {
    ASSERT_EQ(output.value, value_);
    ASSERT_EQ(std::equal(priors_.begin(), priors_.end(), output.policy.begin()),
              true);
  }

  server_thread.join();
}

}  // namespace
}  // namespace minigo
