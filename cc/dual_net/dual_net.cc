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

#include "cc/dual_net/dual_net.h"

#include <vector>

#include "cc/color.h"
#include "cc/constants.h"

// TODO(csigg): Expand explanation.
DEFINE_int32(batch_size, 256, "Inference batch size.");
DEFINE_int32(num_gpus, 1, "Number of GPUs to use.");

namespace minigo {

constexpr int DualNet::kNumStoneFeatures;
constexpr int DualNet::kNumBoardFeatures;

DualNet::Service::Service(std::unique_ptr<DualNet> dual_net)
    : dual_net_(std::move(dual_net)) {}

DualNet::Service::~Service() = default;

void DualNet::Service::IncrementClientCount() {}

void DualNet::Service::DecrementClientCount() {}

void DualNet::Service::FlushClient() {}

std::future<DualNet::Result> DualNet::Service::RunManyAsync(
    std::vector<BoardFeatures>&& features) {
  std::vector<const BoardFeatures*> feature_ptrs;
  feature_ptrs.reserve(features.size());
  CopyPointers(features.begin(), features.size(),
               std::back_inserter(feature_ptrs));

  Functor functor(std::move(features));

  std::vector<Output*> output_ptrs;
  output_ptrs.reserve(feature_ptrs.size());
  CopyPointers(functor.outputs.begin(), feature_ptrs.size(),
               std::back_inserter(output_ptrs));

  auto future = functor.promise.get_future();
  dual_net_->RunManyAsync(std::move(feature_ptrs), std::move(output_ptrs),
                          Continuation(std::move(functor)));
  return future;
}

std::future<DualNet::Result> DualNet::Client::RunManyAsync(
    std::vector<DualNet::BoardFeatures>&& features) {
  return service_->RunManyAsync(std::move(features));
}

void DualNet::Client::Flush() { service_->FlushClient(); }

DualNet::DualNet(const std::string& model_path) : model_path_(model_path) {}

DualNet::~DualNet() = default;

DualNet::Result DualNet::RunMany(std::vector<BoardFeatures>&& features) {
  std::vector<const DualNet::BoardFeatures*> feature_ptrs;
  feature_ptrs.reserve(features.size());
  CopyPointers(features.begin(), features.size(),
               std::back_inserter(feature_ptrs));

  Functor functor(std::move(features));

  std::vector<DualNet::Output*> output_ptrs;
  output_ptrs.reserve(feature_ptrs.size());
  CopyPointers(functor.outputs.begin(), feature_ptrs.size(),
               std::back_inserter(output_ptrs));

  auto future = functor.promise.get_future();
  RunManyAsync(std::move(feature_ptrs), std::move(output_ptrs),
               Continuation(std::move(functor)));
  return future.get();
}

void DualNet::SetFeatures(absl::Span<const Position::Stones* const> history,
                          Color to_play, BoardFeatures* features) {
  MG_CHECK(history.size() <= kMoveHistory);
  Color my_color = to_play;
  Color their_color = OtherColor(my_color);

  // Write the features for the position history that we have.
  size_t j = 0;
  for (j = 0; j < history.size(); ++j) {
    auto* dst = features->data() + j * 2;
    const auto* end = dst + kNumBoardFeatures;
    const auto* src = history[j]->data();
    while (dst < end) {
      auto color = src->color();
      ++src;
      dst[0] = color == my_color ? 1 : 0;
      dst[1] = color == their_color ? 1 : 0;
      dst += kNumStoneFeatures;
    }
  }

  // Pad the features with zeros if we have fewer than 8 moves of history.
  for (; j < kMoveHistory; ++j) {
    auto* dst = features->data() + j * 2;
    const auto* end = dst + kNumBoardFeatures;
    while (dst < end) {
      dst[0] = 0;
      dst[1] = 0;
      dst += kNumStoneFeatures;
    }
  }

  // Set the "to play" feature plane.
  float to_play_feature = to_play == Color::kBlack ? 1 : 0;
  auto* dst = features->data() + kPlayerFeature;
  const auto* end = dst + kNumBoardFeatures;
  while (dst < end) {
    dst[0] = to_play_feature;
    dst += kNumStoneFeatures;
  }
}

}  // namespace minigo
