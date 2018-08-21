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

#include "cc/dual_net/fake_dual_net.h"

#include "absl/memory/memory.h"
#include "cc/check.h"

namespace minigo {
FakeDualNet::FakeDualNet() : FakeDualNet(0.0f) {}

FakeDualNet::FakeDualNet(float value)
    : FakeDualNet(std::vector<float>(kNumMoves, 1.0f / kNumMoves), value) {}

FakeDualNet::FakeDualNet(std::vector<float> priors, float value)
    : DualNet("FakeDualNet"), priors_(std::move(priors)), value_(value) {}

void FakeDualNet::RunManyAsync(std::vector<const BoardFeatures*>&& features,
                               std::vector<Output*>&& outputs,
                               Continuation continuation) {
  for (auto* output : outputs) {
    std::copy(priors_.begin(), priors_.end(), output->policy.begin());
    output->value = value_;
  }
  continuation(model_path_);
}
}  // namespace minigo
