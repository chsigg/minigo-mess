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
namespace {
class FakeDualNet : public DualNet {
 public:
  FakeDualNet(absl::Span<const float> priors, float value) : value_(value) {
    MG_CHECK(priors.empty() || priors.size() == kNumMoves);
    if (priors.empty()) {
      priors_.resize(kNumMoves, 1.0 / kNumMoves);
    } else {
      priors_.assign(priors.begin(), priors.end());
    }
  }

  void RunMany(Task&& task, absl::Span<const BoardFeatures*> features,
               absl::Span<Output*> outputs, std::string* model) override {
    for (auto* output : outputs) {
      std::copy(priors_.begin(), priors_.end(), output->policy.begin());
      output->value = value_;
    }
    if (model != nullptr) {
      *model = "FakeNet";
    }
    task();
  }

 private:
  std::vector<float> priors_;
  float value_;
};
}  // namespace

std::unique_ptr<DualNet> FakeDualNetFactory::New() {
  return absl::make_unique<FakeDualNet>(absl::MakeSpan(priors_), value_);
}

}  // namespace minigo
