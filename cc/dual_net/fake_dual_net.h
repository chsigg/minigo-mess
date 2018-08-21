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

#ifndef CC_DUAL_NET_FAKE_DUAL_NET_H_
#define CC_DUAL_NET_FAKE_DUAL_NET_H_

#include <vector>

#include "cc/dual_net/dual_net.h"

namespace minigo {

class FakeDualNet : public DualNet {
 public:
  FakeDualNet();
  explicit FakeDualNet(float value);
  FakeDualNet(std::vector<float> priors, float value);

  void RunManyAsync(std::vector<const BoardFeatures*>&& features,
                    std::vector<Output*>&& outputs,
                    Continuation continuation) override;

 private:
  std::vector<float> priors_;
  float value_;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_FAKE_DUAL_NET_H_
