#ifndef CC_DUAL_NET_BATCHING_DUAL_NET_H_
#define CC_DUAL_NET_BATCHING_DUAL_NET_H_

#include "cc/dual_net/dual_net.h"

namespace minigo {

namespace internal {
class BatchingService;
}

class BatchingDualNetFactory : public DualNetFactory {
 public:
  BatchingDualNetFactory(std::unique_ptr<DualNetFactory> parent);
  ~BatchingDualNetFactory();

  std::unique_ptr<DualNet> New() override;

 private:
  std::unique_ptr<internal::BatchingService> service_;
  std::unique_ptr<DualNetFactory> parent_;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_BATCHING_DUAL_NET_H_
