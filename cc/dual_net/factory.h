#ifndef MINIGO_CC_DUAL_NET_FACTORY_H_
#define MINIGO_CC_DUAL_NET_FACTORY_H_

#include <memory>
#include <string>
#include <utility>

#include "cc/dual_net/dual_net.h"

namespace minigo {

std::unique_ptr<DualNetFactory> NewDualNetFactory(std::string model_path,
                                                  int parallel_games);

}  // namespace minigo

#endif  // MINIGO_CC_DUAL_NET_FACTORY_H_
