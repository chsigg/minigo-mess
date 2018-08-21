#include "cc/dual_net/factory.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "cc/dual_net/batching_service.h"
#include "gflags/gflags.h"

#ifdef MG_ENABLE_REMOTE_DUAL_NET
#include "cc/dual_net/remote_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "remote"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_REMOTE_DUAL_NET

#ifdef MG_ENABLE_TF_DUAL_NET
#include "cc/dual_net/tf_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "tf"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_TF_DUAL_NET

#ifdef MG_ENABLE_LITE_DUAL_NET
#include "cc/dual_net/lite_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "lite"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_LITE_DUAL_NET

#ifdef MG_ENABLE_TRT_DUAL_NET
#include "cc/dual_net/trt_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "trt"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_TRT_DUAL_NET

DEFINE_string(engine, MG_DEFAULT_ENGINE,
              "The inference engine to use. Accepted values:"
#ifdef MG_ENABLE_REMOTE_DUAL_NET
              " \"remote\""
#endif
#ifdef MG_ENABLE_TF_DUAL_NET
              " \"tf\""
#endif
#ifdef MG_ENABLE_LITE_DUAL_NET
              " \"lite\""
#endif
#ifdef MG_ENABLE_TRT_DUAL_NET
              " \"trt\""
#endif
);

DECLARE_int32(virtual_losses);

namespace minigo {

DualNetFactory::~DualNetFactory() = default;

std::unique_ptr<DualNet::Service> NewDualNetService(
    const std::string& model_path) {
  std::unique_ptr<DualNet> dual_net;

  if (FLAGS_engine == "remote") {
#ifdef MG_ENABLE_REMOTE_DUAL_NET
    dual_net = NewRemoteDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with remote inference support";
#endif  // MG_ENABLE_REMOTE_DUAL_NET
  }

  if (FLAGS_engine == "tf") {
#ifdef MG_ENABLE_TF_DUAL_NET
    dual_net = NewTfDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with tf inference support";
#endif  // MG_ENABLE_TF_DUAL_NET
  }

  if (FLAGS_engine == "lite") {
#ifdef MG_ENABLE_LITE_DUAL_NET
    dual_net = NewLiteDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with lite inference support";
#endif  // MG_ENABLE_LITE_DUAL_NET
  }

  if (FLAGS_engine == "trt") {
#ifdef MG_ENABLE_TRT_DUAL_NET
    dual_net = NewTrtDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with TensorRT inference support";
#endif  // MG_ENABLE_TRT_DUAL_NET
  }

  if (!dual_net) {
    MG_FATAL() << "Unrecognized inference engine \"" << FLAGS_engine << "\"";
  }

  if (FLAGS_batch_size > 0) {
    // Batching was requested, return BatchingService.
    return NewBatchingService(std::move(dual_net));
  }

  return absl::make_unique<DualNet::Service>(std::move(dual_net));
}

}  // namespace minigo
