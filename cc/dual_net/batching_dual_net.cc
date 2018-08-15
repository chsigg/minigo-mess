#include "cc/dual_net/batching_dual_net.h"

#include <queue>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "cc/check.h"

namespace minigo {
namespace internal {
class BatchingService {
  struct SharedData {
    DualNet::Task task;
    std::atomic<int> num_features;
  };

  struct InferenceData {
    DualNet::Task task;
    absl::Span<const DualNet::BoardFeatures*> features;
    absl::Span<DualNet::Output*> outputs;
    std::string* model;
    std::shared_ptr<SharedData> shared_data;
  };

 public:
  BatchingService(std::unique_ptr<DualNet> parent)
      : num_clients_(0), parent_(std::move(parent)) {}

  void IncrementClients() { ++num_clients_; }
  void DecrementClients() {
    --num_clients_;
    std::lock_guard<std::mutex> lock(queue_mutex_);
    MaybeRunBatches();
  }

  void RunMany(DualNet::Task&& task,
               absl::Span<const DualNet::BoardFeatures*> features,
               absl::Span<DualNet::Output*> outputs, std::string* model) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.push(InferenceData{std::move(task), features, outputs, model});
    queue_counter_ += features.size();
    MaybeRunBatches();
  }

 private:
  void MaybeRunBatches() EXCLUSIVE_LOCKS_REQUIRED(queue_mutex) {
    while (size_t batch_size = [&] {
      size_t batch_size = std::min(queue_counter_ - batch_counter_,
                                   static_cast<size_t>(FLAGS_batch_size));
      // Stop if we won't fill a batch yet but more request will come.
      if (static_cast<int>(batch_size) < FLAGS_batch_size &&
          queue_.size() < num_clients_.load()) {
        return 0;
      }
      return batch_size;
    }()) {
      RunBatch(batch_size);
    }
  }

  void RunBatch(size_t batch_size) EXCLUSIVE_LOCKS_REQUIRED(queue_mutex) {
    /*
    VLOG(2) << "Assembling batch (games = " << queue_.size()
            << ", features = " << batch_size
            << ", size = " << FLAGS_batch_size << ")";
    */

    batch_counter_ += batch_size;

    std::vector<const DualNet::BoardFeatures*> features;
    std::vector<DualNet::Output*> outputs;
    std::vector<InferenceData> batch;

    while (batch_size > 0) {
      auto inference = std::move(queue_.front());
      if (inference.features.size() > batch_size) {
        if (!inference.shared_data) {
          inference.shared_data.reset(
              new SharedData{std::move(inference.task),
                             {static_cast<int>(inference.features.size())}});
        }
        queue_.front().features = inference.features.subspan(batch_size);
        queue_.front().outputs = inference.outputs.subspan(batch_size);
        queue_.front().shared_data = inference.shared_data;

        inference.features = inference.features.subspan(0, batch_size);
        inference.outputs = inference.outputs.subspan(0, batch_size);
      } else {
        queue_.pop();
      }
      batch_size -= inference.features.size();
      batch.push_back(std::move(inference));
    }

    std::string model;
    auto functor = [model](std::vector<InferenceData>& batch) {
      for (auto& inference : batch) {
        auto task = std::move(inference.task);
        if (inference.shared_data) {
          auto remaining = inference.shared_data->num_features -=
              inference.features.size();
          if (remaining == 0) {
            task = std::move(inference.shared_data->task);
          }
        }
        if (task.valid()) {
          if (inference.model) {
            *inference.model = model;
          }
          task();
        }
      }
    };

    DualNet::Task task(std::bind(functor, std::move(batch)));
    parent_->RunMany(std::move(task), absl::MakeSpan(features),
                     absl::MakeSpan(outputs), &model);
  }

  std::atomic<size_t> num_clients_;

  std::mutex queue_mutex_;
  std::queue<InferenceData> queue_ GUARDED_BY(&queue_mutex_);

  size_t queue_counter_ = 0;  // Number of features pushed to queue
  size_t batch_counter_ = 0;  // Number of features pushed to batches

  std::unique_ptr<DualNet> parent_;
};
}  // namespace internal

namespace {
class BatchingDualNet : public DualNet {
 public:
  explicit BatchingDualNet(internal::BatchingService* service)
      : service_(service) {
    service_->IncrementClients();
  }

  ~BatchingDualNet() { service_->DecrementClients(); }

  void RunMany(DualNet::Task&& task, absl::Span<const BoardFeatures*> features,
               absl::Span<Output*> outputs, std::string* model) override {
    service_->RunMany(std::move(task), features, outputs, model);
  }

 private:
  internal::BatchingService* service_;
};
}  // namespace

BatchingDualNetFactory::BatchingDualNetFactory(
    std::unique_ptr<DualNetFactory> parent)
    : DualNetFactory(parent->model()),
      service_(new internal::BatchingService(parent->New())),
      parent_(std::move(parent)) {}

BatchingDualNetFactory::~BatchingDualNetFactory() = default;

std::unique_ptr<DualNet> BatchingDualNetFactory::New() {
  return absl::make_unique<BatchingDualNet>(service_.get());
}

}  // namespace minigo
