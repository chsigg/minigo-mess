#include "cc/dual_net/batching_service.h"

#include "absl/memory/memory.h"
#include "cc/check.h"

namespace minigo {
namespace {
class BatchingService : public DualNet::Service {
  struct SharedFunctor {
    DualNet::Functor functor;
    std::atomic<size_t> num_remaining;
  };

  struct BatchFunctor {
    void operator()(const std::string& model) {
      for (auto& functor : functors) {
        functor(model);
      }
      for (size_t i = 0; i < shared_functors.size(); ++i) {
        const auto& shared_functor = shared_functors[i];
        if (!(shared_functor->num_remaining -= shared_num_features[i])) {
          shared_functor->functor(model);
        }
      }
    }

    std::vector<DualNet::Functor> functors;
    std::vector<std::shared_ptr<SharedFunctor>> shared_functors;
    std::vector<size_t> shared_num_features;
  };

  struct InferenceData {
    std::vector<DualNet::BoardFeatures> features;
    std::promise<DualNet::Result> promise;
  };

 public:
  explicit BatchingService(std::unique_ptr<DualNet> dual_net)
      : Service(std::move(dual_net)),
        num_clients_(0),
        queue_counter_(0),
        run_counter_(0),
        shared_size_(0),
        num_runs_(0) {}

  ~BatchingService() override {
    std::cerr << "Ran " << num_runs_ << " batches with an average size of "
              << static_cast<float>(run_counter_) / num_runs_ << ".\n";
  }

  void IncrementClientCount() override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++num_clients_;
  }

  void DecrementClientCount() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (--num_clients_ > 0) {
      MaybeRunBatches();
    }
  }

  void FlushClient() override {
    std::lock_guard<std::mutex> lock(mutex_);
    flush_queue_.push(queue_counter_);
    MaybeRunBatches();
  }

  // Runs inference on a batch of input features aynchronously.
  std::future<DualNet::Result> RunManyAsync(
      std::vector<DualNet::BoardFeatures>&& features) override {
    InferenceData inference = {std::move(features)};
    auto future = inference.promise.get_future();

    std::lock_guard<std::mutex> lock(mutex_);
    size_t num_features = inference.features.size();
    MG_CHECK(num_features > 0) << "Empty features not supported.";
    queue_counter_ += num_features;
    inference_queue_.push(std::move(inference));
    MaybeRunBatches();

    return future;
  }

 private:
  void MaybeRunBatches() EXCLUSIVE_LOCKS_REQUIRED(queue_mutex) {
    for (;;) {
      size_t batch_size = std::min(queue_counter_ - run_counter_,
                                   static_cast<size_t>(FLAGS_batch_size));

      // Stop if we won't fill a batch yet but more request will come.
      if (static_cast<int>(batch_size) < FLAGS_batch_size &&
          num_clients_ > flush_queue_.size()) {
        break;
      }

      if (batch_size) {
        RunBatch(batch_size);
      }

      // Take elements from flush queue which were scheduled to run.
      while (!flush_queue_.empty() && flush_queue_.front() <= run_counter_) {
        flush_queue_.pop();
      }
    }
  }

  void RunBatch(size_t batch_size) {
    /*
    std::cerr << "Assembling batch (games = " << inference_queue_.size()
              << ", features = " << batch_size
              << ", size = " << FLAGS_batch_size << ")" << std::endl;
    */
    run_counter_ += batch_size;

    BatchFunctor batch_functor;

    std::vector<const DualNet::BoardFeatures*> feature_ptrs;
    std::vector<DualNet::Output*> output_ptrs;
    feature_ptrs.reserve(FLAGS_batch_size);
    output_ptrs.reserve(FLAGS_batch_size);
    auto feature_inserter = std::back_inserter(feature_ptrs);
    auto output_inserter = std::back_inserter(output_ptrs);

    MG_CHECK(batch_size > 0);
    while (batch_size > 0) {
      // Consume features in shared_functor_ first.
      if (shared_functor_) {
        size_t num_features = std::min(shared_size_, batch_size);
        batch_functor.shared_functors.push_back(shared_functor_);
        batch_functor.shared_num_features.push_back(num_features);

        shared_size_ -= num_features;
        batch_size -= num_features;

        DualNet::CopyPointers(
            shared_functor_->functor.features.begin() + shared_size_,
            num_features, feature_inserter);
        DualNet::CopyPointers(
            shared_functor_->functor.outputs.begin() + shared_size_,
            num_features, output_inserter);

        if (shared_size_ == 0) {
          shared_functor_.reset();
        }
        continue;
      }

      auto inference = std::move(inference_queue_.front());
      inference_queue_.pop();
      size_t num_features = inference.features.size();
      DualNet::Functor functor(std::move(inference.features),
                               std::move(inference.promise));

      // If inference doesn't fit into the batch anymore, move it to shared
      // functor and fill up the batch in the next iteration.
      if (num_features > batch_size) {
        shared_size_ = num_features;
        shared_functor_.reset(
            new SharedFunctor{std::move(functor), {num_features}});
        continue;
      }

      // Add entire inference to batch.
      DualNet::CopyPointers(functor.features.begin(), num_features,
                            feature_inserter);
      DualNet::CopyPointers(functor.outputs.begin(), num_features,
                            output_inserter);
      batch_functor.functors.push_back(std::move(functor));
      MG_CHECK(batch_size >= num_features);
      batch_size -= num_features;
    }

    dual_net_->RunManyAsync(std::move(feature_ptrs), std::move(output_ptrs),
                            DualNet::Continuation(std::move(batch_functor)));
    ++num_runs_;
  }

  std::mutex mutex_;

  size_t num_clients_ GUARDED_BY(&mutex_);
  // Values of queue_counter_ when FlushClient() was called.
  std::queue<size_t> flush_queue_ GUARDED_BY(&mutex_);

  std::queue<InferenceData> inference_queue_ GUARDED_BY(&mutex_);
  // Number of features pushed to queue
  size_t queue_counter_ GUARDED_BY(&mutex_);
  // Number of features pushed to dual net.
  size_t run_counter_ GUARDED_BY(&mutex_);

  std::shared_ptr<SharedFunctor> shared_functor_ GUARDED_BY(&mutex_);
  // Number of remaining features in shared_functor_.
  size_t shared_size_ GUARDED_BY(&mutex_);

  size_t num_runs_ GUARDED_BY(&mutex_);
};
}  // namespace

std::unique_ptr<DualNet::Service> NewBatchingService(
    std::unique_ptr<DualNet> dual_net) {
  return absl::make_unique<BatchingService>(std::move(dual_net));
}
}  // namespace minigo
