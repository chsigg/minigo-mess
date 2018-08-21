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

#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/check.h"
#include "cc/constants.h"
#include "cc/dual_net/factory.h"
#include "cc/file/filesystem.h"
#include "cc/file/path.h"
#include "cc/gtp_player.h"
#include "cc/init.h"
#include "cc/mcts_player.h"
#include "cc/random.h"
#include "cc/sgf.h"
#include "cc/tf_utils.h"
#include "gflags/gflags.h"

// Game options flags.
DEFINE_string(mode, "", "Mode to run in: \"selfplay\", \"eval\" or \"gtp\"");
DEFINE_int32(
    ponder_limit, 0,
    "If non-zero and in GTP mode, the number times of times to perform tree "
    "search while waiting for the opponent to play.");
DEFINE_bool(
    courtesy_pass, false,
    "If true and in GTP mode, we will always pass if the opponent passes.");
DEFINE_double(resign_threshold, -0.999, "Resign threshold.");
DEFINE_double(komi, minigo::kDefaultKomi, "Komi.");
DEFINE_double(disable_resign_pct, 0.1,
              "Fraction of games to disable resignation for.");
DEFINE_uint64(seed, 0,
              "Random seed. Use default value of 0 to use a time-based seed. "
              "This seed is used to control the moves played, not whether a "
              "game has resignation disabled or is a holdout.");

// Tree search flags.
DEFINE_int32(num_readouts, 100,
             "Number of readouts to make during tree search for each move.");
DEFINE_int32(virtual_losses, 8,
             "Number of virtual losses when running tree search.");
DEFINE_bool(inject_noise, true,
            "If true, inject noise into the root position at the start of "
            "each tree search.");
DEFINE_bool(soft_pick, true,
            "If true, choose moves early in the game with a probability "
            "proportional to the number of times visited during tree search. "
            "If false, always play the best move.");
DEFINE_bool(random_symmetry, true,
            "If true, randomly flip & rotate the board features before running "
            "the model and apply the inverse transform to the results.");
DEFINE_string(flags_path, "",
              "Optional path to load flags from. Flags specified in this file "
              "take priority over command line flags. When running selfplay "
              "with run_forever=true, the flag file is reloaded periodically. "
              "Note that flags_path is different from gflags flagfile, which "
              "is only parsed once on startup.");

// Time control flags.
DEFINE_double(seconds_per_move, 0,
              "If non-zero, the number of seconds to spend thinking about each "
              "move instead of using a fixed number of readouts.");
DEFINE_double(
    time_limit, 0,
    "If non-zero, the maximum amount of time to spend thinking in a game: we "
    "spend seconds_per_move thinking for each move for as many moves as "
    "possible before exponentially decaying the amount of time.");
DEFINE_double(decay_factor, 0.98,
              "If time_limit is non-zero, the decay factor used to shorten the "
              "amount of time spent thinking as the game progresses.");
DEFINE_bool(run_forever, false,
            "When running 'selfplay' mode, whether to run forever.");

// Inference flags.
DEFINE_string(model, "",
              "Path to a minigo model. If remote_inference=false, the model "
              "should be a serialized GraphDef proto. If "
              "remote_inference=true, the model should be saved checkpoint.");
DEFINE_string(model_two, "",
              "When running 'eval' mode, provide a path to a second minigo "
              "model, also serialized as a GraphDef proto.");
DEFINE_int32(parallel_games, 32,
             "Number of games to play in parallel. For performance reasons, "
             "parallel_games should equal games_per_inference * 2 because this "
             "allows the transfer of inference requests & responses to be "
             "overlapped with model evaluation.");
DEFINE_int32(game_threads, 16, "Number of threads that play games.");

// Output flags.
DEFINE_string(output_dir, "",
              "Output directory. If empty, no examples are written.");
DEFINE_string(holdout_dir, "",
              "Holdout directory. If empty, no examples are written.");
DEFINE_string(sgf_dir, "", "SGF directory. If empty, no SGF is written.");
DEFINE_double(holdout_pct, 0.03,
              "Fraction of games to hold out for validation.");

// Self play flags:
//   --inject_noise=true
//   --soft_pick=true
//   --random_symmetery=true
//
// Two player flags:
//   --inject_noise=false
//   --soft_pick=false
//   --random_symmetry=true

namespace minigo {
namespace {

std::string GetOutputName(absl::Time now, size_t i) {
  auto timestamp = absl::ToUnixSeconds(now);
  std::string output_name;
  char hostname[64];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    std::strncpy(hostname, "unknown", sizeof(hostname));
  }
  return absl::StrCat(timestamp, "-", hostname, "-", i);
}

std::string GetOutputDir(absl::Time now, const std::string& root_dir) {
  auto sub_dirs = absl::FormatTime("%Y-%m-%d-%H", now, absl::UTCTimeZone());
  return file::JoinPath(root_dir, sub_dirs);
}

void WriteExample(const std::string& output_dir, const std::string& output_name,
                  const MctsPlayer& player) {
  MG_CHECK(file::RecursivelyCreateDir(output_dir));

  // Write the TensorFlow examples.
  std::vector<tensorflow::Example> examples;
  examples.reserve(player.history().size());
  DualNet::BoardFeatures features;
  std::vector<const Position::Stones*> recent_positions;
  for (const auto& h : player.history()) {
    h.node->GetMoveHistory(DualNet::kMoveHistory, &recent_positions);
    DualNet::SetFeatures(absl::MakeConstSpan(recent_positions),
                         h.node->position.to_play(), &features);
    examples.push_back(
        tf_utils::MakeTfExample(features, h.search_pi, player.result()));
  }

  auto output_path = file::JoinPath(output_dir, output_name + ".tfrecord.zz");
  tf_utils::WriteTfExamples(output_path, absl::MakeConstSpan(examples));
}

void WriteSgf(const std::string& output_dir, const std::string& output_name,
              const MctsPlayer& player_b, const MctsPlayer& player_w,
              bool write_comments) {
  MG_CHECK(file::RecursivelyCreateDir(output_dir));
  MG_CHECK(player_b.history().size() == player_w.history().size());

  bool log_names = player_b.name() != player_w.name();

  std::vector<sgf::MoveWithComment> moves;
  moves.reserve(player_b.history().size());

  for (size_t i = 0; i < player_b.history().size(); ++i) {
    const auto& h = i % 2 == 0 ? player_b.history()[i] : player_w.history()[i];
    const auto& color = h.node->position.to_play();
    std::string comment;
    if (write_comments) {
      if (i == 0) {
        comment = absl::StrCat(
            "Resign Threshold: ", player_b.options().resign_threshold, "\n",
            h.comment);
      } else {
        if (log_names) {
          comment = absl::StrCat(i % 2 == 0 ? player_b.name() : player_w.name(),
                                 "\n", h.comment);
        } else {
          comment = h.comment;
        }
      }
      moves.emplace_back(color, h.c, std::move(comment));
    } else {
      moves.emplace_back(color, h.c, "");
    }
  }

  std::string player_name =
      static_cast<std::string>(file::Basename(FLAGS_model));
  sgf::CreateSgfOptions options;
  options.komi = player_b.options().komi;
  options.result = player_b.result_string();
  options.black_name = player_b.name();
  options.white_name = player_w.name();
  auto sgf_str = sgf::CreateSgfString(absl::MakeConstSpan(moves), options);

  auto output_path = file::JoinPath(output_dir, output_name + ".sgf");
  TF_CHECK_OK(tf_utils::WriteFile(output_path, sgf_str));
}

void WriteSgf(const std::string& output_dir, const std::string& output_name,
              const MctsPlayer& player, bool write_comments) {
  WriteSgf(output_dir, output_name, player, player, write_comments);
}

void ParseMctsPlayerOptionsFromFlags(MctsPlayer::Options* options) {
  options->inject_noise = FLAGS_inject_noise;
  options->soft_pick = FLAGS_soft_pick;
  options->random_symmetry = FLAGS_random_symmetry;
  options->resign_threshold = FLAGS_resign_threshold;
  options->virtual_losses = FLAGS_virtual_losses;
  options->komi = FLAGS_komi;
  options->random_seed = FLAGS_seed;
  options->num_readouts = FLAGS_num_readouts;
  options->seconds_per_move = FLAGS_seconds_per_move;
  options->time_limit = FLAGS_time_limit;
  options->decay_factor = FLAGS_decay_factor;
}

class SelfPlayer {
 public:
  void Run() {
    auto start_time = absl::Now();
    network_ = NewDualNetService(FLAGS_model);
    int games_per_thread =
        (FLAGS_parallel_games + FLAGS_game_threads - 1) / FLAGS_game_threads;
    for (int i = FLAGS_parallel_games, thread_id = 0; i > 0; ++thread_id) {
      int num_games = std::min(i, games_per_thread);
      threads_.emplace_back(
          std::bind(&SelfPlayer::ThreadRun, this, thread_id, num_games));
      i -= num_games;
    }
    for (auto& t : threads_) {
      t.join();
    }
    std::cerr << "All threads stopped, total time "
              << absl::ToDoubleSeconds(absl::Now() - start_time) << " sec."
              << std::endl;
  }

 private:
  // Struct that holds the options for a game. Each thread has its own
  // GameOptions instance, which are initialized with the SelfPlayer's mutex
  // held. This allows us to safely update the command line arguments from a
  // flag file without causing any race conditions.
  struct GameOptions {
    void Init() {
      ParseMctsPlayerOptionsFromFlags(&player_options);

      run_forever = FLAGS_run_forever;
      holdout_pct = FLAGS_holdout_pct;
      output_dir = FLAGS_output_dir;
      holdout_dir = FLAGS_holdout_dir;
      sgf_dir = FLAGS_sgf_dir;
    }

    void IncrementSeed(size_t increment) {
      if (player_options.random_seed != 0) {
        player_options.random_seed += increment;
      }
    }

    MctsPlayer::Options player_options;
    bool run_forever;
    float holdout_pct;
    std::string output_dir;
    std::string holdout_dir;
    std::string sgf_dir;
  };

  void ThreadRun(int thread_id, size_t num_games) {
    pthread_setname_np(pthread_self(), "Player");

    // Only print the board using ANSI colors if stderr is sent to the
    // terminal.
    const bool use_ansi_colors = isatty(fileno(stderr));

    GameOptions game_options;
    do {
      {
        absl::MutexLock lock(&mutex_);
        auto old_model = FLAGS_model;
        MaybeReloadFlags();
        MG_CHECK(old_model == FLAGS_model)
            << "Manually changing the model during selfplay is not supported. "
               "Use --checkpoint_dir and --engine=remote to perform inference "
               "using the most recent checkpoint from training.";
        game_options.Init();
      }

      auto client = absl::make_unique<DualNet::Client>(network_.get());

      // Create players. If an random seed was explicitly specified, offset it
      // with a different value for each player.
      game_options.IncrementSeed(1299283 * thread_id);
      std::vector<std::unique_ptr<MctsPlayer>> players;
      players.reserve(num_games);
      for (size_t i = 0; i < num_games; ++i) {
        game_options.player_options.verbose = false;  //! i && !thread_id;
        players.push_back(absl::make_unique<MctsPlayer>(
            client.get(), game_options.player_options));
        game_options.IncrementSeed(9491);
      }

      // Play the game.
      auto start_time = absl::Now();
      size_t num_moves = 0;
      while (!players.empty()) {
        std::vector<std::unique_ptr<MctsPlayer::AsyncStepper>> steppers;
        steppers.reserve(players.size());
        for (const auto& player : players) {
          steppers.push_back(
              absl::make_unique<MctsPlayer::AsyncStepper>(player.get()));
        }

        do {
          client->Flush();
          steppers.erase(
              std::remove_if(
                  steppers.begin(), steppers.end(),
                  [&](const std::unique_ptr<MctsPlayer::AsyncStepper>&
                          stepper) {
                    if (auto move_opt = stepper->TryGet()) {
                      auto* player = stepper->player();
                      if (player->options().verbose) {
                        std::cerr << player->root()->position.ToPrettyString(
                            use_ansi_colors);
                        std::cerr << player->root()->Describe() << std::endl;
                      }
                      player->PlayMove(move_opt.value());
                      return true;
                    }
                    return false;
                  }),
              steppers.end());
        } while (!steppers.empty());

        num_moves += players.size();
        auto last =
            std::remove_if(players.begin(), players.end(),
                           [](const std::unique_ptr<MctsPlayer>& player) {
                             return player->game_over();
                           });
        for (auto it = last; it != players.end(); ++it) {
          const auto& player = *it;

          // Write the outputs.
          auto now = absl::Now();
          auto output_name = GetOutputName(now, thread_id);

          bool is_holdout = [&] {
            absl::MutexLock lock(&mutex_);
            return rnd_();
          }() < game_options.holdout_pct;
          auto example_dir =
              is_holdout ? game_options.holdout_dir : game_options.output_dir;
          if (!example_dir.empty()) {
            WriteExample(GetOutputDir(now, example_dir), output_name, *player);
          }

          if (!game_options.sgf_dir.empty()) {
            WriteSgf(GetOutputDir(
                         now, file::JoinPath(game_options.sgf_dir, "clean")),
                     output_name, *player, false);
            WriteSgf(
                GetOutputDir(now, file::JoinPath(game_options.sgf_dir, "full")),
                output_name, *player, true);
          }
        }
        players.erase(last, players.end());
      }

      std::cerr << "Thread " << thread_id << " played " << num_games
                << " games with " << num_moves << " total moves in "
                << absl::ToDoubleSeconds(absl::Now() - start_time) << " sec."
                << std::endl;

    } while (game_options.run_forever);
  }

  void MaybeReloadFlags() EXCLUSIVE_LOCKS_REQUIRED(&mutex_) {
    if (FLAGS_flags_path.empty()) {
      return;
    }
    uint64_t new_flags_timestamp;
    TF_CHECK_OK(tf_utils::GetModTime(FLAGS_flags_path, &new_flags_timestamp));
    if (new_flags_timestamp == flags_timestamp_) {
      return;
    }

    flags_timestamp_ = new_flags_timestamp;
    std::string contents;
    TF_CHECK_OK(tf_utils::ReadFile(FLAGS_flags_path, &contents));

    for (auto line : absl::StrSplit(contents, '\n')) {
      std::pair<absl::string_view, absl::string_view> line_comment =
          absl::StrSplit(line, absl::MaxSplits('#', 1));
      line = absl::StripAsciiWhitespace(line_comment.first);
      if (line.empty()) {
        continue;
      }
      MG_CHECK(line.length() > 2 && line[0] == '-' && line[1] == '-') << line;
      std::pair<std::string, std::string> flag_value =
          absl::StrSplit(line, absl::MaxSplits('=', 1));
      flag_value.first = flag_value.first.substr(2);
      std::cerr << "Setting command line flag: --" << flag_value.first << "="
                << flag_value.second << std::endl;
      gflags::SetCommandLineOption(flag_value.first.c_str(),
                                   flag_value.second.c_str());
    }
  }

  std::unique_ptr<DualNet::Service> network_;

  absl::Mutex mutex_;
  Random rnd_ GUARDED_BY(&mutex_);
  std::vector<std::thread> threads_;
  uint64_t flags_timestamp_ = 0;
};

void SelfPlay() { SelfPlayer().Run(); }

void Eval() {
  MctsPlayer::Options options;
  ParseMctsPlayerOptionsFromFlags(&options);
  options.inject_noise = false;
  options.soft_pick = false;
  options.random_symmetry = true;

  options.name = static_cast<std::string>(file::Stem(FLAGS_model));
  auto black_network = NewDualNetService(FLAGS_model);
  auto black_client = absl::make_unique<DualNet::Client>(black_network.get());
  auto black = absl::make_unique<MctsPlayer>(black_client.get(), options);

  options.name = static_cast<std::string>(file::Stem(FLAGS_model_two));
  auto white_network = NewDualNetService(FLAGS_model_two);
  auto white_client = absl::make_unique<DualNet::Client>(white_network.get());
  auto white = absl::make_unique<MctsPlayer>(white_client.get(), options);

  auto* player = black.get();
  auto* other_player = white.get();
  while (!player->game_over()) {
    auto move = player->SuggestMove();
    std::cerr << player->root()->Describe() << "\n";
    player->PlayMove(move);
    other_player->PlayMove(move);
    std::cerr << player->root()->position.ToPrettyString();
    std::swap(player, other_player);
  }
  std::cerr << player->result_string() << "\n";
  std::cerr << "Black was: " << black->name() << "\n";

  std::string output_name = absl::StrCat(GetOutputName(absl::Now(), 0), "-",
                                         black->name(), "-", white->name());

  // Write SGF.
  if (!FLAGS_sgf_dir.empty()) {
    WriteSgf(FLAGS_sgf_dir, output_name, *black, *white, true);
  }
}

void Gtp() {
  GtpPlayer::Options options;
  ParseMctsPlayerOptionsFromFlags(&options);

  options.name = absl::StrCat("minigo-", file::Basename(FLAGS_model));
  options.ponder_limit = FLAGS_ponder_limit;
  options.courtesy_pass = FLAGS_courtesy_pass;
  auto network = NewDualNetService(FLAGS_model);
  auto client = absl::make_unique<DualNet::Client>(network.get());
  auto player = absl::make_unique<GtpPlayer>(client.get(), options);
  player->Run();
}

}  // namespace

}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);

  if (FLAGS_mode == "selfplay") {
    minigo::SelfPlay();
  } else if (FLAGS_mode == "eval") {
    minigo::Eval();
  } else if (FLAGS_mode == "gtp") {
    minigo::Gtp();
  } else {
    std::cerr << "Unrecognized mode \"" << FLAGS_mode << "\"\n";
    return 1;
  }

  return 0;
}
