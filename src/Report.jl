#####
##### Analytical reports for debugging and hyperparameters tuning
#####

module Report

struct Loss
  L :: Float64
  Lp :: Float64
  Lv :: Float64
  Lreg :: Float64
end

struct Network
  maxw  :: Float64
  meanw :: Float64
  pbiases :: Vector{Float64}
  vbias :: Float64
end

struct LearningStatus
  loss :: Loss
  network :: Network
end

struct Game
  reward :: Float64
  length :: Int
end

struct Evaluation
  games :: Vector{Game}
  average_reward :: Float64 # redundant
end

struct Checkpoint
  epoch_id :: Int
  time_eval :: Float64
  eval :: Evaluation
end

struct Epoch
  time_train :: Float64
  time_loss :: Float64
  status_after :: LearningStatus
end

struct Samples
  num_samples :: Int
  num_boards :: Int
  Wtot :: Float64
  loss :: Loss
  Hp :: Float64
  Hp̂ :: Float64
end

struct Memory
  latest_batch :: Samples
  all_samples :: Samples
  # Average remaining turns, stats
  per_game_stage :: Vector{Tuple{Float64, Samples}}
end

struct Learning
  time_convert :: Float64
  initial_status :: LearningStatus
  epochs :: Vector{Epoch}
  checkpoints :: Vector{Checkpoint}
  nn_replaced :: Bool
end

struct SelfPlay
  games :: Vector{Game}
  inference_time_ratio :: Float64
end

struct Iteration
  time_self_play :: Float64
  time_learning :: Float64
  self_play :: SelfPlay
  memory :: Memory
  learning :: Learning
end

struct Training
  num_nn_params :: Int
  iterations :: Vector{Iteration}
end

#####
##### Printing and plotting
#####

using Formatting
using ..Log

const NUM_COL = Log.ColType(7, x -> fmt(".4f", x))
const BIGINT_COL = Log.ColType(10, n -> format(ceil(Int, n), commas=true))

const LEARNING_STATUS_TABLE = Log.Table(
  ("Loss",   NUM_COL,     s -> s.loss.L),
  ("Lv",     NUM_COL,     s -> s.loss.Lv),
  ("Lp",     NUM_COL,     s -> s.loss.Lp),
  ("MaxW",   NUM_COL,     s -> s.network.maxw),
  ("MeanW",  NUM_COL,     s -> s.network.meanw))

const SAMPLES_STATS_TABLE = Log.Table(
  ("Loss",   NUM_COL,     s -> s.loss.L),
  ("Lv",     NUM_COL,     s -> s.loss.Lv),
  ("Lp",     NUM_COL,     s -> s.loss.Lp),
  ("Hp",     NUM_COL,     s -> s.Hp),
  ("Hpnet",  NUM_COL,     s -> s.Hp̂),
  ("Λtot",   BIGINT_COL,  s -> s.Wtot),
  ("Nb",     BIGINT_COL,  s -> s.num_boards),
  ("Ns",     BIGINT_COL,  s -> s.num_samples))

function print(logger::Logger, status::Report.LearningStatus, args...; kw...)
  Log.table_row(logger, LEARNING_STATUS_TABLE, status, args...; kw...)
end

function print(logger::Logger, stats::Report.Samples, args...; kw...)
  Log.table_row(logger, SAMPLES_STATS_TABLE, stats, args...; kw...)
end

function print(logger::Logger, stats::Report.Memory)
  print(logger, stats.all_samples, ["all samples"], style=Log.BOLD)
  print(logger, stats.latest_batch, ["latest batch"], style=Log.BOLD)
  for (t, stats) in stats.per_game_stage
    rem = fmt(".1f", t)
    print(logger, stats, ["$rem turns left"])
  end
end

function print(logger::Logger, stats::SelfPlay)
  t = round(Int, 100 * stats.inference_time_ratio)
  Log.print(logger, "Time spent on inference: $(t)%")
end

end
