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
  loss :: Float64
  Hp :: Float64
  Hp̂ :: Float64
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
end

struct Iteration
  time_self_play :: Float64
  time_learning :: Float64
  self_play :: SelfPlay
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

const LEARNING_STATUS_TABLE =
  let num_fmt(x) = fmt(".4f", x)
    Log.Table((
      "Loss"  => num_fmt,
      "Lp"    => num_fmt,
      "Lv"    => num_fmt,
      "MaxW"  => num_fmt,
      "MeanW" => num_fmt))
  end

function print_learning_status(
    logger::Logger, status::Report.LearningStatus, comments=[])
  Log.table_row(logger, LEARNING_STATUS_TABLE, (
    "Loss"  => status.loss.L,
    "Lp"    => status.loss.Lp,
    "Lv"    => status.loss.Lv,
    "MaxW"  => status.network.maxw,
    "MeanW" => status.network.meanw
  ), comments)
end

const SAMPLES_STATS_TABLE =
  let num_fmt(x) = fmt(".4f", x)
  let bigint_fmt(n) = format(n, width=8, autoscale=:metric)
    Log.Table((
      "L"  => num_fmt,
      "Lv" => num_fmt,
      "Lp" => num_fmt,
      "Hp" => num_fmt,
      "Hp̂" => num_fmt,
      "Nb" => bigint_fmt,
      "Ns" => bigint_fmt,
      "W"  => bigint_fmt
    ))
  end end

function print_samples_stats(
    logger::Logger, stats::Report.Samples, comments=[])
  Log.table_row(logger, SAMPLES_STATS_TABLE, (
    "L"  => stats.loss.L,
    "Lp" => stats.loss.Lp,
    "Lv" => stats.loss.Lv,
    "Hp" => stats.Hp,
    "Hp̂" => stats.Hp̂,
    "Nb" => stats.num_boards,
    "Ns" => stats.num_samples,
    "W"  => stats.Wtot
  ), comments)
end

end
