#####
##### Analytical reports for debugging and hyperparameters tuning
#####

module Report

struct Loss
  L :: Float32
  Lp :: Float32
  Lv :: Float32
  Lreg :: Float32
  Linv :: Float32
end

struct LearningStatus
  loss :: Loss
  Hpnet :: Float32
end

struct Checkpoint
  epoch_id :: Int
  reward :: Float64
  nn_replaced :: Bool
end

struct Epoch
  status_after :: LearningStatus
end

struct Learning
  time_convert :: Float64
  time_loss :: Float64
  time_train :: Float64
  time_eval :: Float64
  initial_status :: LearningStatus
  epochs :: Vector{Epoch}
  checkpoints :: Vector{Checkpoint}
  nn_replaced :: Bool
end

struct Samples
  num_samples :: Int
  num_boards :: Int
  Wtot :: Float32
  Hp :: Float32
  status :: LearningStatus
end

struct StageSamples
  mean_remaining_length :: Float64
  samples_stats :: Samples
end

struct Memory
  latest_batch :: Samples
  all_samples :: Samples
  per_game_stage :: Vector{StageSamples}
end

struct SelfPlay
  inference_time_ratio :: Float64
  samples_gen_speed :: Float64 # in samples/second
  mcts_memory_footprint :: Int
end

struct Iteration
  time_self_play :: Float64
  time_memory_analysis :: Float64
  time_learning :: Float64
  self_play :: SelfPlay
  memory :: Memory
  learning :: Learning
end

struct Initial
  num_network_parameters :: Int
  mcts_footprint_per_node :: Int
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
  ("Lreg",   NUM_COL,     s -> s.loss.Lreg),
  ("Linv",   NUM_COL,     s -> s.loss.Linv),
  ("Hpnet",  NUM_COL,     s -> s.Hpnet))

const SAMPLES_STATS_TABLE = Log.Table(
  ("Loss",   NUM_COL,     s -> s.status.loss.L),
  ("Lv",     NUM_COL,     s -> s.status.loss.Lv),
  ("Lp",     NUM_COL,     s -> s.status.loss.Lp),
  ("Lreg",   NUM_COL,     s -> s.status.loss.Lreg),
  ("Linv",   NUM_COL,     s -> s.status.loss.Linv),
  ("Hpnet",  NUM_COL,     s -> s.status.Hpnet),
  ("Hp",     NUM_COL,     s -> s.Hp),
  ("Wtot",   BIGINT_COL,  s -> s.Wtot),
  ("Nb",     BIGINT_COL,  s -> s.num_boards),
  ("Ns",     BIGINT_COL,  s -> s.num_samples))

function print(logger::Logger, status::Report.LearningStatus; kw...)
  Log.table_row(logger, LEARNING_STATUS_TABLE, status; kw...)
end

function print(logger::Logger, stats::Report.Memory)
  content, styles, comments = [], [], []
  # All samples
  push!(content, stats.all_samples)
  push!(styles, Log.BOLD)
  push!(comments, ["all samples"])
  # Latest batch
  push!(content, stats.latest_batch)
  push!(styles, Log.BOLD)
  push!(comments, ["latest batch"])
  # Per game stage
  for stage in stats.per_game_stage
    rem = fmt(".1f", stage.mean_remaining_length)
    push!(content, stage.samples_stats)
    push!(styles, Log.NO_STYLE)
    push!(comments, ["$rem turns left"])
  end
  Log.table(
    logger, SAMPLES_STATS_TABLE, content, styles=styles, comments=comments)
end

function print(logger::Logger, report::SelfPlay)
  t = round(Int, 100 * report.inference_time_ratio)
  Log.print(logger, "Time spent on inference: $(t)%")
  sspeed = format(round(Int, report.samples_gen_speed), commas=true)
  Log.print(logger, "Generating $(sspeed) samples per second on average")
  memf = format(report.mcts_memory_footprint, autoscale=:metric, precision=2)
  Log.print(logger, "MCTS memory footprint: $(memf)B")
end

function print(logger::Logger, report::Initial)
  nnparams = format(report.num_network_parameters, commas=true)
  Log.print(logger, "Number of network parameters: $nnparams")
  mfpn = report.mcts_footprint_per_node
  Log.print(logger, "Memory footprint per MCTS node: $(mfpn) bytes")
end

end
