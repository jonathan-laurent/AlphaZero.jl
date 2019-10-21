#####
##### Analytical reports for debugging and hyperparameters tuning
#####

module Report

struct Initial
  num_network_parameters :: Int
  mcts_footprint_per_node :: Int
end

struct Loss
  L :: Float64
  Lp :: Float64
  Lv :: Float64
  Lreg :: Float64
  Linv :: Float64
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
  Wtot :: Float64
  loss :: Loss
  Hp :: Float64
  Hp̂ :: Float64
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
  ("MaxW",   NUM_COL,     s -> s.network.maxw),
  ("MeanW",  NUM_COL,     s -> s.network.meanw))

const SAMPLES_STATS_TABLE = Log.Table(
  ("Loss",   NUM_COL,     s -> s.loss.L),
  ("Lv",     NUM_COL,     s -> s.loss.Lv),
  ("Lp",     NUM_COL,     s -> s.loss.Lp),
  ("Lreg",   NUM_COL,     s -> s.loss.Lreg),
  ("Linv",   NUM_COL,     s -> s.loss.Linv),
  ("Hp",     NUM_COL,     s -> s.Hp),
  ("Hpnet",  NUM_COL,     s -> s.Hp̂),
  ("Λtot",   BIGINT_COL,  s -> s.Wtot),
  ("Nb",     BIGINT_COL,  s -> s.num_boards),
  ("Ns",     BIGINT_COL,  s -> s.num_samples))

function print(logger::Logger, status::Report.LearningStatus, args...; kw...)
  Log.table_row(logger, LEARNING_STATUS_TABLE, status, args...; kw...)
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
