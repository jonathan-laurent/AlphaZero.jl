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
  Hp :: Float32 # property of the memory, constant during a learning iteration
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
  average_exploration_depth :: Float64
  mcts_memory_footprint :: Int
  # Basic memory statistics
  memory_size :: Int
  memory_num_distinct_boards :: Int
end

struct Perfs
  time :: Float64
  allocated :: Int64
  gc_time :: Float64
end

struct Iteration
  perfs_self_play :: Perfs
  perfs_memory_analysis :: Perfs
  perfs_learning :: Perfs
  self_play :: SelfPlay
  memory :: Union{Memory, Nothing}
  learning :: Learning
end

struct Initial
  num_network_parameters :: Int
  num_network_regularized_parameters :: Int
  mcts_footprint_per_node :: Int
end

#####
##### Profiling utilities
#####

macro timed(e)
  quote
    local v, t, mem, gct = Base.@timed $(esc(e))
    v, Perfs(t, mem, gct)
  end
end

#####
##### Printing and plotting
#####

using Formatting
using ..Log

const NUM_COL = Log.ColType(7, x -> fmt(".4f", x))
const BIGINT_COL = Log.ColType(10, n -> format(ceil(Int, n), commas=true))

const LEARNING_STATUS_TABLE = Log.Table([
  ("Loss",   NUM_COL,     s -> s.loss.L),
  ("Lv",     NUM_COL,     s -> s.loss.Lv),
  ("Lp",     NUM_COL,     s -> s.loss.Lp),
  ("Lreg",   NUM_COL,     s -> s.loss.Lreg),
  ("Linv",   NUM_COL,     s -> s.loss.Linv),
  ("Hp",     NUM_COL,     s -> s.Hp),
  ("Hpnet",  NUM_COL,     s -> s.Hpnet)])

const SAMPLES_STATS_TABLE = Log.Table([
  ("Loss",   NUM_COL,     s -> s.status.loss.L),
  ("Lv",     NUM_COL,     s -> s.status.loss.Lv),
  ("Lp",     NUM_COL,     s -> s.status.loss.Lp),
  ("Lreg",   NUM_COL,     s -> s.status.loss.Lreg),
  ("Linv",   NUM_COL,     s -> s.status.loss.Linv),
  ("Hpnet",  NUM_COL,     s -> s.status.Hpnet),
  ("Hp",     NUM_COL,     s -> s.Hp),
  ("Wtot",   BIGINT_COL,  s -> s.Wtot),
  ("Nb",     BIGINT_COL,  s -> s.num_boards),
  ("Ns",     BIGINT_COL,  s -> s.num_samples)])

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
  avgdepth = fmt(".1f", report.average_exploration_depth)
  Log.print(logger, "Average exploration depth: $avgdepth")
  memf = format(report.mcts_memory_footprint, autoscale=:metric, precision=2)
  Log.print(logger, "MCTS memory footprint: $(memf)B")
  mems = format(report.memory_size, commas=true)
  memd = format(report.memory_num_distinct_boards, commas=true)
  Log.print(logger, "Experience buffer size: $(mems) ($(memd) distinct boards)")
end

function print(logger::Logger, report::Initial)
  nnparams = format(report.num_network_parameters, commas=true)
  Log.print(logger, "Number of network parameters: $nnparams")
  nnregparams = format(report.num_network_regularized_parameters, commas=true)
  Log.print(logger, "Number of regularized network parameters: $nnregparams")
  mfpn = report.mcts_footprint_per_node
  Log.print(logger, "Memory footprint per MCTS node: $(mfpn) bytes")
end

end
