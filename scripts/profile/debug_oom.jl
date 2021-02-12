#####
##### Script to replicate the mysterious OOM errors
#####

# NUM_FILTERS=64 julia --project scripts/profile/debug_oom.jl
# NUM_FILTERS=128 julia --project scripts/profile/debug_oom.jl

using AlphaZero
using Setfield

if !haskey(ENV, "NUM_FILTERS")
  error("You should set the NUM_FILTERS env variable.")
end

num_filters = parse(Int, ENV["NUM_FILTERS"])

println("Number of filters: $num_filters")

exp = Examples.experiments["connect-four"]

exp = @set exp.netparams.num_filters = num_filters

session = Session(exp)

# Generate states

n = 5000

println("Generating $n some random traces")

traces = [play_game(exp.gspec, RandomPlayer()) for i in 1:n]

for trace in traces
  AlphaZero.push_trace!(session.env.memory, trace, 1.0)
end

println("Starting a learning iteration")

AlphaZero.learning_step!(session.env, session)