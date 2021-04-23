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

Scripts.test_grad_updates(exp, num_games=5000)