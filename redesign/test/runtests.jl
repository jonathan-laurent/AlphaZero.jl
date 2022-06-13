using Test
using RLZero
using ReinforcementLearningBase
using ReinforcementLearningEnvironments

using Random: MersenneTwister
using Statistics: mean

include("TestUtil/TestUtil.jl")
using .TestUtil

include("runtests/mcts.jl")
