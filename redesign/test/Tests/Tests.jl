module Tests

using Reexport
using Test

export run_all_tests

include("Common/Common.jl")
@reexport using .Common

include("UtilTests.jl")
@reexport using .UtilTests: run_util_tests

include("NetworksTests/NetworksTests.jl")
@reexport using .NetworksTests: run_neural_networks_tests

include("ReplayBufferTests.jl")
@reexport using .ReplayBufferTests: run_replay_buffer_tests

include("EnvTests/EnvTests.jl")
@reexport using .EnvTests: run_env_tests

include("MinimaxTests.jl")
@reexport using .MinimaxTests: run_minimax_tests

include("MctsTests/MctsTests.jl")
@reexport using .MctsTests: run_mcts_tests

include("TrainTests.jl")
@reexport using .TrainTests: run_train_tests

function run_all_tests()
    @testset "RLZero tests" begin
        @testset "utility tests" run_util_tests()
        @testset "neural network tests" run_neural_networks_tests()
        @testset "replay buffer tests" run_replay_buffer_tests()
        @testset "env tests" run_env_tests()
        @testset "minimax tests" run_minimax_tests()
        @testset "mcts tests" run_mcts_tests()
        @testset "train tests" run_train_tests()
    end
    return nothing
end

end
