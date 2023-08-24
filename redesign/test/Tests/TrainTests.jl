module TrainTests

using CUDA
using Flux
using Random
using Test

using ..Common.BitwiseRandomWalk1D
using ..Common.BitwiseTicTacToe
using ..Common.BitwiseConnectFour
using ...BatchedEnvs
using ...Network
using ...Train
using ...TrainUtilities
using ...Util.Devices

import ..Common.BitwiseTicTacToeEvalFns as TTTEvalFns
import ..Common.BitwiseConnectFourEvalFns as CFEvalFns


export run_train_tests


function run_train_tests()
    @testset "Training a BitwiseRandomWalk1DEnv agent" test_train_bitwise_randomwalk1d()
    @testset "Training a TicTacToeEnv agent" test_train_bitwise_tictactoe()
    @testset "Training a ConnectFourEnv agent" test_train_bitwise_connect_four()
end

function test_train_bitwise_randomwalk1d()

    function get_random_walk_1d_config(num_envs)
        return TrainConfig(;
            EnvCls=BitwiseRandomWalk1DEnv,
            env_kwargs=Dict(),
            num_envs=num_envs,
            use_gumbel_mcts=true,
            num_simulations=8,
            num_considered_actions=2,
            mcts_value_scale=0.1f0,
            mcts_max_visit_init=3,
            min_train_samples=10,
            batch_size=10,
            train_freq=50,
            train_logfile="train.log",
            eval_logfile="eval.log",
            tb_logdir="tb_logs",
            num_steps=30
        )
    end

    function get_random_walk_1d_nn(device)
        state_dim = BatchedEnvs.state_size(BitwiseRandomWalk1DEnv)
        action_dim = BatchedEnvs.num_actions(BitwiseRandomWalk1DEnv)

        hp = SimpleNetHP(width=32, depth_common=2)
        nn = SimpleNet(state_dim..., action_dim, hp)

        (device == GPU()) && (nn = Flux.gpu(nn))
        return nn
    end

    @testset "RandomWalk1D: CPU Train" begin
        config = get_random_walk_1d_config(10)
        nn = get_random_walk_1d_nn(CPU())
        selfplay!(config, CPU(), nn, false)
        run(`rm train.log eval.log`)
        run(`rm -rf tb_logs`)
        @test true
    end

    CUDA.functional() && @testset "RandomWalk1D: GPU Train" begin
        config = get_random_walk_1d_config(10)
        nn = get_random_walk_1d_nn(GPU())
        selfplay!(config, GPU(), nn, false)
        run(`rm train.log eval.log`)
        run(`rm -rf tb_logs`)
        @test true
    end
end

function test_train_bitwise_tictactoe()

    function get_tictactoe_config(num_envs)
        function get_eval_fns()
            eval_config = (;
                use_gumbel_mcts=true,
                num_simulations=64,
                num_considered_actions=9,
                mcts_value_scale=0.1f0,
                mcts_max_visit_init=5
            )

            az_vs_random_kwargs = Dict(
                "rng" => Random.MersenneTwister(13),
                "num_bilateral_rounds" => 10,
                "config" => deepcopy(eval_config)
            )
            nn_vs_minimax_kwargs = Dict("num_bilateral_rounds" => 10)

            return [
                TTTEvalFns.get_alphazero_vs_random_eval_fn(az_vs_random_kwargs),
                TTTEvalFns.get_nn_vs_minimax_eval_fn(nn_vs_minimax_kwargs)
            ]
        end

        return TrainConfig(;
            EnvCls=BitwiseTicTacToeEnv,
            env_kwargs=Dict(),
            num_envs=num_envs,
            use_gumbel_mcts=true,
            num_simulations=64,
            num_considered_actions=9,
            mcts_value_scale=0.1f0,
            mcts_max_visit_init=5,
            min_train_samples=10,
            batch_size=10,
            train_freq=10,
            train_logfile="train.log",
            eval_logfile="eval.log",
            tb_logdir="tb_logs",
            eval_freq=10,
            eval_fns=get_eval_fns(),
            num_steps=50
        )
    end

    function get_tictactoe_nn(device)
        state_dim = BatchedEnvs.state_size(BitwiseTicTacToeEnv)
        action_dim = BatchedEnvs.num_actions(BitwiseTicTacToeEnv)

        hp = SimpleNetHP(width=128, depth_common=6)
        nn = SimpleNet(state_dim..., action_dim, hp)

        (device == GPU()) && (nn = Flux.gpu(nn))
        return nn
    end

    @testset "TicTacToe: CPU Train" begin
        config = get_tictactoe_config(5)
        nn = get_tictactoe_nn(CPU())
        selfplay!(config, CPU(), nn, false)
        run(`rm train.log eval.log`)
        run(`rm -rf tb_logs`)
        @test true
    end

    CUDA.functional() && @testset "TicTacToe: GPU Train" begin
        config = get_tictactoe_config(5)
        nn = get_tictactoe_nn(GPU())
        selfplay!(config, GPU(), nn, false)
        run(`rm train.log eval.log`)
        run(`rm -rf tb_logs`)
        @test true
    end
end

function test_train_bitwise_connect_four()

    function get_connect_four_config(num_envs)
        function get_eval_fns()
            eval_config = (;
                use_gumbel_mcts=false,
                num_simulations=64,
                c_puct=1.5f0,
                alpha_dirichlet=0.10f0,  # -------------------------------------------------
                epsilon_dirichlet=0.25f0,  # -----------------------------------------------
                tau=1.0f0,  # these values won't be used during evaluation but must be set
                collapse_tau_move=7  # -----------------------------------------------------
            )

            nn_vs_random_kwargs = Dict(
                "rng" => Random.MersenneTwister(13),
                "num_bilateral_rounds" => 10
            )
            az_vs_minimax_kwargs = Dict(
                "stochastic_minimax_depth" => 4,
                "stochastic_minimax_rng" => Random.MersenneTwister(42),
                "num_bilateral_rounds" => 10,
                "config" => deepcopy(eval_config)
            )

            return [
                CFEvalFns.get_nn_vs_random_eval_fn(nn_vs_random_kwargs),
                CFEvalFns.get_alphazero_vs_minimax_eval_fn(az_vs_minimax_kwargs)
            ]
        end

        return TrainConfig(;
            EnvCls=BitwiseConnectFourEnv,
            env_kwargs=Dict(),
            num_envs=num_envs,
            use_gumbel_mcts=false,
            num_simulations=64,
            c_puct=1.5f0,
            alpha_dirichlet=0.10f0,
            epsilon_dirichlet=0.25f0,
            tau=1.0f0,
            collapse_tau_move=7,
            min_train_samples=100,
            batch_size=10,
            train_freq=50,
            train_logfile="train.log",
            eval_logfile="eval.log",
            tb_logdir="tb_logs",
            eval_freq=50,
            eval_fns=get_eval_fns(),
            num_steps=100
        )
    end

    function get_connect_four_nn(device)
        state_dim = BatchedEnvs.state_size(BitwiseConnectFourEnv)
        action_dim = BatchedEnvs.num_actions(BitwiseConnectFourEnv)

        hp = SimpleResNetHP(width=256, depth_common=4)
        nn = SimpleResNet(state_dim..., action_dim, hp)

        (device == GPU()) && (nn = Flux.gpu(nn))
        return nn
    end

    @testset "Connect Four: CPU Train" begin
        config = get_connect_four_config(13)
        nn = get_connect_four_nn(CPU())
        selfplay!(config, CPU(), nn, false)
        run(`rm train.log eval.log`)
        run(`rm -rf tb_logs`)
        @test true
    end

    CUDA.functional() && @testset "Connect Four: GPU Train" begin
        config = get_connect_four_config(13)
        nn = get_connect_four_nn(GPU())
        selfplay!(config, GPU(), nn, false)
        run(`rm train.log eval.log`)
        run(`rm -rf tb_logs`)
        @test true
    end
end

end
