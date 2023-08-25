using RLZero.BatchedEnvs
using RLZero.Network
using RLZero.Train
using RLZero.TrainUtilities: TrainConfig, print_execution_times
using RLZero.Tests.Common.BitwiseTicTacToe
using RLZero.Tests.Common.BitwiseTicTacToeEvalFns
using RLZero.Util.Devices

using Flux
using Random

const SAVEDIR = "examples/models/tictactoe-checkpoints"

state_dim = BatchedEnvs.state_size(BitwiseTicTacToeEnv)
action_dim = BatchedEnvs.num_actions(BitwiseTicTacToeEnv)

# small architecture
neural_net_hyperparams = SimpleNetHP(
    width=128,
    depth_common=6,
    depth_phead=1,
    depth_vhead=1,
    use_batch_norm=true,
    batch_norm_momentum=0.6f0
)
nn_cpu = SimpleNet(state_dim..., action_dim, neural_net_hyperparams)


"""Returns a list with all the evaluation functions to be called during evaluation sessions.
    Most of these functions are closures, which are being computed by the parent functions
    defined in: Tests/Common/Evaluation/EvaluationFunctions/BitwiseTicTacToeEvalFns.jl."""
function get_eval_fns(num_mcts_simulations)

    # This config object will be used by `TrainUtilities.init_mcts_config()`
    #   (in TrainUtilities.jl) to create a `MctsConfig` object for the evaluation functions.
    #   It has to define all the specific-MCTS hyperparameters, even those that won't be
    #   used during evaluation. In this case, only `use_gumbel_mcts`, `num_simulations` and
    #   `c_puct` will be used. The other traditional-MCTS hyperparams will be ignored.
    eval_config = (;
        use_gumbel_mcts=false,
        num_simulations=num_mcts_simulations,
        c_puct=1.5f0,
        alpha_dirichlet=0.10f0,
        epsilon_dirichlet=0.25f0,
        tau=1.0f0,
        collapse_tau_move=7
    )

    az_vs_random_kwargs = Dict(
        "rng" => Random.MersenneTwister(13),
        "num_bilateral_rounds" => 10,
        "config" => deepcopy(eval_config)
    )
    nn_vs_random_kwargs = Dict(
        "rng" => Random.MersenneTwister(13),
        "num_bilateral_rounds" => 10
    )
    az_vs_minimax_kwargs = Dict(
        "num_bilateral_rounds" => 10,
        "config" => deepcopy(eval_config)
    )
    nn_vs_minimax_kwargs = Dict("num_bilateral_rounds" => 10)
    benchmark_fns_kwargs = Dict(
        "device" => GPU(),
        "config" => deepcopy(eval_config)
    )

    # get the actual evaluation functions as closures
    alphazero_vs_random_eval_fn = get_alphazero_vs_random_eval_fn(az_vs_random_kwargs)
    nn_vs_random_eval_fn = get_nn_vs_random_eval_fn(nn_vs_random_kwargs)
    alphazero_vs_minimax_eval_fn = get_alphazero_vs_minimax_eval_fn(az_vs_minimax_kwargs)
    nn_vs_minimax_eval_fn = get_nn_vs_minimax_eval_fn(nn_vs_minimax_kwargs)
    mcts_benchmark_fn, nn_benchmark_fn = get_tictactoe_benchmark_fns(benchmark_fns_kwargs)

    return [
        alphazero_vs_random_eval_fn,
        nn_vs_random_eval_fn,
        alphazero_vs_minimax_eval_fn,
        nn_vs_minimax_eval_fn,
        mcts_benchmark_fn,
        nn_benchmark_fn
    ]
end


"""Returns a `TrainConfig` object. All the hyperparameters can be set here."""
function create_config()

    # environment variables
    EnvCls = BitwiseTicTacToeEnv
    env_kwargs = Dict()
    num_envs = 25_000

    # common MCTS variables
    use_gumbel_mcts = false
    num_simulations = 64

    # Gumbel MCTS variables
    # ...we can omit these since we're using Traditional Alphazero MCTS

    # AlphaZero MCTS variables
    c_puct = 1.5f0
    alpha_dirichlet = 0.10f0
    epsilon_dirichlet = 0.25f0
    tau = 1.0f0
    collapse_tau_move = 7

    # NN Training variables
    replay_buffer_size = num_envs * 90
    min_train_samples = 1_000
    train_freq = num_envs * 10
    adam_learning_rate = 1e-3
    weight_decay = 1e-4
    gradient_clip = 1e-3
    batch_size = 25_000
    train_epochs = 2

    # Logging variables
    train_logfile = "train.log"
    eval_logfile = "eval.log"
    tb_logdir = "tb_logs"  # run `tensorboard --logdir tb_logs --port 6006` on a terminal

    # NN saving dir
    nn_save_dir = SAVEDIR

    # Evaluation variables
    evaluation_fns = get_eval_fns(num_simulations)
    eval_freq = num_envs * 10

    # Total train steps
    num_steps = num_envs * 90

    return TrainConfig(;
        EnvCls=EnvCls,
        env_kwargs=env_kwargs,
        num_envs=num_envs,

        use_gumbel_mcts=use_gumbel_mcts,
        num_simulations=num_simulations,

        c_puct=c_puct,
        alpha_dirichlet=alpha_dirichlet,
        epsilon_dirichlet=epsilon_dirichlet,
        tau=tau,
        collapse_tau_move=collapse_tau_move,

        replay_buffer_size=replay_buffer_size,
        min_train_samples=min_train_samples,
        train_freq=train_freq,
        adam_lr=adam_learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        batch_size=batch_size,
        train_epochs=train_epochs,

        train_logfile=train_logfile,
        eval_logfile=eval_logfile,
        tb_logdir=tb_logdir,

        nn_save_dir=nn_save_dir,

        eval_fns=evaluation_fns,
        eval_freq=eval_freq,

        num_steps=num_steps
    )
end

# empty the save directory
run(`rm -rf $(SAVEDIR)`)

# choose the device to train AlphaZero on (`CPU()` or `GPU()`)
device = GPU()

# get the NN in the specified device
nn = (device == CPU()) ? nn_cpu : Flux.gpu(nn_cpu)

# create the config object
config = create_config()

# train!
nn, execution_times = selfplay!(config, device, nn)

# print some statistics
println("\n")
print_execution_times(execution_times)
