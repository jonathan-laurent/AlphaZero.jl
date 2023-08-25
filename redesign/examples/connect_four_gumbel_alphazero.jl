using RLZero.BatchedEnvs
using RLZero.Network
using RLZero.Train
using RLZero.TrainUtilities: TrainConfig, print_execution_times
using RLZero.Tests.Common.BitwiseConnectFour
using RLZero.Tests.Common.BitwiseConnectFourEvalFns
using RLZero.Util.Devices

using Flux
using Random

const SAVEDIR = "examples/models/connect-four-checkpoints"

state_dim = BatchedEnvs.state_size(BitwiseConnectFourEnv)
action_dim = BatchedEnvs.num_actions(BitwiseConnectFourEnv)

# large architecture
neural_net_hyperparams = SimpleResNetHP(
    width=512,
    depth_common=6,
    depth_phead=1,
    depth_vhead=1
)
nn_cpu = SimpleResNet(state_dim..., action_dim, neural_net_hyperparams)


"""Returns a list with all the evaluation functions to be called during evaluation sessions.
    Most of these functions are closures, which are being computed by the parent functions
    defined in: Tests/Common/Evaluation/EvaluationFunctions/BitwiseConnectFourEvalFns.jl."""
function get_eval_fns(num_mcts_simulations)

    # This config object will be used by `TrainUtilities.init_mcts_config()`
    #   (in TrainUtilities.jl) to create a `MctsConfig` object for the evaluation functions.
    #   It has to define all the specific-MCTS hyperparameters, even those that won't be
    #   used during evaluation. In this case, only `use_gumbel_mcts`, `num_simulations`,
    #   `num_considered_actions`, `mcts_value_scale` and `mcts_max_visit_init` will be used.
    eval_config = (;
        use_gumbel_mcts=true,
        num_simulations=num_mcts_simulations,
        num_considered_actions=7,
        mcts_value_scale=0.1f0,
        mcts_max_visit_init=1
    )

    az_vs_random_kwargs = Dict(
        "rng" => Random.MersenneTwister(13),
        "num_bilateral_rounds" => 5,
        "config" => deepcopy(eval_config)
    )
    nn_vs_random_kwargs = Dict(
        "rng" => Random.MersenneTwister(13),
        "num_bilateral_rounds" => 10
    )
    az_vs_minimax_kwargs = Dict(
        "stochastic_minimax_depth" => 4,
        "stochastic_minimax_rng" => Random.MersenneTwister(42),
        "num_bilateral_rounds" => 5,
        "config" => deepcopy(eval_config)
    )
    nn_vs_minimax_kwargs = Dict(
        "stochastic_minimax_depth" => 4,
        "stochastic_minimax_rng" => Random.MersenneTwister(42),
        "num_bilateral_rounds" => 50
    )
    benchmark_fns_kwargs = Dict(
        "rng" => Random.MersenneTwister(13),
        "num_states" => 10_000,
        "device" => GPU(),
        "config" => deepcopy(eval_config)
    )

    alphazero_vs_random_eval_fn = get_alphazero_vs_random_eval_fn(az_vs_random_kwargs)
    nn_vs_random_eval_fn = get_nn_vs_random_eval_fn(nn_vs_random_kwargs)
    alphazero_vs_minimax_eval_fn = get_alphazero_vs_minimax_eval_fn(az_vs_minimax_kwargs)
    nn_vs_minimax_eval_fn = get_nn_vs_minimax_eval_fn(nn_vs_minimax_kwargs)
    fns = get_connect_four_benchmark_fns(benchmark_fns_kwargs)
    mcts_benchmark_fn, nn_benchmark_fn = fns

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
    EnvCls = BitwiseConnectFourEnv
    env_kwargs = Dict()
    num_envs = 50_000

    # common MCTS variables
    use_gumbel_mcts = true
    num_simulations = 64

    # Gumbel MCTS variables
    num_considered_actions = 7
    mcts_value_scale = 0.1f0
    mcts_max_visit_init = 10

    # AlphaZero MCTS variables
    # ...we can omit these since we're using Gumbel MCTS

    # NN Training variables
    replay_buffer_size = num_envs * 500
    min_train_samples = 1_000
    train_freq = num_envs * 50
    adam_learning_rate = 1e-3
    weight_decay = 1e-4
    gradient_clip = 1e-3
    batch_size = 50_000
    train_epochs = 1

    # Logging variables
    train_logfile = "train.log"
    eval_logfile = "eval.log"
    tb_logdir = "tb_logs"  # run `tensorboard --logdir tb_logs --port 6006` on a terminal

    # NN saving dir
    nn_save_dir = SAVEDIR

    # Evaluation variables
    evaluation_fns = get_eval_fns(num_simulations)
    eval_freq = num_envs * 50

    # Total train steps
    num_steps = num_envs * 2_000

    return TrainConfig(;
        EnvCls=EnvCls,
        env_kwargs=env_kwargs,
        num_envs=num_envs,

        use_gumbel_mcts=use_gumbel_mcts,
        num_simulations=num_simulations,

        num_considered_actions=num_considered_actions,
        mcts_value_scale=mcts_value_scale,
        mcts_max_visit_init=mcts_max_visit_init,

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
