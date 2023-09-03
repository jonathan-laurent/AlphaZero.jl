using RLZero.BatchedEnvs
using RLZero.Network
using RLZero.Train
using RLZero.TrainUtilities: TrainConfig, print_execution_times
using RLZero.Tests.Common.BitwiseRandomWalk1D
using RLZero.Tests.Common.BitwiseRandomWalk1DEvalFns
using RLZero.Util.Devices

using Flux
using Random

const SAVEDIR = "examples/models/random-walk-1d-checkpoints"

state_dim = BatchedEnvs.state_size(BitwiseRandomWalk1DEnv)
action_dim = BatchedEnvs.num_actions(BitwiseRandomWalk1DEnv)

# minimal architecture
neural_net_hyperparams = SimpleNetHP(
    width=32,
    depth_common=1,
    depth_phead=1,
    depth_vhead=1
)
nn_cpu = SimpleNet(state_dim..., action_dim, neural_net_hyperparams)


"""Returns a list with all the evaluation functions to be called during evaluation sessions.
    Since RandomWalk is a quite simple environment, only the NN will be evaluated. All
    the evaluation functions for the BitwiseRandomWalk1D environment can be found in:
    Tests/Common/Evaluation/EvaluationFunctions/BitwiseRandomWalk1DEvalFns.jl."""
function get_eval_fns()
    return [evaluate_nn]
end


"""Returns a `TrainConfig` object. All the hyperparameters can be set here."""
function create_config()

    # environment variables
    EnvCls = BitwiseRandomWalk1DEnv
    env_kwargs = Dict()
    num_envs = 500

    # common MCTS variables
    use_gumbel_mcts = true
    num_simulations = 4

    # Gumbel MCTS variables
    num_considered_actions = 2
    # mcts_value_scale = 0.1f0
    mcts_value_scale = 1f0
    mcts_max_visit_init = 10

    # AlphaZero MCTS variables
    # ...we can omit these since we're using Gumbel MCTS

    # NN Training variables
    replay_buffer_size = num_envs * 50
    min_train_samples = 10
    train_freq = num_envs * 25
    adam_learning_rate = 1e-3
    gradient_clip = 1e-3
    batch_size = 1_000
    train_epochs = 1

    # Logging variables
    train_logfile = "train.log"
    eval_logfile = "eval.log"
    tb_logdir = "tb_logs"  # run `tensorboard --logdir tb_logs --port 6006` on a terminal

    # NN saving dir
    nn_save_dir = SAVEDIR

    # Evaluation variables
    evaluation_fns = get_eval_fns()
    eval_freq = num_envs * 25

    # Total train steps
    num_steps = num_envs * 200

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
device = CPU()

# get the NN in the specified device
nn = (device == CPU()) ? nn_cpu : Flux.gpu(nn_cpu)

# create the config object
config = create_config()

# train!
nn, execution_times = selfplay!(config, device, nn)

# print some statistics
println("\n")
print_execution_times(execution_times)
