module TrainUtilities

using Base: @kwdef
using Flux
using JLD2

using ...BatchedMctsUtilities
using ...EnvOracles
using ...Network
using ...Util.Devices


export TrainConfig, TrainExecutionTimes
export init_envs, init_mcts_config, save_nn, print_execution_times


"""
    TrainConfig(; kwargs...)

Configuration for training an AlphaZero agent.

# Fields

## Environment variables
- `EnvCls::Type`: The type of the environment to be used. The environment structure should
   follow the `BatchedEnvs` interface.
- `env_kwargs::Dict`: Keyword arguments to be passed when creating the environment instance.
- `num_envs::Int`: The number of environments to be used for parallel simulations.

Note: Environment instances are initialized like so: `env = EnvCls(; env_kwargs...)`

## Common MCTS variables
- `use_gumbel_mcts::Bool = true`: Whether to use Gumbel-MCTS instead of traditional MCTS.
- `num_simulations::Int = 64`: Number of MCTS simulations per move.

Note: The two types of implemented MCTS algorithms can be found in the following links:
- [Gumbel-MCTS](https://openreview.net/pdf?id=bERaNdoegnO)
- [AlphaZero-MCTS](https://arxiv.org/pdf/1712.01815.pdf)

## Gumbel MCTS variables
- `num_considered_actions::Int = 9`: Number of actions considered for selection in
   Gumbel-MCTS.
- `mcts_value_scale::Float32 = 0.1f0`: `c_scale` parameter for Gumbel-MCTS.
- `mcts_max_visit_init::Int = 50`: `c_visit` parameter for Gumbel-MCTS.

## AlphaZero MCTS variables
- `c_puct::Float32 = 1.0f0`: Exploration constant for AlphaZero MCTS.
- `alpha_dirichlet::Float32 = 0.3f0`: Alpha parameter for Dirichlet noise at root nodes.
- `epsilon_dirichlet::Float32 = 0.25f0`: Dirichlet noise weight.
- `tau::Float32 = 1.0f0`: Temperature parameter for action selection.
- `collapse_tau_move::Int = 30`: Action number after which to collapse τ (tau) to 0.

## NN Training variables
- `replay_buffer_size::Int = 500`: Size of the replay buffer. It's advised to express this
   as a multiple of the number of environments.
- `min_train_samples::Int = 1_000`: Minimum number of samples to begin training.
- `train_freq::Int = 500`: Frequency of neural network training. Note that this value
   represents total environment steps, so it's best to express it as a multiple of the
   number of environments.
- `adam_lr::Float32 = 3e-4`: Learning rate for the Adam optimizer.
- `weight_decay::Float32 = 0f0`: L2 regularization weight decay.
- `gradient_clip::Float32 = 1e-3`: Gradient clipping value.
- `batch_size::Int = 50_000`: Batch size for training.
- `train_epochs::Int = 1`: Number of epochs per training step.

## Logging variables
- `train_logfile::String = ""`: File path to save training logs. Unused if equal to "".
- `eval_logfile::String = ""`: File path to save evaluation logs. Unused if equal to "".
- `tb_logdir::String = ""`: Directory for TensorBoard logs. Unused if equal to "".

## NN saving dir
- `nn_save_dir::String = ""`: Directory to save neural network model. Unused if equal to "".

## Evaluation variables
- `eval_fns::Vector{Function} = []`: Vector of evaluation functions.
- `eval_freq::Int = 0`: Frequency of evaluation during training. No evaluation function
   calls will take place if equal to 0.

## Total train steps
- `num_steps::Int = 1_500`: Total number of training steps. Note that this value
   represents total environment steps, so it's best to express it as a multiple of the
   number of environments.
"""
@kwdef struct TrainConfig
    # environment variables
    EnvCls::Type
    env_kwargs::Dict
    num_envs::Int

    # common MCTS variables
    use_gumbel_mcts::Bool = true
    num_simulations::Int = 64

    # Gumbel MCTS variables
    num_considered_actions::Int = 9
    mcts_value_scale::Float32 = 0.1f0
    mcts_max_visit_init::Int = 50

    # AlphaZero MCTS variables
    c_puct::Float32 = 1.0f0
    alpha_dirichlet::Float32 = 0.3f0
    epsilon_dirichlet::Float32 = 0.25f0
    tau::Float32 = 1.0f0
    collapse_tau_move::Int = 30

    # NN Training variables
    replay_buffer_size::Int = 1_500
    min_train_samples::Int = 500
    train_freq::Int = 50
    adam_lr::Float32 = 3e-4
    weight_decay::Float32 = 0f0
    gradient_clip::Float32 = 1e-3
    batch_size::Int = 50_000
    train_epochs::Int = 1

    # Logging variables
    train_logfile::String = ""
    eval_logfile::String = ""
    tb_logdir::String = ""

    # NN saving dir
    nn_save_dir::String = ""

    # Evaluation variables
    eval_fns::Vector{Function} = []
    eval_freq::Int = 0

    # Total train steps
    num_steps::Int = 1_500
end

"""
    init_envs(config::TrainConfig, num_envs::Int, device::Device)

Initializes and returns arrays contraining the environments to be used for training and
the number-of-episode-steps counter for every environments. Both arrays are placed on the
`device` device.
"""
function init_envs(config::TrainConfig, num_envs::Int, device::Device)
    envs = DeviceArray(device)([config.EnvCls(; config.env_kwargs...) for _ in 1:num_envs])
    steps_counter = zeros(Int, device, num_envs)
    return envs, steps_counter
end

"""
    init_mcts_config(device::Device, nn::Net, config) where Net <: FluxNetwork

Initializes and returns an MCTS configuration object based on the `config` configuration
object. The `nn` neural network will be used in the environment oracle.

Note: This function can be dispatched on `config` object, as it can either be of type
`TrainConfig` or just a NamedTuple containing the important fields for the appropriate MCTS
configuration structure. The NamedTuple version will be used during evaluation to avoid
having to create an unnecessary `TrainConfig` object.
"""
function init_mcts_config(device::Device, nn::Net, config) where Net <: FluxNetwork
    oracle = neural_network_env_oracle(; nn)
    if config.use_gumbel_mcts
        return GumbelMctsConfig(;
            device,
            oracle,
            num_simulations=config.num_simulations,
            num_considered_actions=config.num_considered_actions,
            value_scale=config.mcts_value_scale,
            max_visit_init=config.mcts_max_visit_init
        )
    else
        return AlphaZeroMctsConfig(;
            device,
            oracle,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            alpha_dirichlet=config.alpha_dirichlet,
            epsilon_dirichlet=config.epsilon_dirichlet,
            tau=config.tau,
            collapse_tau_move=config.collapse_tau_move
        )
    end
end

"""
    save_nn(nn::Net, save_dir::String, step::Int, steps::Int) where Net <: FluxNetwork

Saves the `nn` neural network to the `save_dir` directory. The `step` and `steps` arguments
are used to create a filename for the saved model. Filenames will have padded 0's in front
to preserve the order of the files.

Notes:
- The `save_dir` directory is created if it doesn't exist.
- The model is transferred to the CPU before saving.
"""
function save_nn(nn::Net, save_dir::String, step::Int, steps::Int) where Net <: FluxNetwork
    !isdir(save_dir) && mkpath(save_dir)
    model_state = Flux.state(Flux.cpu(nn))
    step_str = lpad(step, length(string(steps)), "0")
    jldsave("$save_dir/model_$step_str.jld2"; model_state)
end

"""
    TrainExecutionTimes

Structure for storing execution times of important components of AlphaZeros's training loop.
"""
struct TrainExecutionTimes
    explore_times::Array{Float64}
    selection_times::Array{Float64}
    step_save_reset_times::Array{Float64}
    train_times::Array{Float64}
    eval_times::Array{Float64}
end

function TrainExecutionTimes(total_steps)
    explore_times = zeros(Float64, total_steps)
    selection_times = zeros(Float64, total_steps)
    step_save_reset_times = zeros(Float64, total_steps)
    train_times = zeros(Float64, total_steps)
    eval_times = zeros(Float64, total_steps)
    return TrainExecutionTimes(
        explore_times,
        selection_times,
        step_save_reset_times,
        train_times,
        eval_times
    )
end

function print_execution_times(execution_times::TrainExecutionTimes)
    batch_steps = length(execution_times.explore_times)

    total_exp_time = sum(execution_times.explore_times)
    avg_exp_time = total_exp_time / batch_steps

    total_select_time = round(sum(execution_times.selection_times), digits=4)
    avg_select_time = round(total_select_time / batch_steps, digits=4)

    total_ssr_time = round(sum(execution_times.step_save_reset_times), digits=4)
    avg_ssr_time = round(total_ssr_time / batch_steps, digits=4)

    train_times = [t for t in execution_times.train_times if t != zero(Float64)]
    total_train_time = round(sum(train_times), digits=4)
    avg_train_time = round(total_train_time / length(train_times), digits=4)

    eval_times = [t for t in execution_times.eval_times if t != zero(Float64)]
    total_eval_time = round(sum(eval_times), digits=4)
    avg_eval_time = round(total_eval_time / length(eval_times), digits=4)

    println("Total explore time: $total_exp_time seconds.")
    println("Total selection time: $total_select_time seconds.")
    println("Total step-save-reset time: $total_ssr_time seconds.")
    println("Total train time: $total_train_time seconds.")
    println("Total eval time: $total_eval_time seconds.")
    println()
    println("Average explore time: $avg_exp_time seconds.")
    println("Average selection time: $avg_select_time seconds.")
    println("Average step-save-reset time: $avg_ssr_time seconds.")
    println("Average train time: $avg_train_time seconds.")
    println("Average eval time: $avg_eval_time seconds.")

    train_loop_time = total_exp_time + total_select_time + total_ssr_time + total_train_time
    train_loop_time = round(train_loop_time, digits=4)
    println("\n")
    println("Total train loop time: $train_loop_time seconds.")
end

end
