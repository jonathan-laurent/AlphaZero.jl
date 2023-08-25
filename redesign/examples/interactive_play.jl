using RLZero.BatchedMcts
using RLZero.BatchedEnvs
using RLZero.Network
using RLZero.Tests.Common.BitwiseConnectFour
using RLZero.TrainUtilities: init_mcts_config
using RLZero.Util.Devices

using Flux
using JLD2


const MCTS = BatchedMcts


# set these constants to your preference
const DEVICE = GPU()
const MODEL_PATH = "examples/models/connect-four-checkpoints/model_0000.jld2"
const nn_config = SimpleResNetHP(width=512, depth_common=6, depth_phead=1, depth_vhead=1)


function load_nn()
    state_dim = BatchedEnvs.state_size(BitwiseConnectFourEnv)
    action_dim = BatchedEnvs.num_actions(BitwiseConnectFourEnv)
    nn = SimpleResNet(state_dim..., action_dim, nn_config)

    model_state = JLD2.load(MODEL_PATH, "model_state");
    Flux.loadmodel!(nn, model_state);
    nn = (DEVICE == CPU()) ? Flux.cpu(nn) : Flux.gpu(nn)
    return nn
end

function _mcts_action(env, mcts_config)
    env_vec = DeviceArray(mcts_config.device)([env])
    println("Thinking..")
    tree = MCTS.explore(mcts_config, env_vec)
    action = Array(MCTS.evaluation_policy(tree, mcts_config))[1]
    println("AlphaZero chooses action: $action\n\n")
    return action
end

function _nn_move(env, nn)
    na = BatchedEnvs.num_actions(typeof(env))
    state = BatchedEnvs.vectorize_state(env)
    state = Flux.unsqueeze(state, length(size(state)) + 1)
    on_gpu(nn) && (state = DeviceArray(DEVICE)(state))
    _, logits = forward(nn, state)
    logits = Array(logits)[:, 1]
    invalid_actions = [!BatchedEnvs.valid_action(env, a) for a in 1:na]
    logits[invalid_actions] .= -Inf
    action = argmax(logits)
    println("Raw Neural Network chooses action: $action\n\n")
    return action
end

function _parse_player_action(env)
    na = BatchedEnvs.num_actions(typeof(env))
    valid_actions = findall(!iszero, [BatchedEnvs.valid_action(env, a) for a in 1:na])
    print("Input action: ")
    action = tryparse(Int16, readline())
    while isnothing(action) || action ∉ valid_actions
        print("Invalid action. Please choose one of $valid_actions. Input action: ")
        action = tryparse(Int16, readline())
    end
    println("User chooses action: $action\n\n")
    return action
end

"""Play against an MCTS player using the provided neural network."""
function play_with_mcts(nn, mcts_kwargs, az_goes_first = true)
    mcts_config = init_mcts_config(DEVICE, nn, mcts_kwargs)

    env = BitwiseConnectFourEnv()
    println(env)

    az_plays = az_goes_first
    done, info = false, nothing
    while !done
        action = az_plays ? _mcts_action(env, mcts_config) : _parse_player_action(env)
        env, info = BatchedEnvs.act(env, action)
        info.switched && (az_plays = !az_plays)
        done = BatchedEnvs.terminated(env)
        println(env)
    end

    info.switched && (az_plays = !az_plays)
    last_player = az_plays ? "AlphaZero" : "User"
    println("Game terminated! Last reward: $(info.reward) by player: $last_player.")
end

"""Play against the greedy policy of the neural network."""
function play_with_nn(nn, nn_goes_first)
    env = BitwiseConnectFourEnv()
    println(env)

    nn_plays = nn_goes_first
    done, info = false, nothing
    while !done
        action = nn_plays ? _nn_move(env, nn) : _parse_player_action(env)
        env, info = BatchedEnvs.act(env, action)
        info.switched && (nn_plays = !nn_plays)
        done = BatchedEnvs.terminated(env)
        println(env)
    end

    info.switched && (nn_plays = !nn_plays)
    last_player = nn_plays ? "Neural Network" : "User"
    println("Game terminated! Last reward: $(info.reward) by player: $last_player.")
end


# load the neural network
nn = load_nn()

# set here the important values for the MCTS as a NamedTuple
mcts_kwargs = (;
    # common MCTS variables
    use_gumbel_mcts = true,
    num_simulations = 64,

    # Gumbel MCTS variables
    num_considered_actions = 7,
    mcts_value_scale = 0.1f0,
    mcts_max_visit_init = 50,

    # AlphaZero MCTS variables -- No need to set those since we're using Gumbel, but
    #   they're here for completeness
    c_puct = 1.0f0,
    alpha_dirichlet = 0.3f0,
    epsilon_dirichlet = 0.25f0,
    tau = 1.0f0,
    collapse_tau_move = 30,
)

# play against MCTS
play_with_mcts(nn, mcts_kwargs, true)

# play against the neural network
play_with_nn(nn, true)
