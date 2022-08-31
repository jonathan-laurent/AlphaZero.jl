module MuZero

using Flux
using ParameterSchedulers: Scheduler, Cos

using ..BatchedEnvs
using ..Tests
using ..Util.Devices
using ..Storage
using ..TrainableEnvOracles

export MuZeroTrainableEnvOracle

struct MuZeroTrainableEnvOracle <: TrainableEnvOracle
    neural_networks
end

function make_target(history, state_index, target_end_unroll, train_settings)
    map(state_index:target_end_unroll) do current_index
        bootstrap_index = current_index + train_settings.td_steps
        value = if (bootstrap_index > length(history))
            0
        else
            history.values[bootstrap_index] * train_settings.discount^train_settings.td_steps
        end

        borned_bootstrap_index = min(length(history), bootstrap_index)
        for (i, reward) in enumerate(history.rewards[current_index:borned_bootstrap_index])
            value += reward * train_settings.discount^(i - 1)
        end

        reward = if (current_index > 1 && current_index <= (length(history) + 1))
            history.rewards[current_index - 1]
        else
            0
        end

        policy = if (current_index > length(history))
            length_policy = length(first(history.policies))
            ones(Float32, length_policy) / length_policy
        else
            history.policies[current_index]
        end
        return (value, reward, policy)
    end
end

function TrainableEnvOracles.make_feature_and_target(
    history::GameHistory, ::MuZeroTrainableEnvOracle, state_index, train_settings
)
    target_end_unroll = state_index + train_settings.num_unroll_steps
    # Unroll of actions has one less element
    end_unroll = min(length(history), target_end_unroll - 1)

    targets = make_target(history, state_index, target_end_unroll, train_settings)
    actions = history.actions[state_index:end_unroll]
    state = history.states[state_index]

    return (state, actions, targets)
end

"""
### Michael's Code ###
# Would need a (great) refacto to improve code readbility
"""
##### Representation #####

using Base: @kwdef, ident_cmp
using Statistics: mean
using Flux:
    Chain, Dense, Conv, BatchNorm, SkipConnection, flatten, relu, elu, softmax, unstack
using Zygote: Zygote
using CUDA

module GameInterface

    export AbstractGameSpec, AbstractGameEnv

    abstract type AbstractGameSpec end
    abstract type AbstractGameEnv end

    function actions end
    function vectorize_state end

    function state_dim(game_spec::AbstractGameSpec)
        return (27,)
    end

    num_actions(game_spec::AbstractGameSpec) = length(actions(game_spec))
    function encode_action end

end

GI = GameInterface

to_singletons(x) = reshape(x, size(x)..., 1)
from_singletons(x) = reshape(x, size(x)[1:(end - 1)])

abstract type AbstractPrediction end  # gspec, hyper, common, policyhead, valuehead
abstract type AbstractDynamics end    # gspec, hyper, common, statehead, rewardhead
abstract type AbstractRepresentation end # gspec, hyper, common

function forward(nn::AbstractRepresentation, observation)
    hiddenstate = nn.common(observation)
    return hiddenstate
end

function evaluate(nn::AbstractRepresentation, observation)
    x = GI.vectorize_state(nn.gspec, observation)
    # TODO: convert_input for GPU usage
    xnet = to_singletons(x)
    net_output = forward(nn, xnet)
    hiddenstate = from_singletons(net_output)
    #hiddenstate = net_output
    return hiddenstate
end

(nn::AbstractRepresentation)(observation) = evaluate(nn, observation)

function evaluate_batch(nn::AbstractRepresentation, batch)
    X = Flux.batch(GI.vectorize_state(nn.gspec, b) for b in batch)
    Xnet = to_nndevice(nn, X)
    net_outputs = forward(nn, Xnet)
    Hiddenstates = from_nndevice(nn, net_outputs)
    batchdim = ndims(Hiddenstates)
    return unstack(Hiddenstates, batchdim)
end

### Dynamics ###

function forward(nn::AbstractDynamics, hiddenstate_action)
    c = nn.common(hiddenstate_action)
    r = nn.rewardhead(c)
    s⁺¹ = nn.statehead(c)
    return (r, s⁺¹)
end

#TODO add GPU support
function evaluate(nn::AbstractDynamics, hiddenstate, action)
    snet = to_singletons(hiddenstate)
    batchdim = ndims(snet)
    if batchdim == 2
        avalactions = Base.OneTo(length(GI.actions(nn.gspec)))
        encoded_action = onehot(action, avalactions)
    else
        encoded_action = GI.encode_action(nn.gspec, action)
    end
    # gspec = nn.gspec
    # action_one_hot = onehot(action, GI.actions(gspec))
    anet = to_singletons(encoded_action)
    dim = ndims(snet)
    dim == 2 && (anet = Flux.flatten(anet))
    xnet = cat(snet, anet; dims=dim - 1) # make dims=3 universal
    net_output = forward(nn, xnet)
    r, s = from_singletons.(net_output)
    return (r[1], s)
end

(nn::AbstractDynamics)((hiddenstate, action)) = evaluate(nn, hiddenstate, action)

function onehot(x::Integer, labels::Base.OneTo; type=Float32)
    result = zeros(type, length(labels))
    result[x] = one(type)
    return result
end

function encode_a(gspec, a; batchdim=4)
    if batchdim == 2
        avalactions = Base.OneTo(length(GI.actions(gspec)))
        ret_a = onehot(a, avalactions)
    else
        ret_a = GI.encode_action(gspec, a)
    end
    return ret_a
end

function evaluate_batch(nn::AbstractDynamics, batch)
    S = Flux.batch(b[1] for b in batch)
    batchdim = ndims(S)
    A = Flux.batch(encode_a(nn.gspec, b[2]; batchdim) for b in batch)
    batchdim == 2 && (A = Flux.flatten(A))
    X = cat(S, A; dims=batchdim - 1)
    Xnet = to_nndevice(nn, X)
    net_outputs = forward(nn, Xnet)
    (R, S⁺¹) = from_nndevice(nn, net_outputs)
    # # 
    # R_itr = unstack_itr(R, 2)
    # S⁺¹_itr = unstack_itr(S⁺¹, batchdim)
    # return collect(zip(R_itr, S⁺¹_itr))
    # return [(R[1,i], S⁺¹[:,i]) for i in eachindex(batch)]
    # return [(R[1,i], S⁺¹[:,:,:,i]) for i in eachindex(batch)]
    # return [(R[1,i], collect(selectdim(S⁺¹,batchdim,i))) for i in eachindex(batch)]
    return collect(zip(unstack(R, 2), unstack(S⁺¹, batchdim)))
end

### Prediction ###

function forward(nn::AbstractPrediction, hiddenstate)
    c = nn.common(hiddenstate)
    v = nn.valuehead(c)
    p = nn.policyhead(c)
    return (p, v)
end

function evaluate(nn::AbstractPrediction, hiddenstate)
    x = hiddenstate
    xnet = to_singletons(x)
    net_output = forward(nn, xnet)
    p, v = from_singletons.(net_output)
    return (p, v[1])
end

(nn::AbstractPrediction)(hiddenstate) = evaluate(nn, hiddenstate)

function evaluate_batch(nn::AbstractPrediction, batch)
    X = Flux.batch(batch)
    Xnet = to_nndevice(nn, X)
    net_output = forward(nn, Xnet)
    P, V = from_nndevice(nn, net_output)
    return collect(zip(unstack(P, 2), unstack(V, 2)))
end

# TODO create constructor from gspec, hidden_state_shape... and get rid of gspecs
struct MuNetworkHP{GameSpec,Fhp,Ghp,Hhp}
    # hidden_state_shape
    gspec::GameSpec
    predictionHP::Fhp
    dynamicsHP::Ghp
    representationHP::Hhp
end

struct MuNetwork{F<:AbstractPrediction,G<:AbstractDynamics,H<:AbstractRepresentation}
    params::MuNetworkHP
    f::F
    g::G
    h::H
end

function MuNetwork(params::MuNetworkHP)
    fHP = params.predictionHP
    gHP = params.dynamicsHP
    hHP = params.representationHP
    # @assert fHP.hiddenstate_shape == gHP.hiddenstate_shape == hHP.hiddenstate_shape
    f = PredictionNetwork(params.gspec, fHP)
    g = DynamicsNetwork(params.gspec, gHP)
    h = RepresentationNetwork(params.gspec, hHP)
    return MuNetwork(params, f, g, h)
end

# takes output form neural netrork back to CPU, and unstack it along last dimmension
function convert_output(X)
    X = from_nndevice(nothing, X)
    return Flux.unstack(X, ndims(X))
end

struct InitialOracle{H<:AbstractRepresentation,F<:AbstractPrediction}
    h::H
    f::F
end

InitialOracle(nns::MuNetwork) = InitialOracle(nns.h, nns.f)

(init::InitialOracle)(observation) = evaluate(init, observation)

# function evaluate_batch(init::InitialOracle, batch)
#   X = Flux.batch(GI.vectorize_state(init.f.gspec, b) for b in batch) #obsrvation
#   Xnet = to_nndevice(init.f, X)
#   S⁰ = forward(init.h, Xnet) # hiddenstate
#   P⁰, V⁰ = forward(init.f, S⁰) # policy, value

#   P⁰, V⁰, S⁰ = map(convert_output, (P⁰, V⁰, S⁰))
#   V⁰ = [v[1] for v in V⁰]
#   return collect(zip(P⁰, V⁰, S⁰, zero(V⁰)))
# end
function evaluate_batch(init::InitialOracle, observations)
    X = Flux.batch(observations)
    Xnet = to_nndevice(init.f, X)
    S⁰ = forward(init.h, Xnet) # hiddenstate
    P⁰, V⁰ = forward(init.f, S⁰) # policy, value
    P⁰, V⁰, S⁰ = map(convert_output, (P⁰, V⁰, S⁰))
    V⁰ = [v[1] for v in V⁰]
    return P⁰, V⁰, S⁰
end

#TODO test nonmutable struct
struct RecurrentOracle{G<:AbstractDynamics,F<:AbstractPrediction}
    g::G
    f::F
end

RecurrentOracle(nns::MuNetwork) = RecurrentOracle(nns.g, nns.f)

(recur::RecurrentOracle)((state, action)) = evaluate(recur, (state, action))

function evaluate_batch(recur::RecurrentOracle, observations, actions)
    batchdim = ndims(observations)
    A = Flux.batch(encode_a(recur.f.gspec, action; batchdim) for action in actions)
    S_A = cat(observations, A; dims=batchdim - 1)
    S_A_net = to_nndevice(recur.f, S_A) # assuming all networks are on the same device
    R, S⁺¹ = forward(recur.g, S_A_net)
    P⁺¹, V⁺¹ = forward(recur.f, S⁺¹)
    P⁺¹, V⁺¹, R, S⁺¹ = map(convert_output, (P⁺¹, V⁺¹, R, S⁺¹))
    V⁺¹ = first.(V⁺¹)
    R = first.(R)
    return P⁺¹, V⁺¹, S⁺¹, R
end
# TODO cleanup the rest
evaluate(nn, x) = evaluate_batch(nn, [x])[1]

# TODO regularized params
regularized_params_(l) = []
regularized_params_(l::Flux.Dense) = [l.weight]
regularized_params_(l::Flux.Conv) = [l.weight]

function regularized_params(net::MuNetwork)
    return (w for l in Flux.modules(net) for w in regularized_params_(l))
end

#adhoc
function regularized_params(net)
    return (w for l in Flux.modules(net.f.net) for w in regularized_params_(l))
end

array_on_gpu(::Array) = false
array_on_gpu(::CuArray) = true
array_on_gpu(arr) = error("Usupported array type: ", typeof(arr))

using Flux: Conv, SkipConnection, relu
"""
    ResNetHP

Hyperparameters for the convolutional resnet architecture.

| Parameter                 | Type                | Default   |
|:--------------------------|:--------------------|:----------|
| `num_blocks`              | `Int`               |  -        |
| `num_filters`             | `Int`               |  -        |
| `conv_kernel_size`        | `Tuple{Int, Int}`   |  -        |
| `num_policy_head_filters` | `Int`               | `2`       |
| `num_value_head_filters`  | `Int`               | `1`       |
| `batch_norm_momentum`     | `Float32`           | `0.6f0`   |

The trunk of the two-head network consists of `num_blocks` consecutive blocks.
Each block features two convolutional layers with `num_filters` filters and
with kernel size `conv_kernel_size`. Note that both kernel dimensions must be
odd.

During training, the network is evaluated in training mode on the whole
dataset to compute the loss before it is switched to test model, using
big batches. Therefore, it makes sense to use a high batch norm momentum
(put a lot of weight on the latest measurement).

# AlphaGo Zero Parameters

The network in the original paper from Deepmind features 20 blocks with 256
filters per convolutional layer.
"""
@kwdef struct ResNetHP_
    indim::Tuple{Int,Int,Int}
    outdim_head1::Int
    num_blocks::Int
    num_filters::Int
    conv_kernel_size::Tuple{Int,Int}
    num_head1_filters::Int = 2 # policy
    num_head2_filters::Int = 1 # value
    batch_norm_momentum::Float32 = 0.6f0
end

"""
    ResNet <: TwoHeadNetwork

The convolutional residual network architecture that is used
in the original AlphaGo Zero paper.
"""
mutable struct ResNet_
    gspec
    hyper
    common
    head1 #policy/state (vector/tensor like)
    head2 #value/reward (scalar like)
end

function ResNetBlock(size, n, bnmom)
    pad = size .÷ 2
    layers = Chain(
        Conv(size, n => n; pad=pad, bias=false),
        BatchNorm(n, relu; momentum=bnmom),
        Conv(size, n => n; pad=pad, bias=false),
        BatchNorm(n; momentum=bnmom),
    )
    return Chain(SkipConnection(layers, +), x -> relu.(x))
    # relu)
end

function ResNet_(gspec::GI.AbstractGameSpec, hyper::ResNetHP_)
    indim = hyper.indim
    outdim = hyper.outdim_head1
    ksize = hyper.conv_kernel_size
    @assert all(ksize .% 2 .== 1)
    pad = ksize .÷ 2
    nf = hyper.num_filters
    npf = hyper.num_head1_filters
    nvf = hyper.num_head2_filters
    bnmom = hyper.batch_norm_momentum
    common = Chain(
        Conv(ksize, indim[3] => nf; pad=pad, bias=false),
        BatchNorm(nf, relu; momentum=bnmom),
        [ResNetBlock(ksize, nf, bnmom) for i in 1:(hyper.num_blocks)]...,
    )
    phead = Chain(
        Conv((1, 1), nf => npf; bias=false),
        BatchNorm(npf, relu; momentum=bnmom),
        flatten,
        Dense(indim[1] * indim[2] * npf, outdim),
        softmax,
    )
    vhead = Chain(
        Conv((1, 1), nf => nvf; bias=false),
        BatchNorm(nvf, relu; momentum=bnmom),
        flatten,
        Dense(indim[1] * indim[2] * nvf, nf, relu),
        Dense(nf, 1, tanh),
    )
    return ResNet_(gspec, hyper, common, phead, vhead)
end

### Representation ###

@kwdef struct RepresentationResnetHP
    num_blocks::Int
    num_filters::Int
    conv_kernel_size::Tuple{Int,Int}
    batch_norm_momentum::Float32 = 0.6f0
end

mutable struct RepresentationResnetNetwork <: AbstractRepresentation
    gspec
    hyper
    common
end

function RepresentationNetwork(gspec::GI.AbstractGameSpec, hyper::RepresentationResnetHP)
    indim = GI.state_dim(gspec)
    resnethyper = ResNetHP_(
        indim,
        1, # outdim not used
        hyper.num_blocks,
        hyper.num_filters,
        hyper.conv_kernel_size,
        1, # num_head1_filters not used
        1, # num_head2_filters not used
        hyper.batch_norm_momentum,
    )
    resnet = ResNet_(gspec, resnethyper)
    common = Chain(resnet.common, Conv((1, 1), hyper.num_filters => indim[3], relu))
    return RepresentationResnetNetwork(gspec, hyper, common)
end

### Dynamics ###

@kwdef struct DynamicsResnetHP
    num_blocks::Int
    num_filters::Int
    conv_kernel_size::Tuple{Int,Int}
    num_reward_head2_filters::Int = 1 # head2
    batch_norm_momentum::Float32 = 0.6f0
end

mutable struct DynamicsResnetNetwork <: AbstractDynamics
    gspec
    hyper
    common
    statehead
    rewardhead
end

# Dynamics Network with identity as state head
function DynamicsNetwork(gspec::GI.AbstractGameSpec, hyper::DynamicsResnetHP)
    state_dim = GI.state_dim(gspec)
    # indim = (state_dim[1], state_dim[2], hyper.num_filters+1)
    indim = (state_dim[1], state_dim[2], state_dim[3] + 1)
    resnethyper = ResNetHP_(
        indim, # indim not used
        1, # outdim_head1 not used
        hyper.num_blocks,
        hyper.num_filters,
        hyper.conv_kernel_size,
        1, # num_head1_filters not used
        hyper.num_reward_head2_filters,
        hyper.batch_norm_momentum,
    )
    resnet = ResNet_(gspec, resnethyper)
    small_state = Conv((1, 1), hyper.num_filters => state_dim[3], relu) # a lot quicker hash
    return DynamicsResnetNetwork(
        gspec,
        hyper,
        resnet.common,
        small_state, # state head as identity
        resnet.head2,
    )
end

@kwdef struct PredictionResnetHP
    num_blocks::Int
    num_filters::Int
    conv_kernel_size::Tuple{Int,Int}
    num_policy_head1_filters::Int = 2 # head1
    num_value_head2_filters::Int = 1 # head2
    batch_norm_momentum::Float32 = 0.6f0
end

mutable struct PredictionResnetNetwork <: AbstractPrediction
    gspec
    hyper
    common
    policyhead
    valuehead
end

function PredictionNetwork(gspec::GI.AbstractGameSpec, hyper::PredictionResnetHP)
    indim = GI.state_dim(gspec)
    outdim_head1 = GI.num_actions(gspec)
    resnethyper = ResNetHP_(
        indim, # indim not used
        outdim_head1,
        hyper.num_blocks,
        hyper.num_filters,
        hyper.conv_kernel_size,
        hyper.num_policy_head1_filters,
        hyper.num_value_head2_filters,
        hyper.batch_norm_momentum,
    )
    resnet = ResNet_(gspec, resnethyper)
    return PredictionResnetNetwork(
        gspec,
        hyper,
        resnet.common, # leaving only residual layers
        resnet.head1,
        resnet.head2,
    )
end

@kwdef struct RepresentationSimpleHP
    width::Int
    depth::Int
    hiddenstate_shape::Int
    use_batch_norm::Bool = false
    batch_norm_momentum::Float32 = 0.6f0
end

mutable struct RepresentationSimpleNetwork <: AbstractRepresentation
    gspec
    hyper
    common
end

function RepresentationNetwork(gspec::GI.AbstractGameSpec, hyper::RepresentationSimpleHP)
    bnmom = hyper.batch_norm_momentum
    function make_dense(indim, outdim)
        if hyper.use_batch_norm
            Chain(Dense(indim, outdim; bias=false), BatchNorm(outdim, relu; momentum=bnmom))
        else
            Dense(indim, outdim, relu)
        end
    end
    indim = prod(GI.state_dim(gspec))
    outdim = hyper.hiddenstate_shape
    hsize = hyper.width
    hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth]
    if hyper.depth == -1 # somewhat unintuitive, jump from 1 to 3 layers #? depth-1 
        common = Chain(
            flatten,
            # make_dense(indim, outdim)
            Dense(indim, outdim),
        )
    else
        common = Chain(
            flatten,
            make_dense(indim, hsize),
            hlayers(hyper.depth)...,
            make_dense(hsize, outdim),
        )
    end
    return RepresentationSimpleNetwork(gspec, hyper, common)
end

### general SimpleNet used in Prediction and Dynamics
# TODO make constructor from vector of sizes 
@kwdef struct SimpleNetHP_
    indim::Int
    outdim::Int
    width::Int
    depth_common::Int
    depth_vectorhead::Int = 1
    depth_scalarhead::Int = 1
    use_batch_norm::Bool = false
    batch_norm_momentum::Float32 = 0.6f0
end

mutable struct SimpleNet_
    gspec
    hyper
    common
    vectorhead
    scalarhead
end
# TODO remove gspec dependence
function SimpleNet_(gspec::GI.AbstractGameSpec, hyper::SimpleNetHP_)
    bnmom = hyper.batch_norm_momentum
    function make_dense(indim, outdim)
        if hyper.use_batch_norm
            Chain(Dense(indim, outdim; bias=false), BatchNorm(outdim, elu; momentum=bnmom))
        else
            Dense(indim, outdim, elu)
        end
    end
    indim = hyper.indim
    outdim = hyper.outdim
    hsize = hyper.width
    hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth] #? 1:depth-1
    # common = depth_common == -1 ?
    #   flatten :
    #   Chain(
    #     flatten, #?
    #     make_dense(indim, hsize),
    #     hlayers(hyper.depth_common)...)
    # scalarhead = Chain(
    #   hlayers(hyper.depth_scalarhead)...,
    #   Dense(hsize, 1, tanh))
    # vectorhead = Chain(
    #   hlayers(hyper.depth_vectorhead)...,
    #   Dense(hsize, outdim),
    #   softmax)
    if hyper.depth_common == -1
        common = identity
        outcomm = indim
    else
        common = Chain(
            flatten, #? identity
            make_dense(indim, hsize),
            hlayers(hyper.depth_common)...,
        )
        outcomm = hsize
    end
    if hyper.depth_scalarhead == -1
        scalarhead = Dense(outcomm, 1, tanh)
    else
        scalarhead = Chain(
            outcomm != hsize ? make_dense(outcomm, hsize) : identity,
            hlayers(hyper.depth_scalarhead)...,
            Dense(hsize, 1, tanh),
        )
    end
    if hyper.depth_vectorhead == -1
        vectorhead = Chain(Dense(outcomm, outdim))
    else
        vectorhead = Chain(
            outcomm != hsize ? make_dense(outcomm, hsize) : identity,
            hlayers(hyper.depth_vectorhead)...,
            Dense(hsize, outdim),
        )
    end

    return SimpleNet_(gspec, hyper, common, vectorhead, scalarhead)
end

### Dynamics ###

@kwdef struct DynamicsSimpleHP
    hiddenstate_shape::Int
    width::Int
    depth_common::Int
    depth_vectorhead::Int = 1 # depth state-head
    depth_scalarhead::Int = 1 # depth reward-head
    use_batch_norm::Bool = false
    batch_norm_momentum::Float32 = 0.6f0
end

mutable struct DynamicsSimpleNetwork <: AbstractDynamics
    gspec
    hyper
    common
    statehead
    rewardhead
end

function DynamicsNetwork(gspec::GI.AbstractGameSpec, hyper::DynamicsSimpleHP)
    indim = hyper.hiddenstate_shape + GI.num_actions(gspec)
    outdim = hyper.hiddenstate_shape
    simplenethyper = SimpleNetHP_(
        indim,
        outdim,
        hyper.width,
        hyper.depth_common,
        hyper.depth_vectorhead,
        hyper.depth_scalarhead,
        hyper.use_batch_norm,
        hyper.batch_norm_momentum,
    )
    simplenet = SimpleNet_(gspec, simplenethyper)
    return DynamicsSimpleNetwork(
        gspec, hyper, simplenet.common, simplenet.vectorhead, simplenet.scalarhead
    )
end

### Prediction ###

@kwdef struct PredictionSimpleHP
    hiddenstate_shape::Int
    width::Int
    depth_common::Int
    depth_vectorhead::Int = 1 # depth policy-head
    depth_scalarhead::Int = 1 # depth value-head
    use_batch_norm::Bool = false
    batch_norm_momentum::Float32 = 0.6f0
end

mutable struct PredictionSimpleNetwork <: AbstractPrediction
    gspec
    hyper
    common
    policyhead
    valuehead
end

function PredictionNetwork(gspec::GI.AbstractGameSpec, hyper::PredictionSimpleHP)
    indim = hyper.hiddenstate_shape
    outdim = GI.num_actions(gspec)
    simplenethyper = SimpleNetHP_(
        indim,
        outdim,
        hyper.width,
        hyper.depth_common,
        hyper.depth_vectorhead,
        hyper.depth_scalarhead,
        hyper.use_batch_norm,
        hyper.batch_norm_momentum,
    )
    simplenet = SimpleNet_(gspec, simplenethyper)
    return PredictionSimpleNetwork(
        gspec,
        hyper,
        simplenet.common,
        Chain(simplenet.vectorhead, softmax),
        simplenet.scalarhead,
    )
end

# function evaluate_batch(nn::AbstractPrediction, batch) # is the same as resnet

#TODO this it kinda ugly, maybe create macro that create proper on_gpu based on networks
function on_gpu(nn::Union{PredictionSimpleNetwork,PredictionResnetNetwork})
    return array_on_gpu(nn.valuehead[end].bias)
end
function on_gpu(nn::Union{DynamicsSimpleNetwork,DynamicsResnetNetwork})
    return array_on_gpu(nn.rewardhead[end].bias)
end
on_gpu(nn::RepresentationSimpleNetwork) = array_on_gpu(nn.common[end].bias)
on_gpu(nn::RepresentationResnetNetwork) = array_on_gpu(nn.common[end][1].layers[1].weight)

to_nndevice(nn, x) = on_gpu(nn) ? Flux.gpu(x) : x
from_nndevice(nn, x) = Flux.cpu(x)

# remembert to put fields in correct order, otherwise they are swapped when |>gpu
Flux.@functor RepresentationSimpleNetwork (common,)
Flux.@functor RepresentationResnetNetwork (common,)
Flux.@functor DynamicsSimpleNetwork (common, statehead, rewardhead)
Flux.@functor DynamicsResnetNetwork (common, statehead, rewardhead)
Flux.@functor PredictionSimpleNetwork (common, policyhead, valuehead)
Flux.@functor PredictionResnetNetwork (common, policyhead, valuehead)
Flux.@functor MuNetwork (f, g, h)

struct GameSpec <: GI.AbstractGameSpec end
GI.actions(::GameSpec) = collect(1:9)

function GI.vectorize_state(::GameSpec, state)
    board = state.curplayer == WHITE ? state.board : flip_colors(state.board)
    return Float32[
        board[pos_of_xy((x, y))] == c for x in 1:BOARD_SIDE, y in 1:BOARD_SIDE,
        c in [nothing, WHITE, BLACK]
    ]
end

function MuZeroTrainableEnvOracle()
    gspec = GameSpec()
    simμNetworkHP = hs_shape = 32
    simμNetworkHP = MuNetworkHP(
        gspec,
        PredictionSimpleHP(;
            hiddenstate_shape=hs_shape,
            width=64,
            depth_common=-1,
            depth_vectorhead=0,
            depth_scalarhead=0,
            use_batch_norm=false,
            batch_norm_momentum=0.8f0,
        ),
        DynamicsSimpleHP(;
            hiddenstate_shape=hs_shape,
            width=64,
            depth_common=-1,
            depth_vectorhead=0,
            depth_scalarhead=0,
            use_batch_norm=false,
            batch_norm_momentum=0.8f0,
        ),
        RepresentationSimpleHP(; hiddenstate_shape=hs_shape, width=0, depth=-1),
    )
    neural_networks = MuNetwork(simμNetworkHP)

    return MuZeroTrainableEnvOracle(neural_networks)
end

"""
    fill_episode_in_batch(sample, num_unroll_steps)

Fill `actions` and `targets` of a `sample` so that they have the expected length:
`num_unroll_steps`.

Necessary to have a consistent size for `Flux.batch`.
"""
function fill_actions_in_batch(sample, num_unroll_steps)
    (state, actions, targets) = sample

    extra_length = num_unroll_steps - length(actions)
    extra_actions = fill(1, extra_length) # XXX: fill with an arbitrary actions
    filled_actions = cat(actions, extra_actions; dims=1)

    return (state, filled_actions, targets)
end

function onehot(action, num_actions; type=Float32)
    result = zeros(type, num_actions)
    result[action] = one(type)
    return result
end

function encode_action_in_batch(sample, num_actions)
    (state, actions, targets) = sample
    encoded_actions = onehot.(actions, [num_actions])
    return state, encoded_actions, targets
end

function fluxify_batch(batch, num_unroll_steps, num_actions)
    batch = fill_actions_in_batch.(batch, [num_unroll_steps])
    batch = encode_action_in_batch.(batch, [num_actions])

    X = Flux.batch(sample[1] for sample in batch)
    A_mask = Flux.batch(fill(true, 9) for sample in batch) # XXX: issue ?
    As = Flux.batch(hcat((sample[2] for sample in batch)...))
    Ps = Flux.batch([hcat([target[3] for target in sample[3]]...) for sample in batch])
    Vs = Flux.batch([[target[1] for target in sample[3]] for sample in batch])
    Rs = Flux.batch([[target[2] for target in sample[3]] for sample in batch])

    f32(arr) = convert(AbstractArray{Float32}, arr)
    return map(f32, (; X, A_mask, As, Ps, Vs, Rs))
end

function TrainableEnvOracles.update_weights(
    trainable_oracle::MuZeroTrainableEnvOracle, batches, train_settings
)
    hyper = (;
        num_unroll_steps=5, #if =0, g is not learning, muzero-general=20
        td_steps=20, # with max length=9, always go till the end of the game, rootvalues don't count
        discount=0.997,
        #// value_loss_weight = 0.25, #TODO
        l2_regularization=1.0f-4, #Float32
        #// l2_regularization=0f0, #Float32
        loss_computation_batch_size=512,
        batches_per_checkpoint=10,
        num_checkpoints=1,
        opt=Scheduler(
            Cos(; λ0=3e-3, λ1=1e-5, period=10000), # cosine annealing, google period 2e4, generat doesn't use any
            Flux.ADAM(),
        ),
        model_type=:mlp, # :mlp || :resnet
        device=Flux.cpu, # Flux.cpu || Flux.gpu
    )
    NOT_USED = 0

    mu_trainer = MuTrainer(
        NOT_USED, trainable_oracle.neural_networks, NOT_USED, hyper, hyper.opt
    )
    samples = (
        mu_trainer.hyper.device(fluxify_batch(batch, hyper.num_unroll_steps, 9)) for
        batch in batches
    )
    update_weights!(mu_trainer, samples)
    return nothing
end

using ..BatchedMcts
function TrainableEnvOracles.get_env_oracle(trainable_oracle::MuZeroTrainableEnvOracle)
    A = 9 # Number of case in the Tic-Tac-Toe grid
    nns = Flux.testmode!(Flux.cpu(deepcopy(trainable_oracle.neural_networks)))

    init_oracle = InitialOracle(nns)
    recurrent_oracle = RecurrentOracle(nns)

    get_fake_valid_actions(A, B, device) = fill(true, device, (A, B))
    get_fake_player_switch(B, device) = fill(true, device, B)
    get_fake_terminated(B, device) = fill(false, device, B)

    function init_fn(envs)
        B = length(envs)
        device = get_device(envs)

        @assert B > 0
        states = vectorize_state.(envs)
        policy_prior, value_prior, internal_states = evaluate_batch(init_oracle, states)
        internal_states = hcat(internal_states...)
        policy_prior = hcat(policy_prior...)

        return (;
            internal_states,
            valid_actions=get_fake_valid_actions(A, B, device),
            policy_prior,
            value_prior,
        )
    end

    function transition_fn(states, aids)
        B = size(states)[end]
        device = get_device(states)

        policy_prior, value_prior, internal_states, rewards = evaluate_batch(
            recurrent_oracle, states, aids
        )
        internal_states = hcat(internal_states...)
        policy_prior = hcat(policy_prior...)

        return (;
            internal_states,
            rewards,
            terminal=get_fake_terminated(B, device),
            valid_actions=get_fake_valid_actions(A, B, device),
            player_switched=get_fake_player_switch(B, device),
            policy_prior,
            value_prior,
        )
    end
    return BatchedMcts.EnvOracle(; init_fn, transition_fn)
end

lossₚ(p̂, p)::Float32 = Flux.Losses.crossentropy(p̂, p) #TODO move to hyper
lossᵥ(v̂, v)::Float32 = Flux.Losses.mse(v̂, v)
# lossᵣ(r̂, r)::Float32 = 0f0
lossᵣ(r̂, r)::Float32 = Flux.Losses.mse(r̂, r)

# lossᵥ(v̂, v)::Float32 = Flux.Losses.crossentropy(v̂, v)
# lossᵣ(r̂, r)::Float32 = Flux.Losses.crossentropy(r̂, r)

function normalize_p(P, actions_mask)
    P = P .* actions_mask # Zygote doesn't work with .*= 
    sp = sum(P; dims=1)
    P = P ./ (sp .+ eps(eltype(P)))
    return P
end

# TODO add assertions about sizes
function losses(nns, hyper, (X, A_mask, As, Ps, Vs, Rs))
    prediction, dynamics, representation = nns.f, nns.g, nns.h
    creg::Float32 = hyper.l2_regularization
    Ksteps = hyper.num_unroll_steps
    dimₐ = hyper.model_type == :mlp ? 2 : 3

    # initial step, from the real observation
    Hiddenstate = forward(representation, X)
    P̂⁰, V̂⁰ = forward(prediction, Hiddenstate)
    P̂⁰ = normalize_p(P̂⁰, A_mask)
    # R̂⁰ = zero(V̂⁰)
    # batchdim = ndims(Hiddenstate)

    scale_initial = iszero(Ksteps) ? 1.0f0 : 0.5f0
    Lp = scale_initial * lossₚ(P̂⁰, Ps[:, 1, :]) # scale=1
    Lv = scale_initial * lossᵥ(V̂⁰, Vs[1:1, :])
    Lr = zero(Lv) # starts at next step (see MuZero paper appendix)

    scale_recurrent = iszero(Ksteps) ? nothing : 0.5f0 / Ksteps #? instead of constant scale, maybe 2^(-i+1)
    # recurrent inference 
    for k in 1:Ksteps
        # targets are stored as follows: [A⁰¹ A¹² ...] [P⁰ P¹ ...] [V⁰ V¹ ...] but [R¹ R² ...]
        # A = As[k, :]
        # A = As[:,:,k:k,:]
        A = selectdim(As, dimₐ, k)
        S_A = cat(Hiddenstate, A; dims=ndims(Hiddenstate) - 1)
        # R̂, Hiddenstate = forward(dynamics, Hiddenstate, A) # obtain next hiddenstate
        R̂, Hiddenstate = forward(dynamics, S_A) # obtain next hiddenstate
        P̂, V̂ = forward(prediction, Hiddenstate) #? should flip V based on players
        # scale loss so that the overall weighting of the recurrent_inference (g,f nns)
        # is equal to that of the initial_inference (h,f nns)
        Lp += scale_recurrent * lossₚ(P̂, Ps[:, k + 1, :]) #? @view
        Lv += scale_recurrent * lossᵥ(V̂, Vs[(k + 1):(k + 1), :])
        Lr += scale_recurrent * lossᵣ(R̂, Rs[k:k, :])
    end
    Lreg =
        iszero(creg) ? zero(Lv) : creg * sum(sum(w .^ 2) for w in regularized_params(nns))
    L = Lp + Lv + Lr + Lreg # + Lr
    # L = Lp + Lreg # + Lr
    # Zygote.@ignore @info "Loss" loss_total=L loss_policy=Lp loss_value=Lv loss_reward=Lr loss_reg_params=Lreg relative_entropy=Lp-Flux.Losses.crossentropy(Ps, Ps) #? check if compute means inside logger is avaliable
    return (L, Lp, Lv, Lr, Lreg)
end

# #TODO replace Zygote.withgradient() - new version
# function lossgrads(f, args...)
#   val, back = Zygote.pullback(f, args...)
#   grad = back(Zygote.sensitivity(val))
#   return val, grad
# end

# function train!(nns, opt, loss, data; cb=()->())
function μtrain!(nns, loss, data, opt)
    ps = Flux.params(nns)
    losses = Float32[]
    for (i, d) in enumerate(data)
        l, gs = Zygote.withgradient(ps) do
            loss(d...)
        end
        push!(losses, l)
        Flux.update!(opt, ps, gs)
        # @info "debug" η=opt.optim.eta
    end
    @show losses
    return mean_loss_total = mean(losses)
end

struct MuTrainer
    gspec
    nns # MuNetwork
    memory
    hyper
    opt
    function MuTrainer(gspec, nns, memory, hyper, opt)
        return new(gspec, Flux.trainmode!(hyper.device(nns)), memory, hyper, opt)
    end
end

function update_weights!(tr::MuTrainer, samples)
    L(batch...) = losses(tr.nns, tr.hyper, batch)[1]

    return μtrain!(tr.nns, L, samples, tr.opt)
end

end
