#####
##### AlphaZero Parameters
#####

"""
Parameters of an MCTS player.

| Parameter              | Type                    | Default             |
|:-----------------------|:------------------------|:--------------------|
| `num_workers`          | `Int`                   | `1`                 |
| `use_gpu`              | `Bool`                  | `false`             |
| `num_iters_per_turn`   | `Int`                   |  -                  |
| `cpuct`                | `Float64`               | `1.`                |
| `temperature`          | `StepSchedule{Float64}` | `StepSchedule(1.)`  |
| `dirichlet_noise_ϵ`    | `Float64`               |  -                  |
| `dirichlet_noise_α`    | `Float64`               |  -                  |

# Explanation

An MCTS player picks actions as follows. Given a game state, it launches
`num_iters_per_turn` MCTS iterations that are executed asynchronously on
`num_workers` workers, with UCT exploration constant `cpuct`.

Then, an action is picked according to the distribution ``π`` where
``π_i ∝ n_i^τ`` with ``n_i`` the number of time that the ``i^{\\text{th}}``
action was visited and ``τ`` the `temperature` parameter.

It is typical to use a high value of the temperature parameter ``τ``
during the first moves of a game to increase exploration and then switch to
a small value. Therefore, `temperature` has type [`StepSchedule`](@ref).

For information on parameters `cpuct`, `dirichlet_noise_ϵ` and
`dirichlet_noise_α`, see [`MCTS.Env`](@ref).

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper:

+ The number of MCTS iterations per move is 1600, which
  corresponds to 0.4s of computation time.
+ The temperature is set to 1 for the 30 first moves and then to an
  infinitesimal value.
+ The ``ϵ`` parameter for the Dirichlet noise is set to ``0.25`` and
  the ``α`` parameter to ``0.03``, which is consistent with the heuristic
  of using ``α = 10/n`` with ``n`` the maximum number of possibles moves,
  which is ``19 × 19 + 1 = 362`` in the case of Go.
"""
@kwdef struct MctsParams
  num_workers :: Int = 1
  use_gpu :: Bool = false
  cpuct :: Float64 = 1.
  num_iters_per_turn :: Int
  temperature :: StepSchedule{Float64} = StepSchedule(1.)
  dirichlet_noise_ϵ :: Float64
  dirichlet_noise_α :: Float64
end

"""
    ArenaParams

Parameters that govern the evaluation process that compares
a new neural network to the current best.

| Parameter            | Type                  | Default        |
|:---------------------|:----------------------|:---------------|
| `mcts`               | [`MctsParams`](@ref)  |  -             |
| `num_games`          | `Int`                 |  -             |
| `reset_mcts_every`   | `Union{Int, Nothing}` | `nothing`      |
| `update_threshold`   | `Float64`             |  -             |

# Explanation

+ The two competing networks are instantiated into two MCTS players
  of parameter `mcts` and then play `num_games` games, exchanging color
  after each game.
+ The new network is to replace the current best one if its
  average collected reward is greater or equal than `update_threshold`.
+ To avoid running out of memory, the MCTS trees of both player are
  reset every `reset_mcts_every` games (or never if `nothing` is passed).

# Remarks

+ See [`necessary_samples`](@ref) to make an informed choice for `num_games`.

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper, 400 games are played to evaluate a network
and the `update_threshold` parameter is set to a value that corresponds to a
55% win rate.
"""
@kwdef struct ArenaParams
  num_games :: Int
  reset_mcts_every :: Union{Nothing, Int} = nothing
  mcts :: MctsParams
  update_threshold :: Float64
end

"""
    SelfPlayParams

Parameters governing self-play.

| Parameter            | Type                  | Default        |
|:---------------------|:----------------------|:---------------|
| `mcts`               | [`MctsParams`](@ref)  |  -             |
| `num_games`          | `Int`                 |  -             |
| `reset_mcts_every`   | `Union{Int, Nothing}` | `nothing`      |
| `gc_every`           | `Union{Int, Nothing}` | `nothing`      |

# Explanation

+ The `gc_every` field, when set, forces a full garbage collection
  and an emptying of the GPU memory pool periodically, the period being
  specified in terms of a fixed number of games.
+ To avoid running out of memory, the MCTS tree is
  reset every `reset_mcts_every` games (or never if `nothing` is passed).

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper, `num_games = 25_000` (5 millions games
of self-play across 200 iterations).
"""
@kwdef struct SelfPlayParams
  num_games :: Int
  reset_mcts_every :: Union{Nothing, Int} = nothing
  gc_every :: Union{Nothing, Int} = nothing
  mcts :: MctsParams
end

"""
    SamplesWeighingPolicy

During self-play, early board positions are possibly encountered many
times across several games. The corresponding samples are then merged
together and given a weight ``W`` that is a nondecreasing function of the
number ``n`` of merged samples:

  - `CONSTANT_WEIGHT`: ``W(n) = 1``
  - `LOG_WEIGHT`: ``W(n) = \\log_2(n) + 1``
  - `LINEAR_WEIGHT`: ``W(n) = n``
"""
@enum SamplesWeighingPolicy CONSTANT_WEIGHT LOG_WEIGHT LINEAR_WEIGHT

"""
    LearningParams

Parameters governing the learning phase of a training iteration, where
the neural network is updated to fit the data in the memory buffer.

| Parameter                     | Type                            | Default    |
|:------------------------------|:--------------------------------|:-----------|
| `use_gpu`                     | `Bool`                          | `true`     |
| `gc_every`                    | `Union{Nothing, Int}`           | `nothing`  |
| `samples_weighing_policy`     | [`SamplesWeighingPolicy`](@ref) |  -         |
| `optimiser`                   | [`OptimiserSpec`](@ref)         |  -         |
| `l2_regularization`           | `Float32`                       |  -         |
| `nonvalidity_penalty`         | `Float32`                       | `1f0`      |
| `batch_size`                  | `Int`                           |  -         |
| `loss_computation_batch_size` | `Int`                           |  -         |
| `checkpoints`                 | `Vector{Int}`                   |  -         |

# Description

The neural network gets to see the whole content of the memory buffer at each
learning epoch, for `maximum(checkpoints)` epochs in total. After each epoch
whose number is in `checkpoints`, the current network is evaluated against
the best network so far (see [`ArenaParams`](@ref)).

+ `nonvalidity_penalty` is the multiplicative constant of a loss term that
   corresponds to the average probability weight that the network puts on
   invalid actions.
+ `batch_size` is the batch size used for gradient descent.
+ `loss_computation_batch_size` is the batch size that is used to compute
  the loss between each epochs.

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper:
+ The batch size for gradient updates is ``2048``.
+ The L2 regularization parameter is set to ``10^{-4}``.
+ Checkpoints are produced every 1000 training steps, which corresponds
  to seeing about 20% of the samples in the memory buffer:
  ``(1000 × 2048) / 10^7  ≈ 0.2``.
+ It is unclear how many checkpoints are taken or how many training steps
  are performed in total.
"""
@kwdef struct LearningParams
  use_gpu :: Bool = true
  gc_every :: Union{Nothing, Int} = nothing
  samples_weighing_policy :: SamplesWeighingPolicy
  optimiser :: OptimiserSpec
  l2_regularization :: Float32
  nonvalidity_penalty :: Float32 = 1f0
  batch_size :: Int
  loss_computation_batch_size :: Int
  checkpoints :: Vector{Int}
end

"""
    MemAnalysisParams

Parameters governing the analysis of the memory buffer
(for debugging and profiling purposes).

| Parameter           | Type           | Default   |
|:--------------------|:---------------|:----------|
| `num_game_stages`   | `Int`          |  -        |

# Explanation

The memory analysis consists in partitioning the memory buffer in
`num_game_stages` parts of equal size, according to the number of
remaining moves until the end of the game for each sample. Then,
the quality of the predictions of the current neural network is
evaluated on each subset (see [`Report.Memory`](@ref)).

This is useful to get an idea of how the neural network performance
varies depending on the game stage (typically, good value estimates for
endgame board positions are available earlier in the training process
than good values for middlegame positions).
"""
@kwdef struct MemAnalysisParams
  num_game_stages :: Int
end

"""
    Params

The AlphaZero parameters.

| Parameter                  | Type                                | Default   |
|:---------------------------|:------------------------------------|:----------|
| `self_play`                | [`SelfPlayParams`](@ref)            |  -        |
| `learning`                 | [`LearningParams`](@ref)            |  -        |
| `arena`                    | [`ArenaParams`](@ref)               |  -        |
| `memory_analysis`          | `Union{Nothing, MemAnalysisParams}` | `nothing` |
| `num_iters`                | `Int`                               |  -        |
| `mem_buffer_size`          | `PLSchedule{Int}`                   |  -        |
| `ternary_rewards`          | `Bool`                              | `false`   |

# Explanation

The AlphaZero training process consists in `num_iters` iterations. Each
iteration can be decomposed into a self-play phase
(see [`SelfPlayParams`](@ref)) and a learning phase
(see [`LearningParams`](@ref)).

  - `ternary_rewards`: set to `true` if the rewards issued by
    the game environment always belong to ``\\{-1, 0, 1\\}`` so that
    the logging and profiling tools can take advantage of this property.
  - `mem_buffer_size`: size schedule of the memory buffer, in terms of number
    of samples. It is typical to start with a small memory buffer that is grown
    progressively so as to wash out the initial low-quality self-play data
    more quickly.

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper:
+ About 5 millions games of self-play are played across 200 iterations.
+ The memory buffer contains 500K games, which makes about 100M samples
  as an average game of Go lasts about 200 turns.
"""
@kwdef struct Params
  self_play :: SelfPlayParams
  memory_analysis :: Union{Nothing, MemAnalysisParams} = nothing
  learning :: LearningParams
  arena :: ArenaParams
  num_iters :: Int
  mem_buffer_size :: PLSchedule{Int}
  ternary_rewards :: Bool = false
end

for T in [MctsParams, ArenaParams, SelfPlayParams, LearningParams, Params]
  Util.generate_update_constructor(T) |> eval
end

#####
##### Utilities
#####

"""
    necessary_samples(ϵ, β) = log(1 / β) / (2 * ϵ^2)

Compute the number of times ``N`` that a random variable
``X \\sim \\text{Ber}(p)`` has to be sampled so that if the
empirical average of ``X`` is greather than
``1/2 + ϵ``, then ``p > 1/2`` with probability at least ``1-β``.

This bound is based on [Hoeffding's inequality
](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality).
"""
necessary_samples(ϵ, β) = log(1 / β) / (2 * ϵ^2)
