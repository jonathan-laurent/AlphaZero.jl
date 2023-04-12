#####
##### AlphaZero Parameters
#####

"""
Parameters of an MCTS player.

| Parameter              | Type                         | Default             |
|:-----------------------|:-----------------------------|:--------------------|
| `num_iters_per_turn`   | `Int`                        |  -                  |
| `gamma`                | `Float64`                    | `1.`                |
| `cpuct`                | `Float64`                    | `1.`                |
| `temperature`          | `AbstractSchedule{Float64}`  | `ConstSchedule(1.)` |
| `dirichlet_noise_ϵ`    | `Float64`                    |  -                  |
| `dirichlet_noise_α`    | `Float64`                    |  -                  |
| `prior_temperature`    | `Float64`                    | `1.`                |

# Explanation

An MCTS player picks an action as follows. Given a game state, it launches
`num_iters_per_turn` MCTS iterations, with UCT exploration constant `cpuct`.
Rewards are discounted using the `gamma` factor.

Then, an action is picked according to the distribution ``π`` where
``π_i ∝ n_i^{1/τ}`` with ``n_i`` the number of times that the ``i^{\\text{th}}``
action was visited and ``τ`` the `temperature` parameter.

It is typical to use a high value of the temperature parameter ``τ``
during the first moves of a game to increase exploration and then switch to
a small value. Therefore, `temperature` is am [`AbstractSchedule`](@ref).

For information on parameters `cpuct`, `dirichlet_noise_ϵ`,
`dirichlet_noise_α` and `prior_temperature`, see [`MCTS.Env`](@ref).

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper:

+ The discount factor `gamma` is set to 1.
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
  gamma :: Float64 = 1.
  cpuct :: Float64 = 1.
  num_iters_per_turn :: Int
  temperature :: AbstractSchedule{Float64} = ConstSchedule(1.)
  dirichlet_noise_ϵ :: Float64
  dirichlet_noise_α :: Float64
  prior_temperature :: Float64 = 1.
end

"""
    SimParams

Parameters for parallel game simulations.

These parameters are common to self-play data generation, neural network evaluation
and benchmarking.

| Parameter            | Type                  | Default        |
|:---------------------|:----------------------|:---------------|
| `num_games`          | `Int`                 |  -             |
| `num_workers`        | `Int`                 |  -             |
| `batch_size `        | `Int`                 |  -             |
| `use_gpu`            | `Bool`                | `false`        |
| `fill_batches`       | `Bool`                | `true`         |
| `flip_probability`   | `Float64`             | `0.`           |
| `reset_every`        | `Union{Nothing, Int}` | `1`            |
| `alternate_colors`   | `Float64`             | `false`        |

## Explanations

  + On each machine (process), `num_workers` simulation tasks are spawned. Inference
    requests are processed by an inference server by batch of size `batch_size`. Note that
    we must have `batch_size <= num_workers`.
  + If `fill_batches` is set to `true`, we make sure that batches sent to the
    neural network for inference have constant size.
  + Both players are reset (e.g. their MCTS trees are emptied)
    every `reset_every` games (or never if `nothing` is passed).
  + To add randomization and before every game turn, the game board is "flipped"
    according to a symmetric transformation with probability `flip_probability`.
  + In the case of (symmetric) two-player games and if `alternate_colors` is set to`true`,
    then the colors of both players are swapped between each simulated game.
"""
@kwdef struct SimParams
  num_games :: Int
  num_workers :: Int
  batch_size :: Int
  use_gpu :: Bool = false
  fill_batches :: Bool = true
  reset_every :: Union{Nothing, Int} = 1
  flip_probability :: Float64 = 0.
  alternate_colors :: Bool = false
end

"""
    ArenaParams

Parameters that govern the evaluation process that compares
the current neural network with the best one seen so far
(which is used to generate data).

| Parameter            | Type                  | Default        |
|:---------------------|:----------------------|:---------------|
| `mcts`               | [`MctsParams`](@ref)  |  -             |
| `sim`                | [`SimParams`](@ref)   |  -             |
| `update_threshold`   | `Float64`             |  -             |

# Explanation (two-player games)

+ The two competing networks are instantiated into two MCTS players
  of parameter `mcts` and then play `sim.num_games` games.
+ The evaluated network replaces the current best one if its
  average collected reward is greater or equal than `update_threshold`.

# Explanation (single-player games)

+ The two competing networks play `sim.num_games` games each.
+ The evaluated network replaces the current best one if its average collected rewards
  exceeds the average collected reward of the old one by `update_threshold` at least. 

# Remarks

+ See [`necessary_samples`](@ref) to make an informed choice for `sim.num_games`.

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper, 400 games are played to evaluate a network
and the `update_threshold` parameter is set to a value that corresponds to a
55% win rate.
"""
@kwdef struct ArenaParams
  mcts :: MctsParams
  sim :: SimParams
  update_threshold :: Float64
end

"""
    SelfPlayParams

Parameters governing self-play.

| Parameter            | Type                  | Default        |
|:---------------------|:----------------------|:---------------|
| `mcts`               | [`MctsParams`](@ref)  |  -             |
| `sim`                | [`SimParams`](@ref)   |  -             |

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper, `sim.num_games=25_000` (5 millions games
of self-play across 200 iterations).
"""
@kwdef struct SelfPlayParams
  mcts :: MctsParams
  sim :: SimParams
end

"""
    SamplesWeighingPolicy

During self-play, early board positions are possibly encountered many
times across several games. The corresponding samples can be merged
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
| `use_gpu`                     | `Bool`                          | `false`    |
| `use_position_averaging`      | `Bool`                          | `true`     |
| `samples_weighing_policy`     | [`SamplesWeighingPolicy`](@ref) |  -         |
| `optimiser`                   | [`OptimiserSpec`](@ref)         |  -         |
| `l2_regularization`           | `Float32`                       |  -         |
| `rewards_renormalization`     | `Float32`                       | `1f0`      |
| `nonvalidity_penalty`         | `Float32`                       | `1f0`      |
| `batch_size`                  | `Int`                           |  -         |
| `loss_computation_batch_size` | `Int`                           |  -         |
| `min_checkpoints_per_epoch`   | `Float64`                       |  -         |
| `max_batches_per_checkpoint`  | `Int`                           |  -         |
| `num_checkpoints`             | `Int`                           |  -         |

# Description

The neural network goes through `num_checkpoints` series of `n` updates using
batches of size `batch_size` drawn from memory, where `n` is defined as follows:

```
n = min(max_batches_per_checkpoint, ntotal ÷ min_checkpoints_per_epoch)
```

with `ntotal` the total number of batches in memory. Between each series,
the current network is evaluated against the best network so far
(see [`ArenaParams`](@ref)).

+ `nonvalidity_penalty` is the multiplicative constant of a loss term that
   corresponds to the average probability weight that the network puts on
   invalid actions.
+ `batch_size` is the batch size used for gradient descent.
+ `loss_computation_batch_size` is the batch size that is used to compute
  the loss between each epochs.
+ All rewards are divided by `rewards_renormalization` before the MSE loss is computed.
+ If `use_position_averaging` is set to true, samples in memory that correspond
  to the same board position are averaged together. The merged sample is
  reweighted according to `samples_weighing_policy`.

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
  use_gpu :: Bool = false
  use_position_averaging :: Bool = true
  samples_weighing_policy :: SamplesWeighingPolicy
  optimiser :: OptimiserSpec
  l2_regularization :: Float32
  rewards_renormalization :: Float32 = 1f0
  nonvalidity_penalty :: Float32 = 1f0
  batch_size :: Int
  loss_computation_batch_size :: Int
  min_checkpoints_per_epoch :: Int
  max_batches_per_checkpoint :: Int
  num_checkpoints :: Int
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

The AlphaZero training hyperparameters.

| Parameter                  | Type                                | Default   |
|:---------------------------|:------------------------------------|:----------|
| `self_play`                | [`SelfPlayParams`](@ref)            |  -        |
| `learning`                 | [`LearningParams`](@ref)            |  -        |
| `arena`                    | `Union{Nothing, ArenaParams}`       |  -        |
| `memory_analysis`          | `Union{Nothing, MemAnalysisParams}` | `nothing` |
| `num_iters`                | `Int`                               |  -        |
| `use_symmetries`           | `Bool`                              | `false`   |
| `ternary_rewards`          | `Bool`                              | `false`   |
| `mem_buffer_size`          | `PLSchedule{Int}`                   |  -        |

# Explanation

The AlphaZero training process consists in `num_iters` iterations. Each
iteration can be decomposed into a self-play phase
(see [`SelfPlayParams`](@ref)) and a learning phase
(see [`LearningParams`](@ref)).

  - `ternary_rewards`: set to `true` if the rewards issued by
     the game environment always belong to ``\\{-1, 0, 1\\}`` so that
     the logging and profiling tools can take advantage of this property.
  - `use_symmetries`: if set to `true`, board symmetries are used for
     data augmentation before learning.
  - `mem_buffer_size`: size schedule of the memory buffer, in terms of number
     of samples. It is typical to start with a small memory buffer that is grown
     progressively so as to wash out the initial low-quality self-play data
     more quickly.
  - `memory_analysis`: parameters for the memory analysis step that is
     performed at each iteration (see [`MemAnalysisParams`](@ref)), or
     `nothing` if no analysis is to be performed.

# AlphaGo Zero Parameters

In the original AlphaGo Zero paper:
- About 5 millions games of self-play are played across 200 iterations.
- The memory buffer contains 500K games, which makes about 100M samples
  as an average game of Go lasts about 200 turns.
"""
@kwdef struct Params
  self_play :: SelfPlayParams
  memory_analysis :: Union{Nothing, MemAnalysisParams} = nothing
  learning :: LearningParams
  arena :: Union{Nothing, ArenaParams}
  num_iters :: Int
  use_symmetries :: Bool = false
  ternary_rewards :: Bool = false
  mem_buffer_size :: PLSchedule{Int}
end

for T in [MctsParams, SimParams, ArenaParams, SelfPlayParams, LearningParams, Params]
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

#####
##### Consistency checking
#####

# This function checks for inconsistencies in the parameters.
# It returns a pair of lists of strings: `(errors, warnings)`.
# TODO: add more consistency checks.
function check_params(gspec::AbstractGameSpec, p::Params)
  errors = String[]
  warns = String[]
  # Collecting all relevant params
  mctss = [p.self_play.mcts]
  sims = [p.self_play.sim]
  if !isnothing(p.arena)
    push!(mctss, p.arena.mcts)
    push!(sims, p.arena.sim)
  end
  if any(sim.batch_size > sim.num_workers for sim in sims)
    push!(errors,
      "The number of simulation workers must be " *
      "greater or equal than the inference batch size.")
  end
  # Detecing non-provided symmetries
  if any(sim.flip_probability != 0 for sim in sims)
    state = GI.current_state(GI.init(gspec))
    if isempty(GI.symmetries(gspec, state))
      push!(errors, "You must specify some game symmetries to use flip_probability>0.")
    end
  end
  return (errors, warns)
end
