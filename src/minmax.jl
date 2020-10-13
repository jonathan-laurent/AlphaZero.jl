#####
##### A simple minmax player to be used as a baseline
#####

"""
A simple implementation of the minmax tree search algorithm, to be used as
a baseline against AlphaZero. Heuristic board values are provided by the
[`GameInterface.heuristic_value`](@ref) function.
"""
module MinMax

using ..AlphaZero

amplify(r) = iszero(r) ? r : Inf * sign(r)

# Return the value of the current state for the player playing
function value(player, game, depth)
  if GI.game_terminated(game)
    return 0.
  elseif depth == 0
    return GI.heuristic_value(game)
  else
    qs = [qvalue(player, game, a, depth) for a in GI.available_actions(game)]
    return maximum(qs)
  end
end

function qvalue(player, game, action, depth)
  @assert !GI.game_terminated(game)
  next = GI.clone(game)
  GI.play!(next, action)
  wr = GI.white_reward(next)
  r = GI.white_playing(game) ? wr : -wr
  if player.amplify_rewards
    r = amplify(r)
  end
  nextv = value(player, next, depth - 1)
  if GI.white_playing(game) != GI.white_playing(next)
    nextv = -nextv
  end
  return r + player.gamma * nextv
end

function minmax(player, game, actions, depth)
  return argmax([qvalue(player, game, a, player.depth) for a in actions])
end

"""
    MinMax.Player <: AbstractPlayer

A stochastic minmax player, to be used as a baseline.

    MinMax.Player(;depth, amplify_rewards, τ=0.)

The minmax player explores the game tree exhaustively at depth `depth`
to build an estimate of the Q-value of each available action. Then, it
chooses an action as follows:

- If there are winning moves (with value `Inf`), one of them is picked
  uniformly at random.
- If all moves are losing (with value `-Inf`), one of them is picked
  uniformly at random.

Otherwise,

- If the temperature `τ` is zero, a move is picked uniformly among those
  with maximal Q-value (there is usually only one choice).
- If the temperature `τ` is nonzero, the probability of choosing
  action ``a`` is proportional to ``e^{\\frac{q_a}{Cτ}}`` where ``q_a`` is the
  Q value of action ``a`` and ``C`` is the maximum absolute value of all
  finite Q values, making the decision invariant to rescaling of
  [`GameInterface.heuristic_value`](@ref).

If the `amplify_rewards` option is set to true, every received positive reward
is converted to ``∞`` and every negative reward is converted to ``-∞``.
"""
struct Player <: AbstractPlayer
  depth :: Int
  amplify_rewards :: Bool
  τ :: Float64
  gamma :: Float64
  function Player(;depth, amplify_rewards, τ=0., γ=1.)
    return new(depth, amplify_rewards, τ, γ)
  end
end

function AlphaZero.think(p::Player, game)
  actions = GI.available_actions(game)
  n = length(actions)
  qs = [qvalue(p, game, a, p.depth) for a in actions]
  winning = findall(==(Inf), qs)
  if isempty(winning)
    notlosing = findall(>(-Inf), qs)
    best = argmax(qs)
    if isempty(notlosing)
      π = ones(n)
    elseif iszero(p.τ)
      π = zeros(n)
      all_best = findall(==(qs[best]), qs)
      π[all_best] .= 1.
    else
      qmax = qs[best]
      @assert qmax > -Inf
      C = maximum(abs(qs[a]) for a in notlosing) + eps()
      π = exp.((qs .- qmax) ./ C)
      π .^= (1 / p.τ)
    end
  else
    π = zeros(n)
    π[winning] .= 1.
  end
  π ./= sum(π)
  return actions, π
end

end
