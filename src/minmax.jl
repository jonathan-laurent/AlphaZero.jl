#####
##### A simple minmax player to be used as a baseline
#####

"""
A simple implementation of the minmax tree search algorithm that relies on
[`GameInterface`](@ref Main.AlphaZero.GameInterface), to be used
as a baseline against AlphaZero. Heuristic board values are provided by
the [`GameInterface.heuristic_value`](@ref) function.
"""
module MinMax

import ..GI

function current_player_value(white_reward, white_playing) :: Float64
  if iszero(white_reward)
    return 0.
  else
    v = Inf * sign(white_reward)
    return white_playing ? v : - v
  end
end

# Return the value of a state for the player playing
function value(game, depth)
  wr = GI.white_reward(game)
  wp = GI.white_playing(game)
  if isnothing(wr)
    if depth == 0
      return GI.heuristic_value(game)
    else
      return maximum(qvalue(game, a, depth)
        for a in GI.available_actions(game))
    end
  else
    return current_player_value(wr, wp)
  end
end

function qvalue(game, action, depth)
  @assert isnothing(GI.white_reward(game))
  wp = GI.white_playing(game)
  game = copy(game)
  GI.play!(game, action)
  pswitch = wp != GI.white_playing(game)
  nextv = value(game, depth - 1)
  return pswitch ? - nextv : nextv
end

minmax(game, actions, depth) = argmax([qvalue(game, a, depth) for a in actions])

struct Player{G} <: GI.AbstractPlayer{G}
  depth :: Int
  τ :: Float64
  Player{G}(;depth, τ=0.) where G = new{G}(depth, τ)
end

function GI.think(p::Player, game, turn=nothing)
  actions = GI.available_actions(game)
  n = length(actions)
  qs = [qvalue(game, a, p.depth) for a in actions]
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
