# A simple grid world MDP
# All cells with reward are also terminal
# Environment created by: Zachary Sunberg

using CommonRLInterface
using StaticArrays
using Crayons

const RL = CommonRLInterface

# To avoid episodes of unbounded length, we put an arbitrary limit to the length of an
# episode. Because time is not captured in the state, this introduces a slight bias in
# the value function.
const EPISODE_LENGTH_BOUND = 200
const SIZE = SA[10, 10]
const REWARDS =     Dict(
		    SA[9,3] =>  10.0,
		    SA[8,8] =>   3.0,
		    SA[4,3] => -10.0,
		    SA[4,6] =>  -5.0)

mutable struct World <: AbstractEnv
  position ::SVector{2, Int}
  time :: Int
end

function World()
  return World(
    SA[rand(1:10), rand(1:10)],
    0)
end

RL.reset!(env::World) = (env.position = SA[rand(1:SIZE[1]), rand(1:SIZE[2])])
RL.actions(env::World) = [SA[1,0], SA[-1,0], SA[0,1], SA[0,-1]]
RL.observe(env::World) = env.position

RL.terminated(env::World) =
  haskey(REWARDS, env.position) || env.time > EPISODE_LENGTH_BOUND

function RL.act!(env::World, a)
  # 40% chance of going in a random direction (=30% chance of going in a wrong direction)
  if rand() < 0.4
      a = rand(actions(env))
  end
  env.position = clamp.(env.position + a, SA[1,1], SIZE)
  env.time += 1
  return get(REWARDS, env.position, 0.0)
end

@provide RL.player(env::World) = 1 # An MDP is a one player game
@provide RL.players(env::World) = [1]
@provide RL.observations(env::World) = [SA[x, y] for x in 1:SIZE[1], y in 1:SIZE[2]]
@provide RL.clone(env::World) = World(env.position, env.time)
@provide RL.state(env::World) = env.position
@provide RL.setstate!(env::World, s) = (env.position = s)
@provide RL.valid_action_mask(env::World) = BitVector([1, 1, 1, 1])

# Additional functions needed by AlphaZero.jl that are not present in 
# CommonRlInterface.jl. Here, we provide them by overriding some functions from
# GameInterface. An alternative would be to pass them as keyword arguments to
# CommonRLInterfaceWrapper.

function GI.render(env::World)
  for y in reverse(1:SIZE[2])
    for x in 1:SIZE[1]
      s = SA[x, y]
      r = get(REWARDS, s, 0.0)
      if env.position == s
        c = ("+",)
      elseif r > 0
        c = (crayon"green", "o")
      elseif r < 0
        c = (crayon"red", "o")
      else
        c = (crayon"dark_gray", ".")
      end
      print(c..., " ", crayon"reset")
    end
    println("")
  end
end

function GI.vectorize_state(env::World, state)
  v = zeros(Float32, SIZE[1], SIZE[2])
  v[state[1], state[2]] = 1
  return v
end

const action_names = ["r", "l", "u", "d"]

function GI.action_string(env::World, a)
  idx = findfirst(==(a), RL.actions(env))
  return isnothing(idx) ? "?" : action_names[idx]
end

function GI.parse_action(env::World, s)
  idx = findfirst(==(s), action_names)
  return isnothing(idx) ? nothing : RL.actions(env)[idx]
end

function GI.read_state(env::World)
  try
    s = split(readline())
    @assert length(s) == 2
    x = parse(Int, s[1])
    y = parse(Int, s[2])
    @assert 1 <= x <= SIZE[1]
    @assert 1 <= y <= SIZE[2]
    return SA[x, y]
  catch e
    return nothing
  end
end

GI.heuristic_value(::World) = 0.

GameSpec() = CommonRLInterfaceWrapper.Spec(World())