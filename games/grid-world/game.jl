# A simple grid world MDP
# All cells with reward are also terminal
# Environment created by: Zachary Sunberg

using CommonRLInterface
using StaticArrays
using Crayons

const RL = CommonRLInterface

mutable struct World <: AbstractMarkovEnv
  size::SVector{2, Int}
  rewards::Dict{SVector{2, Int}, Float64}
  state::SVector{2, Int}
end

function World()
  rewards = Dict(
    SA[9,3] =>  10.0,
    SA[8,8] =>   3.0,
    SA[4,3] => -10.0,
    SA[4,6] =>  -5.0)
  return World(
    SA[10, 10],
    rewards,
    SA[rand(1:10), rand(1:10)])
end

RL.reset!(env::World) = (env.state = SA[rand(1:env.size[1]), rand(1:env.size[2])])
RL.actions(env::World) = [SA[1,0], SA[-1,0], SA[0,1], SA[0,-1]]
RL.observe(env::World) = env.state

RL.terminated(env::World) = haskey(env.rewards, env.state)

function RL.act!(env::World, a)
  # 40% chance of going in a random direction (=30% chance of going in a wrong direction)
  if rand() < 0.4
      a = rand(actions(env))
  end
  env.state = clamp.(env.state + a, SA[1,1], env.size)
  return get(env.rewards, env.state, 0.0)
end

# optional functions
@provide RL.observations(env::World) = [SA[x, y] for x in 1:env.size[1], y in 1:env.size[2]]
@provide RL.clone(env::World) = World(env.size, copy(env.rewards), env.state)
@provide RL.state(env::World) = env.state
@provide RL.setstate!(env::World, s) = (env.state = s)
@provide RL.valid_action_mask(env::World) = BitVector([1, 1, 1, 1])

# Additional functions defined for AlphaZero.jl

function render_ascii(env::World)
  for y in reverse(1:env.size[2])
    for x in 1:env.size[1]
      s = SA[x, y]
      r = get(env.rewards, s, 0.0)
      if env.state == s
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

function vectorize_state(env::World, state)
  v = zeros(Float32, env.size[1], env.size[2])
  v[state[1], state[2]] = 1
  return v
end

const action_names = ["r", "l", "u", "d"]

function action_string(env::World, a)
  idx = findfirst(==(a), RL.actions(env))
  return isnothing(idx) ? "?" : action_names[idx]
end

function parse_action(env::World, s)
  idx = findfirst(==(s), action_names)
  return isnothing(idx) ? nothing : RL.actions(env)[idx]
end

function read_state(env::World)
  try
    s = split(readline())
    @assert length(s) == 2
    x = parse(Int, s[1])
    y = parse(Int, s[2])
    @assert 1 <= x <= env.size[1]
    @assert 1 <= y <= env.size[2]
    return SA[x, y]
  catch e
    return nothing
  end
end

# Bridge with AlphaZero.jl

GameSpec() = CommonRLInterfaceWrapper.Spec(
  World(),
  heuristic_value=(s->0.),
  vectorize_state=vectorize_state,
  render=render_ascii,
  action_string=action_string,
  parse_action=parse_action,
  read_state=read_state)