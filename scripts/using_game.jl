#####
##### Utilities to load available games
#####

# Convert a string identifier in upper snake case
function snake(str)
  ret = Char[]
  new_word = true
  for c in str
    if c ∈ ['-', '_']
      new_word = true
    else
      push!(ret, new_word ? uppercase(c) : c)
      new_word = false
    end
  end
  return String(ret)
end

macro using_game(name)
  mod = Symbol(snake(name))
  game_dir = joinpath("..", "games", name)
  game_file = joinpath(game_dir, "game.jl")
  params_file = joinpath(game_dir, "params.jl")
  session_dir = "session-$name"
  @eval begin
    include($game_file)
    using .$mod
    include($params_file)
    SESSION_DIR = $session_dir
  end
end

const DEFAULT_GAME = "connect-four"

const GAME = haskey(ENV, "GAME") ? ENV["GAME"] : DEFAULT_GAME

macro using_default_game()
  :(@using_game $GAME)
end
