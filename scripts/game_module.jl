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

function find_game_dir(name)
  game_dir = joinpath("games", name)
  for i in 0:5
    game_dir = joinpath(
      pwd(), [".." for j in 1:i]..., "games", name)
    isdir(game_dir) && break
  end
  isdir(game_dir) || error("Game directory not found")
  return game_dir
end

const DEFAULT_GAME = "connect-four"

macro game_module(Mname, game=nothing)
  if isnothing(game)
    game = get(ENV, "GAME", DEFAULT_GAME)
  end
  mod = Symbol(snake(game))
  game_dir = find_game_dir(game)
  game_file = joinpath(game_dir, "game.jl")
  params_file = joinpath(game_dir, "params.jl")
  session_dir = joinpath("sessions", game)
  return Expr(:toplevel, quote
    @eval module $Mname
      export Training, Game, Board
      include($game_file)
      module Training
        using AlphaZero
        import ..Game
        include($params_file)
        const SESSION_DIR = $session_dir
      end
    end
  end)
end
