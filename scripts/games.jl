#####
##### Repository of Available Games
#####

const AVAILABLE_GAMES = ["tictactoe", "connect-four" #=, "mancala" =#]

for game in AVAILABLE_GAMES
  file = "../games/$game/main.jl"
  @eval include($file)
end

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

const GAME_MODULE = Dict([g => eval(Symbol(snake(g))) for g in AVAILABLE_GAMES])
