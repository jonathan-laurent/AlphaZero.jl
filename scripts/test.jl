using AlphaZero

include("../games/tictactoe/main.jl")

function generate_states(::Type{Game}, n) where Game
  traces = []
  player = RandomPlayer{Game}()
  for i in 1:n
    trace = play_game(player)
    push!(traces, trace)
  end
  return Set(s for t in traces for s in t.states)
end

using Juno
generate_states(Tictactoe.Game, 3)
