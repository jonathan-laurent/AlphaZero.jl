using Distributed

Distributed.addprocs(4, exeflags="--project")

@show nworkers()

@everywhere using AlphaZero

include("include_workaround.jl")
include_everywhere("../../games/tictactoe/main.jl")

using .Tictactoe: Game, Training

function main()
  # Create an environment
  params = Training.params
  network = Training.Network{Game}(Training.netparams)
  env = AlphaZero.Env{Game}(params, network)
  duel = Training.benchmark[1]
  trace_lens = pmap(1:4) do i
    # println("hello")
    player = MctsPlayer(network, env.params.self_play.mcts)
    trace = play_game(player)
    return length(trace)
  end
  @show trace_lens
end

main()
