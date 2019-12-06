using AlphaZero
using Test

include("../scripts/game_module.jl")
@game_module TicTacToe "tictactoe"
using .TicTacToe: Game, Training

@testset "AlphaZero.jl" begin
  @test begin
    Net, netparams, params, benchmark = Training.get_params(:debug)
    session = Session(
      Game, Net, params, netparams,
      dir=Training.SESSION_DIR, autosave=false, benchmark=benchmark)
    resume!(session)
    return true
  end
end
