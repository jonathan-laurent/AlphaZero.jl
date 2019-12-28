using AlphaZero
using Test

include("../scripts/game_module.jl")
@game_module TicTacToe "tictactoe"
using .TicTacToe: Game, Training

@testset "AlphaZero.jl" begin
  @test begin
    Net, netparams, params, benchmark = Training.get_params(:debug)
    session = Session(Game, Net{Game}, params, netparams,
      benchmark=benchmark, nostdout=true)
    resume!(session)
    return true
  end
end
