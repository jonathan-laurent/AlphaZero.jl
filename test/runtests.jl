using AlphaZero
using Test

include("../games/tictactoe/main.jl")
using .Tictactoe: Game, Training

@testset "AlphaZero.jl" begin
  @test begin
    Net, netparams, params, benchmark = Training.get_params(:debug)
    session = Session(Game, Net{Game}, params, netparams,
      benchmark=benchmark, nostdout=true)
    resume!(session)
    return true
  end
end
