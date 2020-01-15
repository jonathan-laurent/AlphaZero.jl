using AlphaZero
using Test

include("../games/tictactoe/main.jl")
include("../scripts/lib/dummy_run.jl")

@testset "Dummy Run on Tictactoe" begin
  @test dummy_run(Tictactoe)
end
