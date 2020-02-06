using AlphaZero
using Test

include("../games/tictactoe/main.jl")
include("../games/connect-four/main.jl")
include("../scripts/lib/dummy_run.jl")

include("test_game.jl")

@testset "Testing Games" begin
  test_symmetries(Tictactoe.Game)
  test_symmetries(ConnectFour.Game)
end

@testset "Dummy Runs" begin
  @test dummy_run(Tictactoe)
  #@test dummy_run(ConnectFour) # Takes a bit too long for Travis
end
