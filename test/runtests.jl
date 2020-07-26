using AlphaZero
using Test

include("../games/tictactoe/main.jl")
include("../games/connect-four/main.jl")
include("../scripts/lib/dummy_run.jl")

include("test_game.jl")

@testset "Testing Games" begin
  test_game(Tictactoe.Game)
  test_game(ConnectFour.Game)
  @test true
end

@testset "Dummy Runs" begin
  dir = "sessions/test-tictactoe"
  @test dummy_run(Tictactoe, nostdout=false, session_dir=dir)
  #@test dummy_run(ConnectFour) # Takes a bit too long for Travis
end
