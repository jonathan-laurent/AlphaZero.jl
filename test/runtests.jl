using AlphaZero
using AlphaZero.Examples
using AlphaZero.Scripts: dummy_run

using Test

include("test_game.jl")

@testset "Testing Games" begin
  # test_game(Tictactoe.GameSpec())
  test_game(ConnectFour.GameSpec())
  @test true
end

@testset "Dummy Runs" begin
  dir = "sessions/test-tictactoe"
  # @test dummy_run(Tictactoe.Training.experiment, nostdout=false, session_dir=dir)
  # @test dummy_run(ConnectFour) # Takes a bit too long for Travis
end
