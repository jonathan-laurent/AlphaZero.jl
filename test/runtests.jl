using AlphaZero
using AlphaZero.Examples: games, experiments
using AlphaZero.Scripts: dummy_run, test_game

using Base.Filesystem: rm
using Test

const CI = get(ENV, "CI", nothing) == "true"
const FULL = !CI

@testset "Testing Games" begin
  test_game(games["grid-world"])
  test_game(games["tictactoe"])
  test_game(games["connect-four"])
  test_game(games["mancala"])
  @test true
end

@testset "Dummy Runs" begin
  dir = "sessions/test-tictactoe"
  @test dummy_run(experiments["tictactoe"], nostdout=false) == nothing
  if FULL
    @test dummy_run(experiments["connect-four"], nostdout=false) == nothing
    @test dummy_run(experiments["mancala"], nostdout=false) == nothing
  end
end