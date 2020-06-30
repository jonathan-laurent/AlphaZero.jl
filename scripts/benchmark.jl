#####
##### Benchmark.jl
##### Script to run a new benchmark on a previous training session
##### (requires training with option `save_intermediate=true`)
#####

using AlphaZero

include("games.jl")
GAME = get(ENV, "GAME", "connect-four")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game, Training

SESSION_DIR = "sessions/$GAME"


# Examples: run a network-only benchmark

netonly = Benchmark.NetworkOnly(τ=0.5, use_gpu=true)
alphazero = Benchmark.Full(Training.arena.mcts)
baselines = [Training.mcts_baseline, Training.minmax_baseline]

make_duel(player, baseline) =
  Benchmark.Duel(
    player,
    baseline,
    num_games=200,
    reset_every=40,
    flip_probability=0.5,
    color_policy=CONTENDER_WHITE)

netonly_benchmark   = [make_duel(netonly, b) for b in baselines]
alphazero_benchmark = [make_duel(alphazero, b) for b in baselines]

benchmark = alphazero_benchmark
label = "full"

AlphaZero.UserInterface.run_new_benchmark(
  Game, Training.Network{Game}, SESSION_DIR,
  label, benchmark,
  params=Training.params, itcmax=nothing)
