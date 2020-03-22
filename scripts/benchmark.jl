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

# Example: run a network-only benchmark

make_duel(baseline) =
  Benchmark.Duel(
    Benchmark.NetworkOnly(),
    baseline,
    num_games=200,
    flip_probability=0.5,
    color_policy=CONTENDER_WHITE)

benchmark = make_duel.(Training.baselines)

run_new_benchmark(
  Game, Training.Network{Game}, SESSION_DIR,
  "netonly", benchmark,
  params=Training.params, itcmax=nothing)
