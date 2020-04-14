#####
##### Tuning MCTS parameters
#####

using AlphaZero
using AlphaZero: Log, Util
using Formatting
using Base: @kwdef

const NUM_GAMES = 1000
const RESET_MCTS_EVERY = 10
const LOGGER = Log.Logger()

#####
##### MCTS parameters to tune
#####

@kwdef struct Config
  n :: Int      # num simulations
  w :: Int      # num workers
  c :: Float64  # cpuct
  τ :: Float64
  ϵ :: Float64
  α :: Float64
end

Util.generate_update_constructor(Config) |> eval

function Base.string(conf::Config)
  c = fmt(".1f", conf.c)
  τ = fmt(".2f", conf.τ)
  ϵ = fmt(".2f", conf.ϵ)
  α = fmt(".2f", conf.α)
  return "$(conf.n) simulations, $(conf.w) workers; c=$c, τ=$τ, ϵ=$ϵ, α=$α"
end

#####
##### Experiment Report
#####

@kwdef struct Report
  reward :: Float64
  time_per_game :: Float64
  inference_time_ratio :: Float64
  average_exploration_depth :: Float64
end

function print_report(report, legend)
  wr   = Log.ColType(10, x -> format("{:.2f}", x))
  time = Log.ColType(10, x -> format("{:.1f} min", x * 100 / 60))
  itr  = Log.ColType(5,  x -> format("{}%", round(Int, x * 100)))
  expd = Log.ColType(5,  x -> format("{:.1f}", x))
  tab  = Log.Table([
    ("AVGR", wr,   r -> r.reward),
    ("T100", time, r -> r.time_per_game),
    ("ITR",  itr,  r -> r.inference_time_ratio),
    ("EXPD", expd, r -> r.average_exploration_depth)])
  Log.table_row(LOGGER, tab, report, [legend])
end

#####
##### Measurement Code
#####

using Juno

function evaluate_player(network, params, baseline, n)
  player = AlphaZero.MctsPlayer(network, params)
  Rtot = 0.0
  time = @elapsed Juno.@progress for i in 1:n
    if i % RESET_MCTS_EVERY == 0
      MCTS.reset!(player.mcts)
    end
    Rtot += AlphaZero.play_game(player, baseline)
  end
  return Report(
    reward=(Rtot / n),
    time_per_game=(time / n),
    inference_time_ratio=MCTS.inference_time_ratio(player.mcts),
    average_exploration_depth=MCTS.average_exploration_depth(player.mcts))
end

function run_experiments(network, baseline, configs)
  for config in configs
    params = MctsParams(
      use_gpu=true,
      num_workers=config.w,
      num_iters_per_turn=config.n,
      cpuct=config.c,
      temperature=ConstSchedule(config.τ),
      dirichlet_noise_ϵ=config.ϵ,
      dirichlet_noise_α=config.α)
    evaluate_player(network, params, baseline, 1) # Compilation
    rep = evaluate_player(network, params, baseline, NUM_GAMES)
    print_report(rep, string(config))
  end
end

#####
##### Experiments
#####

include("../games.jl")
const GAME = "connect-four"
const SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game, Training

const SESSION_DIR = "sessions/connect-four"

session = Session(
  Game, Training.Network{Game},
  Training.params, Training.netparams, benchmark=Training.benchmark,
  dir=SESSION_DIR, autosave=false, save_intermediate=false)

configs = [
  Config(n=n, w=w, c=c, τ=τ, ϵ=ϵ, α=α)
  for τ in [1.0, 0.5, 0.1]
  for n in [400, 600]
  for w in [8, 16, 32]
  for ϵ in [0.1, 0.2]
  for c in [3.0]
  for α in [1.0]
]

network = session.env.bestnn
baseline = MinMax.Player{Game}(;depth=5, τ=0.2)
run_experiments(network, baseline, configs)
