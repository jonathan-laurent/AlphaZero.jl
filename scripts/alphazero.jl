#####
##### Simple CLI for AlphaZero.jl
##### This file can also be included directly in the REPL.
#####

# ENV["CUARRAYS_MEMORY_LIMIT"] = 7_500_000_000
ENV["CUARRAYS_MEMORY_POOL"] = "split" # "binned" / "split"

# Enables running the script on a distant machine without an X server
ENV["GKSwstype"]="nul"

using AlphaZero
include("games.jl")

#####
##### Parse arguments
#####

using ArgParse
argstab = ArgParseSettings()
@add_arg_table! argstab begin
  "--game"
    help = "Select a game ($(join(AVAILABLE_GAMES, "/")))"
  "train"
    action = :command
    help = "Resume the training session"
  "play"
    action = :command
    help = "Play against the current system"
  "explore"
    action = :command
    help = "Use the interactive exploration system"
  "replot"
    action = :command
    help = "Regenerate all the session plots from the JSON reports"
  "--save-intermediate"
    action = :store_true
    help = "Save all intermediate states during training"
end
args = parse_args(isempty(ARGS) ? ["train"] : ARGS, argstab)
!isnothing(args["game"]) && (ENV["GAME"] = args["game"])
cmd = args["%COMMAND%"]

if !haskey(ENV, "GAME")
  println(stderr, "You must specify a game.")
  exit()
end

#####
##### Main
#####

GAME = ENV["GAME"]
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game, Training

const SESSION_DIR = joinpath("sessions", GAME)

params = Training.params
netparams = Training.netparams
benchmark = Training.benchmark

include("lib/dummy_run.jl")

if get(ENV, "DUMMY_RUN", "false") == "true"
  @warn "Running dummy run"
  params, benchmark = dummy_run_params(Training.params, Training.benchmark)
end

session = Session(
  Game, Training.Network{Game}, params, netparams, benchmark=benchmark,
  dir=SESSION_DIR, autosave=true, save_intermediate=args["save-intermediate"])

if cmd == "train"
  resume!(session)
elseif cmd == "explore"
  start_explorer(session)
elseif cmd == "play"
  play_interactive_game(session)
elseif cmd == "replot"
  AlphaZero.UserInterface.regenerate_plots(session)
end
