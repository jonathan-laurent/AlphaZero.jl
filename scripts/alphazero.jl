#####
##### Simple CLI for AlphaZero.jl
##### This file can also be included directly in the REPL.
#####

# ENV["CUARRAYS_MEMORY_LIMIT"] = 7_500_000_000
ENV["CUARRAYS_MEMORY_POOL"] = "split" # "binned" / "split"

using AlphaZero
include("games.jl")

#####
##### Parse arguments
#####

using ArgParse
argstab = ArgParseSettings()
@add_arg_table argstab begin
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

session = Session(
  Game, Training.Network{Game}, Training.params, Training.netparams,
  dir=SESSION_DIR, autosave=true, benchmark=Training.benchmark)

if cmd == "train"
  resume!(session)
elseif cmd == "explore"
  explore(session)
elseif cmd == "play"
  play_game(session)
end
