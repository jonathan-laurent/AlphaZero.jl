#####
##### Simple CLI for AlphaZero.jl
##### This file can also be included directly in the REPL.
#####

ENV["CUARRAYS_MEMORY_LIMIT"] = 7_500_000_000
ENV["CUARRAYS_MEMORY_POOL"] = "binned" # "binned" / "split"

using AlphaZero

#####
##### Parse arguments
#####

using ArgParse
argstab = ArgParseSettings()
available_games = ["tictactoe", "connect-four", "mancala"]
@add_arg_table argstab begin
  "--game"
    help = "Select a game ($(join(available_games, "/")))"
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

#####
##### Main
#####

include("game_module.jl")
@game_module SelectedGame "tictactoe"
using .SelectedGame: Game, Training

session = Session(
  Game, Training.Network, Training.params, Training.netparams,
  dir=Training.SESSION_DIR, autosave=true, benchmark=Training.benchmark)

if cmd == "train"
  resume!(session)
elseif cmd == "explore"
  explore(session)
elseif cmd == "play"
  play_game(session)
end
