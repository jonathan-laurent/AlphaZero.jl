#####
##### Simple CLI for AlphaZero.jl
##### This file can also be included directly in the REPL.
#####

ENV["CUARRAYS_MEMORY_LIMIT"] = 7_500_000_000
ENV["CUARRAYS_MEMORY_POOL"] = "split" # "binned" / "split"

using Revise
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

include("using_game.jl")
@using_default_game

session = Session(
  Game, Net, params, netparams,
  dir=SESSION_DIR, autosave=true, validation=validation)

if cmd == "train"
  resume!(session)
elseif cmd == "explore"
  explore(session)
elseif cmd == "play"
  play_game(session)
end
