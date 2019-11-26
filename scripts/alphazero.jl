# CuArrays settings
ENV["CUARRAYS_MEMORY_LIMIT"] = 7_500_000_000
ENV["CUARRAYS_MEMORY_POOL"] = "binned" # "binned" / "split"

using ArgParse
using Revise
using AlphaZero

argstab = ArgParseSettings
available_games = ["tictactoe", "connect-four", "mancala"]
@add_arg_table argstab begin
  "train"
    action = :command
    help = "Resume the training session"
  "play"
    action = :command
    help = "Play against the current "
  "--game"
    help = "Select a game ($(join(available_games, "/")))"
end
args = parse_args(ARGS, argstab)
if !isnothing(args["game"])
  ENV["GAME"] = args["game"]
end
cmd = args["%COMMAND%"]
isnothing(cmd) && (cmd = "train")

include("using_game.jl")
@using_default_game

session = Session(
  Game, Network, params, netparams,
  dir=SESSION_DIR, autosave=true, validation=validation)

if cmd == "train"
  resume!(session)
end

# explore(session)

# Play a game against the computer
function play_game(session)
  net = AlphaZero.Network.copy(session.env.bestnn, on_gpu=true, test_mode=true)
  mcts = MCTS.Env{Game}(net, nworkers=64)
  GI.interactive!(Game(), MCTS.AI(mcts, timeout=5.0), GI.Human())
end

if cmd == "play"
  play_game(session)
end
