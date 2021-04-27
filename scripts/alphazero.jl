#####
##### Simple CLI for AlphaZero.jl
#####

# We recommend using the cuda asynchronous memory pool, which should be used by default
# by CUDA.jl if your CUDA toolkit version is 11.2 or higher. As a second choice, you may
# also use the splitting pool (the binned pool is not recommended).

# ENV["JULIA_CUDA_MEMORY_POOL"] = "cuda"  # "cuda" / "split" "binned"

# Enables running the script on a distant machine without an X server
ENV["GKSwstype"]="nul"

using AlphaZero
using ArgParse
import Distributed

available_experiments = keys(Examples.experiments)

settings = ArgParseSettings()

@add_arg_table! settings begin
  "experiment"
    help = "Select an experiment ($(join(available_experiments, ", ")))"
    required=true
  "--save-intermediate"
    action = :store_true
    help = "Save all intermediate states during training"
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
  "check-game"
    action = :command
    help = "Check that the current game respects all expected invariants"
end

args = parse_args(settings)
cmd = args["%COMMAND%"]
experiment_name = args["experiment"]

if experiment_name ∉ available_experiments
  println(stderr, "Unknown experiment: $(experiment_name)")
  exit(1)
end

experiment = Examples.experiments[experiment_name]

session_dir = UserInterface.default_session_dir(experiment_name)

if cmd == "check-game"
  Scripts.test_game(experiment.gspec)
  @info "All tests passed."
else
  println("\nUsing $(Distributed.nworkers()) distributed worker(s).\n")
  session = UserInterface.Session(
    experiment,
    dir=session_dir,
    autosave=true,
    save_intermediate=args["save-intermediate"])
  if cmd == "train"
    UserInterface.resume!(session)
  elseif cmd == "explore"
    UserInterface.explore(session)
  elseif cmd == "play"
    interactive!(experiment.gspec, AlphaZeroPlayer(session), Human())
  elseif cmd == "replot"
    UserInterface.regenerate_plots(session)
  end
end
