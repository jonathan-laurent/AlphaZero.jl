#####
##### Main script for the JuliaCon 2020 demo
#####

ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # "binned" / "split"

# Enables running the script on a distant machine without an X server
ENV["GKSwstype"]="nul"

using AlphaZero

const DUMMY_RUN = false
include("../scripts/lib/dummy_run.jl")

include("../games/connect-four/main.jl")
using .ConnectFour: Game, Training

params, benchmark = Training.params, Training.benchmark
if DUMMY_RUN
  params, benchmark = dummy_run_params(params, benchmark)
end

session = Session(
  Game,
  Training.Network{Game},
  params,
  Training.netparams,
  benchmark=benchmark,
  dir="sessions/connect-four",
  autosave=true,
  save_intermediate=false)

resume!(session)
