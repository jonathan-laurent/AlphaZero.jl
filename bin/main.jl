#####
##### Main script for the JuliaCon 2020 demo
#####

ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # "binned" / "split"

# Enables running the script on a distant machine without an X server
ENV["GKSwstype"]="nul"

using AlphaZero

include("../games/connect-four/main.jl")
using .ConnectFour: Game, Training

session = Session(
  Game,
  Training.Network{Game},
  Training.params,
  Training.netparams,
  benchmark=Training.benchmark,
  dir="sessions/connect-four",
  autosave=true,
  save_intermediate=false)
