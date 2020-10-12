#####
##### AlphaZero.jl
##### Jonathan Laurent, Carnegie Mellon University (2019-2020)
#####

module AlphaZero

# Helper functions used internally
include("util.jl")

# A generic interface for single-player or zero-sum two-players games.
include("game.jl")
const GI = GameInterface

# A standalone, generic MCTS implementation
include("mcts.jl")

# A generic network interface
include("networks/network.jl")

# Utilities to batch oracle calls
include("batchifier.jl")

# Implementation of the core training algorithm
include("simulations.jl")

# Implementation of the core training algorithm
include("core.jl")

# A minmax player to be used as a baseline
include("minmax.jl")

# Utilities to write benchmarks
include("benchmark.jl")

# We provide a library of standard network, both in Knet and Flux.
# Which backend is used to implement this library is determined during precompilation
# based on the value of the ALPHAZERO_DEFAULT_DL_FRAMEWORK environment variable.
const DEFAULT_DL_FRAMEWORK = get(ENV, "ALPHAZERO_DEFAULT_DL_FRAMEWORK", "FLUX")

if DEFAULT_DL_FRAMEWORK == "FLUX"
  @info "Using the Flux implementation of AlphaZero.NetLib."
  @eval begin
    include("networks/flux.jl")
    const NetLib = FluxLib
  end
elseif DEFAULT_DL_FRAMEWORK == "KNET"
  @info "Using the Knet implementation of AlphaZero.NetLib."
  @eval begin
    include("networks/knet.jl")
    const NetLib = KnetLib
  end
else
  error("Unknown DL framework: $(DEFAULT_DL_FRAMEWORK)")
end

# A structure that contains the information necessary to replicate a training session
include("experiments.jl")

# The default user interface is included here for convenience but it could be
# replaced or separated from the main AlphaZero.jl package (which would also
# enable dropping some dependencies such as Crayons or JSON3).
include("ui/ui.jl")

# A small library of standard examples
include("examples.jl")

# Reexporting some names
for m in [
    GameInterface, GI, MCTS, Network, Simulations, Training,
    MinMax, Benchmark, NetLib, UserInterface ]
  for x in names(m)
    println(x)
    @eval export $x
  end
end

end