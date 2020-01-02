#####
##### Interface to Pascal Pons' Connect 4 Solver
##### https://github.com/PascalPons/connect4
#####

# Problem: no Connect4 module. We can change this:

module Solver

import ..Game

import AlphaZero: GI, GameInterface, AbstractPlayer

struct Player <: AbstractPlayer{Game}
  process :: Base.Process
  ϵ_random :: Float32
  function Player(;
      solver_dir=joinpath(@__FILE__, "solver", "connect4"),
      solver_name="c4solver", ϵ_random=0.)
    cmd = Cmd(`./$solver_name`, dir=solver_dir)
    p = open(cmd, "r+")
    return new(p, ϵ_random)
  end
end

end
