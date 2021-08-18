OpenSpiel's game example
(you need cmake, and clang++ in order to build OpenSpiel)

to run this example open Julia session and execute:
```julia
using AlphaZero
import OpenSpiel # load additional features through Requires.jl

ospiel_experiment = Examples.experiments["ospiel_ttt"]
session = Session(ospiel_experiment, dir="sessions/ospiel_ttt")

resume!(session)
```