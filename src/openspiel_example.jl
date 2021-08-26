include("../games/ospiel_ttt/main.jl")
# export OSpielTictactoe
Examples.games["ospiel_ttt"] = OSpielTictactoe.GameSpec()
Examples.experiments["ospiel_ttt"] = OSpielTictactoe.Training.experiment
