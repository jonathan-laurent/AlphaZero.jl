# Example using the OpenSpiel wrapper

This is an example of using AlphaZero.jl with OpenSpiel.

To run this example, open a Julia session and execute:

```julia
using AlphaZero
import OpenSpiel # load additional features through Requires.jl

ospiel_experiment = Examples.experiments["ospiel_ttt"]
session = Session(ospiel_experiment, dir="sessions/ospiel_ttt")

resume!(session)
```

Or more simply:

```sh
julia --project -e 'using AlphaZero; import OpenSpiel; Scripts.train("ospiel_ttt")'
```

Note that you may need to install `cmake` and `clang++` in order to
successfully install and build `OpenSpiel.jl`.

Also, the OpenSpiel wrapper hasn't been extensively tested yet so you
might experience some rough edges when using it with different games.
