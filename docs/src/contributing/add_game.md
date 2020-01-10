# [Adding New Games](@id add_game)

When adding support for a new game, we recommend that you follow a few
conventions that will enable it to work with the game-agnostic utilities
in the `scripts` folder (such as `scripts/alphazero.jl`).

First, create a `your-game` directory in `games`. The name of this directory
will be used as a string identifier for your game by various scripts. In
this directory, create a `main.jl` file that defines a module `YourGame`
according to the following template:

```julia
module YourGame
  export Game, Board
  include("game.jl")
  module Training
    using AlphaZero
    include("params.jl")
  end
end
```

By convention, the name of your module should be the transposition of the
game directory's name in upper snake case. Also:

- `games/your-game/game.jl` should define your `Game` type,
    following the [Game Interface](@ref game_interface)
- `games/your-game/params.jl` should define
  `params`, `Network`, `netparams` and `benchmark` as explained in the
    [Connect Four Tutorial](@ref connect_four).

Finally, to register your game, just add `"your-game"` to `AVAILABLE_GAMES`
in `scripts/games.jl`. If you've done everything correctly, you should be
able to run a training session for your game by running:

```sh
julia --project --color=yes scripts/alphazero.jl --game your-game train
```
