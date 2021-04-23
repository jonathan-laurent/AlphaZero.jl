using Documenter
using AlphaZero

const PRETTY_URLS = get(ENV, "CI", nothing) == "true"

makedocs(
  sitename = "AlphaZero",
  authors="Jonathan Laurent",
  format = Documenter.HTML(prettyurls=PRETTY_URLS),
  modules = [AlphaZero],
  pages = [
    "Home" => "index.md",
    "Guided Tour" => [
      "tutorial/alphazero_intro.md",
      "tutorial/package_overview.md",
      "tutorial/connect_four.md",
      "tutorial/own_game.md"
    ],
    "Reference" => [
      "reference/params.md",
      "reference/game_interface.md",
      "reference/mcts.md",
      "reference/network.md",
      "reference/networks_library.md",
      "reference/player.md",
      "reference/memory.md",
      "reference/environment.md",
      "reference/benchmark.md",
      "reference/reports.md",
      "reference/experiment.md",
      "reference/ui.md",
      "reference/scripts.md"
    ],
    "Contributing" => [
      "contributing/guide.md"
    ]
  ],
  repo="https://github.com/jonathan-laurent/AlphaZero.jl/blob/{commit}{path}#L{line}"
)

deploydocs(;
  repo="github.com/jonathan-laurent/AlphaZero.jl",
)
