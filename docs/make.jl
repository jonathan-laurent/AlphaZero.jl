using Documenter
using AlphaZero

makedocs(
    sitename = "AlphaZero",
    format = Documenter.HTML(prettyurls = false),
    modules = [AlphaZero],
    pages = [
        "index.md",
        "game_interface.md",
        "network.md",
        "mcts.md",
        "misc.md"
    ]
)
