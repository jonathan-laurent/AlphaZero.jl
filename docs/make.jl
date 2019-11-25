using Documenter
using AlphaZero

makedocs(
    sitename = "AlphaZero",
    authors="Jonathan Laurent",
    format = Documenter.HTML(prettyurls=false),
    modules = [AlphaZero],
    pages = [
        "index.md",
        "game_interface.md",
        "network.md",
        "mcts.md",
        "misc.md"
    ],
    repo="github.com/jonathan-laurent/AlphaZero.jl/blob/{commit}{path}#L{line}"
)

deploydocs(;
    repo="github.com/jonathan-laurent/AlphaZero.jl",
)
