using Documenter
using AlphaZero

const PRETTY_URLS = get(ENV, "CI", nothing) == "true"

makedocs(
    sitename = "AlphaZero",
    authors="Jonathan Laurent",
    format = Documenter.HTML(prettyurls=PRETTY_URLS),
    modules = [AlphaZero],
    pages = [
        "index.md",
        "game_interface.md",
        "network.md",
        "mcts.md",
        "misc.md"
    ],
    repo="https://github.com/jonathan-laurent/AlphaZero.jl/blob/{commit}{path}#L{line}"
)

deploydocs(;
    repo="github.com/jonathan-laurent/AlphaZero.jl",
)
