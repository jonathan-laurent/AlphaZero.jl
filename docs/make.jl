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
        "Reference" => [
            "alphazero.md",
            "game_interface.md",
            "mcts.md",
            "network.md",
            "misc.md"
        ]
    ],
    repo="https://github.com/jonathan-laurent/AlphaZero.jl/blob/{commit}{path}#L{line}"
)

deploydocs(;
    repo="github.com/jonathan-laurent/AlphaZero.jl",
)
