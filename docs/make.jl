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
        "Tutorial" => [
            "tutorial/alphazero_intro.md",
            "tutorial/connect-four.md",
            "tutorial/add_game.md"],
        "Reference" => [
            "reference/params.md",
            "reference/game_interface.md",
            "reference/mcts.md",
            "reference/network.md",
            "reference/misc.md"
        ],
        "Contribution Guide" => "contributing.md"
    ],
    repo="https://github.com/jonathan-laurent/AlphaZero.jl/blob/{commit}{path}#L{line}"
)

deploydocs(;
    repo="github.com/jonathan-laurent/AlphaZero.jl",
)
