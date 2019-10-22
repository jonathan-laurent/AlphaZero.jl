using Documenter
using AlphaZero

makedocs(
    sitename = "AlphaZero",
    format = Documenter.HTML(prettyurls = false),
    modules = [AlphaZero]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
