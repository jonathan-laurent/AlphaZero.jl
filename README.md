# AlphaZero.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jonathan-laurent.github.io/AlphaZero.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jonathan-laurent.github.io/AlphaZero.jl/dev)
[![Build Status](https://travis-ci.com/jonathan-laurent/AlphaZero.jl.svg?branch=master)](https://travis-ci.com/jonathan-laurent/AlphaZero.jl)
[![Codecov](https://codecov.io/gh/jonathan-laurent/AlphaZero.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jonathan-laurent/AlphaZero.jl)

A generic implementation of Deepmind's AlphaZero algorithm in Julia.
New games can easily be added by implementing the `AlphaZero.GameInterface`
interface. You can look at `games/tictactoe` for an example.
