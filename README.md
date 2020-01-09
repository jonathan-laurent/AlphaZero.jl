# AlphaZero.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jonathan-laurent.github.io/AlphaZero.jl/dev)
[![Build Status](https://travis-ci.com/jonathan-laurent/AlphaZero.jl.svg?branch=master)](https://travis-ci.com/jonathan-laurent/AlphaZero.jl)
[![Codecov](https://codecov.io/gh/jonathan-laurent/AlphaZero.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jonathan-laurent/AlphaZero.jl)

This package provides a _generic_, _simple_ and _fast_ implementation of
Deepmind's AlphaZero algorithm:

* The core algorithm is only 2,000 lines of pure, hackable Julia code.
* Generic interfaces make it easy to add support for
  new games or new learning frameworks.
* Being about two orders of magnitude faster than competing alternatives
  written in Python, this implementation enables to solve nontrivial games on
  a standard desktop computer with a GPU.
