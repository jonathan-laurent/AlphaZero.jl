using Plots
using Statistics

using CuArrays
using Flux

import AlphaZero; const AZ = AlphaZero
import AlphaZero.GI

include("../games/tictactoe/game.jl")
import .TicTacToe
const Game = TicTacToe.Game

FIG_FILE = "inference-speedup.png"
REP  = 100
NMAX = 64

netparams = AlphaZero.SimpleNetParams(
  width=500,
  depth_common=7)

function measure(network, X, A)
  t = @elapsed begin
    for i in 1:REP
      network(X, A)
    end
  end
  return t/REP
end

network = AlphaZero.SimpleNet{Game, netparams}()
game = Game()
board = GI.board(game)
actions = GI.available_actions(game)
x = GI.vectorize_board(Game, board)
a = GI.actions_mask(Game, actions)

function batchify(n)
  X = AZ.Util.concat_columns((x for i in 1:n))
  A = AZ.Util.concat_columns((a for i in 1:n))
  return X, A
end

single() = mean(measure(network, x, a) for i in 1:10)

function batch(n)
  X, A = batchify(n)
  measure(network, X, A)
end

function batch_gpu(n)
  X, A = Flux.gpu.(batchify(n))
  net = Flux.gpu(network)
  measure(net, X, A)
end

# Compile everything
single()
batch(2)
batch_gpu(2)

tsingle = single()
tbatches = [batch(n) for n in 1:NMAX]
tbatches_gpu = [batch_gpu(n) for n in 1:NMAX]

speedup_curve(ts) = [(i * tsingle) / t for (i, t) in enumerate(ts)]

title="Inference speedup as a funtion of batch size"
hline([1], label="Baseline", title=title)
plot!(speedup_curve(tbatches), label="CPU")
plot!(speedup_curve(tbatches_gpu), label="GPU")
xlabel!("Batch size")
ylabel!("Speedup")
savefig(FIG_FILE)
