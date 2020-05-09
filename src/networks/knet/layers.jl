
#####
##### Chain
#####

# Note: Knet serialization requires everything to be mutable

mutable struct Chain
  layers
  Chain(xs...) = new(xs)
end

children(c::Chain) = c.layers

mapchildren(f, c::Chain) = Chain((f(l) for l in c.layers)...)

function (c::Chain)(x)
  for l in c.layers
    x = l(x)
  end
  return x
end

#####
##### Dense
#####

mutable struct Dense
  W
  b
  σ
end

function Dense(in::Int, out::Int, σ=identity)
  W = Knet.param(Float32, out, in, atype=Array)
  b = Knet.param0(Float32, out, atype=Array)
  return Dense(W, b, σ)
end

children(c::Dense) = (c.W, c.b, c.σ)

mapchildren(f, c::Dense) = Dense(f(c.W), f(c.b), f(c.σ))

(d::Dense)(x) = d.σ.(d.W * x .+ d.b)

#####
##### Conv
#####

mutable struct Conv
  W
  pad
end

function Conv(ksize::Tuple, nchans::Pair; pad=0)
  W = Knet.param(Float32, ksize..., nchans..., atype=Array)
  return Conv(W, pad)
end

(c::Conv)(x) = Knet.conv4(c.W, x, padding=c.pad)

children(c::Conv) = (c.W,)

mapchildren(f, c::Conv) = Conv(f(c.W), c.pad)

#####
##### BatchNorm
#####

mutable struct BatchNorm
  moments
  params
  train :: Bool
  activation
end

function BatchNorm(nchans::Int, activation=identity; momentum)
  moments = Knet.bnmoments(momentum=momentum)
  params = Knet.bnparams(Float32, nchans) |> Knet.param
  BatchNorm(moments, params, true, activation)
end

function (l::BatchNorm)(x)
  y = Knet.batchnorm(x, l.moments, l.params, training=l.train)
  return l.activation.(y)
end

#####
##### Skip connection
#####

mutable struct SkipConnection
  block
  connection
end

children(c::SkipConnection) = (c.block,)

function mapchildren(f, c::SkipConnection)
  SkipConnection(f(c.block), c.connection)
end

function (skip::SkipConnection)(input)
  skip.connection(skip.block(input), input)
end

#####
##### Uilities
#####

using Knet: relu

flatten(x) = reshape(x, :, size(x)[end])

softmax(x) = Knet.softmax(x, dims=1)
