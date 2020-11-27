"""
    LSTMNetHP

Hyperparameters for the LSTMnet architecture.

| Parameter                     | Description                                  |
|:------------------------------|:---------------------------------------------|
| `width :: Int`                | Number of neurons on each lstm layer         |
| `depth_common :: Int`         | Number of lstm layers in the trunk           |
| `depth_phead = 1`             | Number of hidden layers in the actions head  |
| `depth_vhead = 1`             | Number of hidden layers in the value  head   |
| `use_batch_norm = false`      | Use batch normalization between each layer   |
| `batch_norm_momentum = 0.6f0` | Momentum of batch norm statistics updates    |
"""
@kwdef struct LSTMNetHP
  width :: Int
  depth_common :: Int
  depth_phead :: Int = 1
  depth_vhead :: Int = 1
  use_batch_norm :: Bool = false
  batch_norm_momentum :: Float32 = 0.6f0
end

"""
    LSTMNet <: TwoHeadNetwork

A LSTM two-headed architecture with only lstm layers.
"""
mutable struct LSTMNet <: TwoHeadNetwork
  gspec
  hyper
  common
  vhead
  phead
end

function LSTMNet(gspec::AbstractGameSpec, hyper::LSTMNetHP)
  bnmom = hyper.batch_norm_momentum
  actions=GI.actions(gspec)
  actions_tuples=[tuple(actions[i,:]...) for i in 1:size(actions)[1]]
  last_timestep(x)=x[:,end]
  # makehot(x)=collect(partition(batchseq(chunk(map(action -> onehot(action, unique(x)), x),1)),5))
  # makehot(x)=collect(partition(batchseq(chunk(map(action -> onehot(action, actions_tuples), [tuple(x[i,:]...) for i in 1:size(x)[1]]),1)),size(x)[1]))[1]
  makehot(x)=reshape(Int.(Matrix(hcat((map(action -> onehot(action, actions_tuples), [tuple(x[i,:]...) for i in 1:size(x)[1]]))...))),length(actions_tuples),size(x)[1])
  # makehotter(a)=vcat([Flux.onehotbatch(a[:,i],unique(a)) for i in 1:length(a[1,:])]...)
  # make_hot(x)=hcat((map(action -> onehot(action, unique(x)), x))...)
  # extract_last(x)=vcat([(x[i][:,end]) for i in 1:length(x)]...)
  # lastly(j)=vcat(map(j->j[:,end],j)...)
  # makehotter(x)=vcat(collect.(partition.(batchseq.(chunk.([map(action -> onehot(action, unique(j)), j) for j in [x[:,i] for i in 1:length(x[1,:])]],1)),5))...)
  function make_lstm(indim, outdim)
    if hyper.use_batch_norm
      Chain(makehot,# add a function here to make timeseries a onehot timseries, output of which is input to LSTM
        LSTM(indim, outdim),#the indim is the number of assets(features), the state is the timeseries
        BatchNorm(outdim, relu, momentum=bnmom))
    else
      LSTM(indim, outdim)
    end
  end
  function make_dense(indim, outdim)
    if hyper.use_batch_norm
      Chain(
        Dense(indim, outdim),#the indim is the number of assets(features), the state is the timeseries
        BatchNorm(outdim, relu, momentum=bnmom))
    else
      Dense(indim, outdim)
    end
  end
  indim = prod(GI.state_dim(gspec)) #in the case of a buy-sell state_dim=(Assests,HISTORY+FUTURE), the indim should be a one hot timeseries i.e (2,length(timeseries)) 2 for buy-sell actions
  outdim = GI.num_actions(gspec)
  hsize = hyper.width
  lstmlayers(depth)=[make_lstm(hsize, hsize) for i in 1:depth]
  hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth]
  common = Chain(
    make_lstm(outdim, hsize),
    lstmlayers(hyper.depth_common)...,
    x -> last_timestep(x),
    hlayers(hyper.depth_common)...)
  vhead = Chain(
    hlayers(hyper.depth_vhead)...,
    Dense(hsize, 1, tanh))
  phead = Chain(
    hlayers(hyper.depth_phead)...,
    Dense(hsize, outdim),
    softmax)
  LSTMNet(gspec, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{LSTMNet}) = LSTMNetHP
