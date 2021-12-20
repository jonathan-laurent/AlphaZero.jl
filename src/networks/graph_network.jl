abstract type GraphNetwork <: AbstractNetwork end 

"""
    evaluate(::AbstractNetwork, state)

    (nn::AbstractNetwork)(state) = evaluate(nn, state)

Evaluate the neural network as an MCTS oracle on a single state.

Note, however, that evaluating state positions once at a time is slow and so you
may want to use a `BatchedOracle` along with an inference server that uses
[`evaluate_batch`](@ref).
"""
function evaluate(nn::GraphNetwork, state)
  gspec = game_spec(nn)
  actions_mask = GI.actions_mask(GI.init(gspec, state))
  x = GI.graph_state(gspec, state)
  a = Float32.(actions_mask)
  xnet, anet = to_singletons.(convert_input_tuple(nn, (x, a)))
  net_output = forward_normalized(nn, xnet, anet)
  p, v, _ = from_singletons.(convert_output_tuple(nn, net_output))
  return (p[actions_mask], v[1])
end


"""
    evaluate_batch(::AbstractNetwork, batch)

Evaluate the neural network as an MCTS oracle on a batch of states at once.

Take a list of states as input and return a list of `(P, V)` pairs as defined in the
MCTS oracle interface.
"""
function evaluate_batch(nn::GraphNetwork, batch)
  gspec = game_spec(nn)
  X = Flux.batch((GI.graph_state(gspec, b) for b in batch))
  A = Flux.batch((GI.actions_mask(GI.init(gspec, b)) for b in batch))
  Xnet, Anet = convert_input_tuple(nn, (X, Float32.(A)))
  P, V, _ = convert_output_tuple(nn, forward_normalized(nn, Xnet, Anet))
  return [(P[A[:,i],i], V[1,i]) for i in eachindex(batch)]
end
  