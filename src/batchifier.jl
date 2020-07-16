module Batchifier

import ..MCTS, ..Util

export BatchedOracle

"""
    launch_server(f, num_workers)

Launch an inference requests server.

# Arguments

  - `f` is the function to be evaluated (e.g. a neural network).
    It takes a batch of inputs and returns a batch of results
    of similar size.
  - `num_workers` is the number of workers that are expected to
    query the server. It also corresponds to the batch size that is used to
    evaluate `f`.

# How to use

This function returns a channel along which workers can send queries.
A query can be either:

    - `:none` when a worker is done and about to terminate
    - a named tuple with fields `query` (the input to be given to `f`) and
      `answer_channel` (the channel the sever must use to return its answer).

The server stops automatically after all workers send `:none`.
"""
function launch_server(f, num_workers)
  channel = Channel(num_workers)
  Threads.@spawn Util.@printing_errors begin
    num_active = num_workers
    pending = []
    while num_active > 0
      req = take!(channel)
      if req == :done
        num_active -= 1
      else
        push!(pending, req)
      end
      @assert length(pending) <= num_active
      if length(pending) == num_active && num_active > 0
        batch = [p.query for p in pending]
        results = f(batch)
        for i in eachindex(pending)
          put!(pending[i].answer_channel, results[i])
        end
        empty!(pending)
      end
    end
  end
  return channel
end

client_done!(reqc) = put!(reqc, :done)

struct BatchedOracle{Game, F} <: MCTS.Oracle{Game}
  make_query :: F # turn state into a query (this is usually the identity)
  reqchan :: Channel
  anschan :: Channel
  function BatchedOracle{G}(f, reqchan) where G
    return new{G, typeof(f)}(f, reqchan, Channel(1))
  end

end

function BatchedOracle{G}(reqchan) where G
  return BatchedOracle{G}(x -> x, reqchan)
end

function (oracle::BatchedOracle)(state)
  query = oracle.make_query(state)
  put!(oracle.reqchan, (query=query, answer_channel=oracle.anschan))
  answer = take!(oracle.anschan)
  return answer
end

end
