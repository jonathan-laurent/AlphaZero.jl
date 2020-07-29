using Base.Threads

"""
    launch_inference_server(f, num_workers)

Launch an inference server.

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
function launch_inference_server(f, num_workers)
  channel = Channel(num_workers)
  Threads.@spawn begin
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
      if length(pending) == num_active
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

# Imagine I am evaluating a big neural network on a batch of inputs instead...
function compute_squares(batch)
  return [x^2 for x in batch]
end

# Compute the square of every integer in 1:n using num_workers parallel workers.
# Return the answer in a dictionary.
function generate_squares(; n, num_workers)
  d = n ÷ num_workers
  reqc = launch_inference_server(compute_squares, num_workers)
  results = []
  for i in 1:num_workers
    res = Threads.@spawn begin
      answc = Channel(1)
      squares = Dict()
      xs = [(i-1)*d+j for j in 1:d]
      for x in xs
        put!(reqc, (query=x, answer_channel=answc))
        y = take!(answc)
        squares[x] = y
      end
      put!(reqc, :done)
      return squares
    end
    push!(results, res)
  end
  results = fetch.(results)
  return reduce(merge, results)
end

squares = generate_squares(n=1_000_000, num_workers=10)
for (x, y) in squares
  @assert y == x^2
end
