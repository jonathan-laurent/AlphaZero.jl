#####
##### Utilities to manage inference servers
#####

function fill_and_evaluate(net, batch; batch_size, fill)
  n = length(batch)
  @assert n > 0
  if !fill
    return Network.evaluate_batch(net, batch)
  else
    nmissing = batch_size - n
    @assert nmissing >= 0
    if nmissing == 0
      return Network.evaluate_batch(net, batch)
    else
      append!(batch, [batch[1] for _ in 1:nmissing])
      return Network.evaluate_batch(net, batch)[1:n]
    end
  end
end

# Start a server that processes inference requests for TWO networks
# Expected queries must have fields `state` and `netid`.
function inference_server(
    net1::AbstractNetwork,
    net2::AbstractNetwork,
    nworkers;
    fill_batches)
  return Batchifier.launch_server(nworkers) do batch
    n = length(batch)
    mask1 = findall(b -> b.netid == 1, batch)
    mask2 = findall(b -> b.netid == 2, batch)
    @assert length(mask1) + length(mask2) == n
    state(x) = x.state
    batch1 = state.(batch[mask1])
    batch2 = state.(batch[mask2])
    if isempty(mask2) # there are only queries from net1
      return fill_and_evaluate(net1, batch1; batch_size=n, fill=fill_batches)
    elseif isempty(mask1) # there are only queries from net2
      return fill_and_evaluate(net2, batch2; batch_size=n, fill=fill_batches)
    else # both networks sent queries
      res1 = fill_and_evaluate(net1, batch1; batch_size=n, fill=fill_batches)
      res2 = fill_and_evaluate(net2, batch2; batch_size=n, fill=fill_batches)
      @assert typeof(res1) == typeof(res2)
      res = similar(res1, n)
      res[mask1] = res1
      res[mask2] = res2
      return res
    end
  end
end

# Start an inference server for one agent
function inference_server(net::AbstractNetwork, n; fill_batches)
  return Batchifier.launch_server(n) do batch
    fill_and_evaluate(net, batch; batch_size=n, fill=fill_batches)
  end
end

ret_oracle(x) = () -> x
do_nothing!() = nothing
send_done!(reqc) = () -> Batchifier.client_done!(reqc)
zipthunk(f1, f2) = () -> (f1(), f2())

# Two neural network oracles
function batchify_oracles(os::Tuple{AbstractNetwork, AbstractNetwork}, fill, n)
  reqc = inference_server(os[1], os[2], n, fill_batches=fill)
  make1() = Batchifier.BatchedOracle(st -> (state=st, netid=1), reqc)
  make2() = Batchifier.BatchedOracle(st -> (state=st, netid=2), reqc)
  return zipthunk(make1, make2), send_done!(reqc)
end

function batchify_oracles(os::Tuple{<:Any, AbstractNetwork}, fill, n)
  reqc = inference_server(os[2], n, fill_batches=fill)
  make2() = Batchifier.BatchedOracle(reqc)
  return zipthunk(ret_oracle(os[1]), make2), send_done!(reqc)
end

function batchify_oracles(os::Tuple{AbstractNetwork, <:Any}, fill, n)
  reqc = inference_server(os[1], n, fill_batches=fill)
  make1() = Batchifier.BatchedOracle(reqc)
  return zipthunk(make1, ret_oracle(os[2])), send_done!(reqc)
end

function batchify_oracles(os::Tuple{<:Any, <:Any}, fill, n)
  return zipthunk(ret_oracle(os[1]), ret_oracle(os[2])), do_nothing!
end

function batchify_oracles(o::AbstractNetwork, fill, n)
  reqc = inference_server(o, n, fill_batches=fill)
  make() = Batchifier.BatchedOracle(reqc)
  return make, send_done!(reqc)
end

function batchify_oracles(o::Any, fill, n)
  return ret_oracle(o), do_nothing!
end

#####
##### Distributed simulator
#####

"""
    Simulator(make_player, oracles, measure)

A distributed simulator that encapsulates the details of running simulations
across multiple threads and multiple machines.

# Arguments

    - `make_oracles`: a function that takes no argument and returns
        the oracles used by the player, which can be either `nothing`,
        a single oracle or a pair of oracles.
    - `make_player`: a function that takes as an argument the `oracles` field
        above and builds a player from it.
    - `measure(trace, colors_flipped, player)`: the function that is used to
        take measurements after each game simulation.
"""
struct Simulator{MakePlayer, Oracles, Measure}
  make_player :: MakePlayer
  make_oracles :: Oracles
  measure :: Measure
end

"""
    record_trace

A measurement function to be passed to a [`Simulator`](@ref) that produces
named tuples with two fields: `trace::Trace` and `colors_flipped::Bool`.
"""
record_trace(t, cf, p) = (trace=t, colors_flipped=cf)

"""
    simulate(::Simulator, ::AbstractGameSpec; ::SimParams; <kwargs>)

Play a series of games using a given [`Simulator`](@ref).

# Keyword Arguments

  - `game_simulated` is called every time a game simulation is completed with no arguments

# Return

Return a vector of objects computed by `simulator.measure`.
"""
function simulate(
    simulator::Simulator,
    gspec::AbstractGameSpec,
    p::SimParams;
    game_simulated)

  oracles = simulator.make_oracles()
  spawn_oracles, done =
    batchify_oracles(oracles, p.fill_batches, p.num_workers)
  return Util.mapreduce(1:p.num_games, p.num_workers, vcat, []) do
    oracles = spawn_oracles()
    player = simulator.make_player(oracles)
    worker_sim_id = 0
    # For each worker
    function simulate_game(sim_id)
      worker_sim_id += 1
      # Switch players' colors if necessary: "_pf" stands for "possibly flipped"
      if isa(player, TwoPlayers) && p.alternate_colors
        colors_flipped = sim_id % 2 == 1
        player_pf = colors_flipped ? flipped_colors(player) : player
      else
        colors_flipped = false
        player_pf = player
      end
      # Play the game and generate a report
      trace = play_game(gspec, player_pf, flip_probability=p.flip_probability)
      report = simulator.measure(trace, colors_flipped, player)
      # Reset the player periodically
      if !isnothing(p.reset_every) && worker_sim_id % p.reset_every == 0
        reset_player!(player)
      end
      # Signal that a game has been simulated
      game_simulated()
      return report
    end
    return (process=simulate_game, terminate=done)
  end
end

"""
    simulate_distributed(::Simulator, ::AbstractGameSpec, ::SimParams; <kwargs>)

Identical to [`simulate`](@ref) but splits the work across all available workers.
"""
function simulate_distributed(
    simulator::Simulator,
    gspec::AbstractGameSpec,
    p::SimParams;
    game_simulated)

  # Spawning a task to keep count of completed simulations
  chan = Distributed.RemoteChannel(()->Channel{Nothing}(1))
  Threads.@spawn begin
    for i in 1:p.num_games
      take!(chan)
      game_simulated()
    end
  end
  remote_game_simulated() = put!(chan, nothing)
  # Distributing the simulations across workers
  num_each, rem = divrem(p.num_games, Distributed.nworkers())
  @assert num_each >= 1
  workers = Distributed.workers()
  tasks = map(workers) do w
    Distributed.@spawnat w begin
      Util.@printing_errors begin
        simulate(
          simulator,
          gspec,
          SimParams(p; num_games=(w == workers[1] ? num_each + rem : num_each)),
          game_simulated=remote_game_simulated)
        end
    end
  end
  results = fetch.(tasks)
  # If one of the worker raised an exception, we print it
  for r in results
    if isa(r, Distributed.RemoteException)
      showerror(stderr, r, catch_backtrace())
    end
  end
  return reduce(vcat, results)
end

function compute_redundancy(states)
  unique = Set(states)
  return 1. - length(unique) / length(states)
end

# samples is a vector of named tuples with fields `trace` and `colors_flipped`
function rewards_and_redundancy(samples; gamma)
  rewards = map(samples) do s
    wr = total_reward(s.trace, gamma)
    return s.colors_flipped ? -wr : wr
  end
  states = [st for s in samples for st in s.trace.states]
  redundancy = compute_redundancy(states)
  return rewards, redundancy
end