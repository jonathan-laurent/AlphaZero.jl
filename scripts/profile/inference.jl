#####
##### Profiling batched inference
#####

using AlphaZero
using Setfield
import CUDA

using Plots

# using Revise; Revise.includet("scripts/profile/inference.jl")
# profile_inference(on_gpu=true, batch_size=128, num_filters=128)
# With 128 filters: 27μs per sample
# With  64 filters: 17μs per sample

"""
Estimate inference time in μs/sample.
"""
function profile_inference(
    exp::Experiment = Examples.experiments["connect-four"];
    on_gpu=true,
    nrep=100,
    batch_size=128,
    num_filters=64)

  params = exp.netparams
  params = @set params.num_filters = num_filters
  net = exp.mknet(exp.gspec, params)
  net = Network.copy(net; on_gpu, test_mode=true)
  state = GI.current_state(GI.init(exp.gspec))
  batch = [state for _ in 1:batch_size]
  Network.evaluate_batch(net, batch) # To compile everything
  GC.gc(true)
  info = CUDA.@timed for _ in 1:nrep Network.evaluate_batch(net, batch) end
  return info.time / nrep / batch_size * 1_000_000
end

"""
Plot the inference time speedup (in μs/sample) as a function of batch-size.
"""
function plot_inference_speedup(
  file::String,
  exp::Experiment = Examples.experiments["connect-four"];
  on_gpu=true,
  num_filters=64)

  N = 8
  batch_sizes = [2^i for i in 0:N]
  ts = [
    profile_inference(exp;
      nrep=10,
      num_filters, batch_size, on_gpu)
    for batch_size in batch_sizes ]

  xticks = (0:N, map(string, batch_sizes))
  plot(0:N, ts[1] ./ ts,
    title="Inference Speedup (baseline: $(Int(ceil(ts[1]))) μs)",
    ylabel="Speedup",
    xlabel="Batch size",
    ylims=(0, Inf),
    legend=:none,
    xticks=xticks)
  savefig(file)
end

function all_plots()
  plot_inference_speedup("inference-gpu-64.png",  on_gpu=true,  num_filters=64)
  plot_inference_speedup("inference-cpu-64.png",  on_gpu=false, num_filters=64)
  plot_inference_speedup("inference-gpu-128.png", on_gpu=true,  num_filters=128)
  plot_inference_speedup("inference-cpu-128.png", on_gpu=false, num_filters=128)
end