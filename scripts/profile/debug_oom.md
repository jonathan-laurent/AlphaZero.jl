I have been encountering what I interpret as being OOM errors when training fairly small resnets.

The problem happens with both Flux and Knet and with both types of memory pools (binned or split).

I am providing replication instructions below. Replicating the errors takes about a minute.

Here are different errors I have observed (see backtraces at the end of the message):

- `CUDNN_STATUS_EXECUTION_FAILED (code 8)`: when running with Flux
- `CUDNN_STATUS_INTERNAL_ERROR (code 4)`: when running with Knet (with probability ~0.5)
- Surprisingly, when using Knet, I am sometimes getting the following error instead (with probability ~0.5): `MethodError: no method matching LinearIndices(::Knet.KnetArrays.KnetVector{Float32})`

### Replication instructions

```sh
export JULIA_CUDA_MEMORY_POOL=binned #split
export ALPHAZERO_DEFAULT_DL_FRAMEWORK=FLUX #KNET

git clone --branch cuda-oom https://github.com/jonathan-laurent/AlphaZero.jl.git
cd AlphaZero.jl
julia --project -e "import Pkg; Pkg.instantiate()"

NUM_FILTERS=64 julia --project scripts/profile/debug_oom.jl
```

### Configuration

**Julia:**
```
Julia Version 1.6.0-rc1
Commit a58bdd9010 (2021-02-06 15:49 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, skylake)
Environment:
  JULIA_CUDA_MEMORY_POOL = split
  JULIA_NUM_THREADS = 6
```

**CUDA:**
Package version: v2.6.1

```
CUDA toolkit 11.1.1, artifact installation
CUDA driver 11.1.0
NVIDIA driver 455.23.5

Libraries: 
- CUBLAS: 11.2.1
- CURAND: 10.2.2
- CUFFT: 10.3.0
- CUSOLVER: 11.0.1
- CUSPARSE: 11.3.0
- CUPTI: 14.0.0
- NVML: 11.0.0+455.23.5
- CUDNN: 8.0.4 (for CUDA 11.1.0)
- CUTENSOR: 1.2.1 (for CUDA 11.1.0)

Toolchain:
- Julia: 1.6.0-rc1
- LLVM: 11.0.1
- PTX ISA support: 3.2, 4.0, 4.1, 4.2, 4.3, 5.0, 6.0, 6.1, 6.3, 6.4, 6.5, 7.0
- Device support: sm_35, sm_37, sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72, sm_75, sm_80

Environment:
- JULIA_CUDA_MEMORY_POOL: split

1 device:
  0: GeForce RTX 2070 (sm_75, 7.462 GiB / 7.793 GiB available)
```

### Backtraces

**When using Flux:**

```
ERROR: LoadError: CUDNNError: CUDNN_STATUS_EXECUTION_FAILED (code 8)
Stacktrace:
  [1] throw_api_error(res::CUDA.CUDNN.cudnnStatus_t)
    @ CUDA.CUDNN ~/.julia/packages/CUDA/Zmd60/lib/cudnn/error.jl:19
  [2] macro expansion
    @ ~/.julia/packages/CUDA/Zmd60/lib/cudnn/error.jl:30 [inlined]
  [3] cudnnBatchNormalizationForwardTraining(handle::Ptr{Nothing}, mode::CUDA.CUDNN.cudnnBatchNormMode_t, alpha::Base.RefValue{Float32}, beta::Base.RefValue{Float32}, xDesc::CUDA.CUDNN.TensorDesc, x::CUDA.CuArray{Float32, 4}, yDesc::CUDA.CUDNN.TensorDesc, y::CUDA.CuArray{Float32, 4}, bnScaleBiasMeanVarDesc::CUDA.CUDNN.TensorDesc, bnScale::CUDA.CuArray{Float32, 1}, bnBias::CUDA.CuArray{Float32, 1}, exponentialAverageFactor::Float32, resultRunningMean::CUDA.CuArray{Float32, 1}, resultRunningVariance::CUDA.CuArray{Float32, 1}, epsilon::Float32, resultSaveMean::CUDA.CuPtr{Nothing}, resultSaveInvVariance::CUDA.CuPtr{Nothing})
    @ CUDA.CUDNN ~/.julia/packages/CUDA/Zmd60/lib/utils/call.jl:26
  [4] cudnnBNForward!(y::CUDA.CuArray{Float32, 4}, g::CUDA.CuArray{Float32, 1}, b::CUDA.CuArray{Float32, 1}, x::CUDA.CuArray{Float32, 4}, running_mean::CUDA.CuArray{Float32, 1}, running_var::CUDA.CuArray{Float32, 1}, momentum::Float32; cache::Nothing, alpha::Int64, beta::Int64, eps::Float32, training::Bool)
    @ CUDA.CUDNN ~/.julia/packages/CUDA/Zmd60/lib/cudnn/batchnorm.jl:53
  [5] #batchnorm#42
    @ ~/.julia/packages/CUDA/Zmd60/lib/cudnn/batchnorm.jl:25 [inlined]
  [6] #adjoint#17
    @ ~/.julia/packages/Flux/goUGu/src/cuda/cudnn.jl:6 [inlined]
  [7] _pullback(__context__::Zygote.Context, #unused#::CUDA.CUDNN.var"#batchnorm##kw", kw::NamedTuple{(:cache, :alpha, :beta, :eps, :training), Tuple{Nothing, Int64, Int64, Float32, Bool}}, 267::typeof(CUDA.CUDNN.batchnorm), g::CUDA.CuArray{Float32, 1}, b::CUDA.CuArray{Float32, 1}, x::CUDA.CuArray{Float32, 4}, running_mean::CUDA.CuArray{Float32, 1}, running_var::CUDA.CuArray{Float32, 1}, momentum::Float32)
    @ Flux.CUDAint ~/.julia/packages/ZygoteRules/OjfTt/src/adjoint.jl:63
  [8] _pullback
    @ ~/.julia/packages/Flux/goUGu/src/cuda/cudnn.jl:3 [inlined]
  [9] _pullback(::Zygote.Context, ::Flux.BatchNorm{typeof(NNlib.relu), CUDA.CuArray{Float32, 1}, CUDA.CuArray{Float32, 1}, Float32}, ::CUDA.CuArray{Float32, 4}, ::Nothing)
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface2.jl:0
 [10] _pullback
    @ ~/.julia/packages/Flux/goUGu/src/cuda/cudnn.jl:3 [inlined]
 [11] _pullback(ctx::Zygote.Context, f::Flux.BatchNorm{typeof(NNlib.relu), CUDA.CuArray{Float32, 1}, CUDA.CuArray{Float32, 1}, Float32}, args::CUDA.CuArray{Float32, 4})
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface2.jl:0
 [12] _pullback
    @ ~/.julia/packages/Flux/goUGu/src/layers/basic.jl:36 [inlined]
 [13] _pullback(::Zygote.Context, ::typeof(Flux.applychain), ::Tuple{Flux.BatchNorm{typeof(NNlib.relu), CUDA.CuArray{Float32, 1}, CUDA.CuArray{Float32, 1}, Float32}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}}, ::CUDA.CuArray{Float32, 4})
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface2.jl:0
 [14] _pullback
    @ ~/.julia/packages/Flux/goUGu/src/layers/basic.jl:36 [inlined]
 [15] _pullback(::Zygote.Context, ::typeof(Flux.applychain), ::Tuple{Flux.Conv{2, 2, typeof(identity), CUDA.CuArray{Float32, 4}, CUDA.CuArray{Float32, 1}}, Flux.BatchNorm{typeof(NNlib.relu), CUDA.CuArray{Float32, 1}, CUDA.CuArray{Float32, 1}, Float32}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}}, ::CUDA.CuArray{Float32, 4})
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface2.jl:0
 [16] _pullback
    @ ~/.julia/packages/Flux/goUGu/src/layers/basic.jl:38 [inlined]
 [17] _pullback(ctx::Zygote.Context, f::Flux.Chain{Tuple{Flux.Conv{2, 2, typeof(identity), CUDA.CuArray{Float32, 4}, CUDA.CuArray{Float32, 1}}, Flux.BatchNorm{typeof(NNlib.relu), CUDA.CuArray{Float32, 1}, CUDA.CuArray{Float32, 1}, Float32}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}, Flux.Chain{Tuple{Flux.SkipConnection, AlphaZero.FluxLib.var"#17#18"}}}}, args::CUDA.CuArray{Float32, 4})
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface2.jl:0
 [18] _pullback
    @ ~/AlphaZero.jl/src/networks/flux.jl:160 [inlined]
 [19] _pullback(::Zygote.Context, ::typeof(AlphaZero.Network.forward), ::ResNet, ::CUDA.CuArray{Float32, 4})
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface2.jl:0
 [20] _pullback
    @ ~/AlphaZero.jl/src/networks/network.jl:260 [inlined]
 [21] _pullback(::Zygote.Context, ::typeof(AlphaZero.Network.forward_normalized), ::ResNet, ::CUDA.CuArray{Float32, 4}, ::CUDA.CuArray{Float32, 2})
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface2.jl:0
 [22] _pullback
    @ ~/AlphaZero.jl/src/learning.jl:70 [inlined]
 [23] _pullback(::Zygote.Context, ::typeof(AlphaZero.losses), ::ResNet, ::LearningParams, ::Float32, ::Float32, ::Tuple{CUDA.CuArray{Float32, 2}, CUDA.CuArray{Float32, 4}, CUDA.CuArray{Float32, 2}, CUDA.CuArray{Float32, 2}, CUDA.CuArray{Float32, 2}})
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface2.jl:0
 [24] _pullback
    @ ~/AlphaZero.jl/src/learning.jl:122 [inlined]
 [25] _pullback(::Zygote.Context, ::AlphaZero.var"#L#110"{AlphaZero.Trainer}, ::CUDA.CuArray{Float32, 2}, ::CUDA.CuArray{Float32, 4}, ::CUDA.CuArray{Float32, 2}, ::CUDA.CuArray{Float32, 2}, ::CUDA.CuArray{Float32, 2})
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface2.jl:0
 [26] adjoint
    @ ~/.julia/packages/Zygote/KpME9/src/lib/lib.jl:188 [inlined]
 [27] _pullback
    @ ~/.julia/packages/ZygoteRules/OjfTt/src/adjoint.jl:57 [inlined]
 [28] _pullback
    @ ~/AlphaZero.jl/src/networks/flux.jl:82 [inlined]
 [29] _pullback(::Zygote.Context, ::AlphaZero.FluxLib.var"#1#2"{AlphaZero.var"#L#110"{AlphaZero.Trainer}, Tuple{CUDA.CuArray{Float32, 2}, CUDA.CuArray{Float32, 4}, CUDA.CuArray{Float32, 2}, CUDA.CuArray{Float32, 2}, CUDA.CuArray{Float32, 2}}})
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface2.jl:0
 [30] pullback(f::Function, ps::Zygote.Params)
    @ Zygote ~/.julia/packages/Zygote/KpME9/src/compiler/interface.jl:167
 [31] lossgrads(f::Function, args::Zygote.Params)
    @ AlphaZero.FluxLib ~/AlphaZero.jl/src/networks/flux.jl:72
 [32] train!(callback::AlphaZero.var"#109#111"{Vector{Float32}}, nn::ResNet, opt::Adam, loss::Function, data::Base.Iterators.Take{Base.Iterators.Stateful{Base.Iterators.Flatten{Base.Generator{Base.Iterators.Repeated{Nothing}, AlphaZero.Util.var"#12#13"{AlphaZero.var"#106#108"{ResNet}, Tuple{Matrix{Float32}, Array{Float32, 4}, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}}, Int64, Bool}}}, Tuple{NTuple{5, Any}, Tuple{Nothing, Base.Generator{Vector{Tuple{Matrix{Float32}, Array{Float32, 4}, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}}}, AlphaZero.Util.var"#9#11"{AlphaZero.var"#106#108"{ResNet}}}, Int64}}}}, n::Int64)
    @ AlphaZero.FluxLib ~/AlphaZero.jl/src/networks/flux.jl:81
 [33] batch_updates!(tr::AlphaZero.Trainer, n::Int64)
    @ AlphaZero ~/AlphaZero.jl/src/learning.jl:125
 [34] macro expansion
    @ ./timing.jl:356 [inlined]
 [35] learning_step!(env::Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}, handler::Session{Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}})
    @ AlphaZero ~/AlphaZero.jl/src/training.jl:223
 [36] top-level scope
    @ ~/AlphaZero.jl/scripts/profile/debug_oom.jl:39
in expression starting at /home/jonathan/AlphaZero.jl/scripts/profile/debug_oom.jl:39
```

**When using Knet (1/2):**

```
ERROR: CUDNNError: CUDNN_STATUS_INTERNAL_ERROR (code 4)
Stacktrace:
  [1] throw_api_error(res::CUDA.CUDNN.cudnnStatus_t)
    @ CUDA.CUDNN ~/.julia/packages/CUDA/Zmd60/lib/cudnn/error.jl:19
  [2] macro expansion
    @ ~/.julia/packages/CUDA/Zmd60/lib/cudnn/error.jl:30 [inlined]
  [3] cudnnFindConvolutionBackwardFilterAlgorithmEx(handle::Ptr{Nothing}, xDesc::Knet.Ops20_gpu.TD, x::Knet.KnetArrays.KnetArray{Float32, 4}, dyDesc::Knet.Ops20_gpu.TD, y::Knet.KnetArrays.KnetArray{Float32, 4}, convDesc::Knet.Ops20_gpu.CD, dwDesc::Knet.Ops20_gpu.FD, dw::Knet.KnetArrays.KnetArray{Float32, 4}, requestedAlgoCount::Int64, returnedAlgoCount::Vector{Int32}, perfResults::Vector{CUDA.CUDNN.cudnnConvolutionBwdFilterAlgoPerf_t}, workSpace::Knet.KnetArrays.KnetVector{Float32}, workSpaceSizeInBytes::Int64)
    @ CUDA.CUDNN ~/.julia/packages/CUDA/Zmd60/lib/utils/call.jl:26
  [4] conv4w_algo(w::Knet.KnetArrays.KnetArray{Float32, 4}, x::Knet.KnetArrays.KnetArray{Float32, 4}, dy::Knet.KnetArrays.KnetArray{Float32, 4}, dw::Knet.KnetArrays.KnetArray{Float32, 4}; handle::Ptr{Nothing}, o::Base.Iterators.Pairs{Symbol, Int64, Tuple{Symbol}, NamedTuple{(:padding,), Tuple{Int64}}})
    @ Knet.Ops20_gpu ~/.julia/packages/Knet/C0PoK/src/ops20_gpu/conv.jl:194
  [5] conv4w(w::Knet.KnetArrays.KnetArray{Float32, 4}, x::Knet.KnetArrays.KnetArray{Float32, 4}, dy::Knet.KnetArrays.KnetArray{Float32, 4}; handle::Ptr{Nothing}, alpha::Int64, o::Base.Iterators.Pairs{Symbol, Int64, Tuple{Symbol}, NamedTuple{(:padding,), Tuple{Int64}}})
    @ Knet.Ops20_gpu ~/.julia/packages/Knet/C0PoK/src/ops20_gpu/conv.jl:27
  [6] forw(::Function, ::AutoGrad.Param{Knet.KnetArrays.KnetArray{Float32, 4}}, ::Vararg{Any, N} where N; kwargs::Base.Iterators.Pairs{Symbol, Int64, Tuple{Symbol}, NamedTuple{(:padding,), Tuple{Int64}}})
    @ AutoGrad ~/.julia/packages/AutoGrad/TTpeo/src/core.jl:66
  [7] #conv4w#47
    @ ./none:0 [inlined]
  [8] #back#23
    @ ./none:0 [inlined]
  [9] differentiate(::Function; o::Base.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
    @ AutoGrad ~/.julia/packages/AutoGrad/TTpeo/src/core.jl:165
 [10] differentiate
    @ ~/.julia/packages/AutoGrad/TTpeo/src/core.jl:135 [inlined]
 [11] iterate
    @ ~/.julia/packages/Knet/C0PoK/src/train20/train.jl:26 [inlined]
 [12] iterate
    @ ./iterators.jl:159 [inlined]
 [13] iterate
    @ ./iterators.jl:158 [inlined]
 [14] train!(callback::AlphaZero.var"#109#111"{Vector{Float32}}, nn::ResNet, opt::Adam, loss::Function, data::Base.Iterators.Take{Base.Iterators.Stateful{Base.Iterators.Flatten{Base.Generator{Base.Iterators.Repeated{Nothing}, AlphaZero.Util.var"#12#13"{AlphaZero.var"#106#108"{ResNet}, Tuple{Matrix{Float32}, Array{Float32, 4}, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}}, Int64, Bool}}}, Tuple{Tuple{Union{Matrix{Float32}, Knet.KnetArrays.KnetMatrix{Float32}}, Union{Array{Float32, 4}, Knet.KnetArrays.KnetArray{Float32, 4}}, Union{Matrix{Float32}, Knet.KnetArrays.KnetMatrix{Float32}}, Union{Matrix{Float32}, Knet.KnetArrays.KnetMatrix{Float32}}, Union{Matrix{Float32}, Knet.KnetArrays.KnetMatrix{Float32}}}, Tuple{Nothing, Base.Generator{Vector{Tuple{Matrix{Float32}, Array{Float32, 4}, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}}}, AlphaZero.Util.var"#9#11"{AlphaZero.var"#106#108"{ResNet}}}, Int64}}}}, n::Int64)
    @ AlphaZero.KnetLib ~/AlphaZero.jl/src/networks/knet.jl:120
 [15] batch_updates!(tr::AlphaZero.Trainer, n::Int64)
    @ AlphaZero ~/AlphaZero.jl/src/learning.jl:125
 [16] macro expansion
    @ ./timing.jl:356 [inlined]
 [17] learning_step!(env::Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}, handler::Session{Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}})
    @ AlphaZero ~/AlphaZero.jl/src/training.jl:223
 [18] macro expansion
    @ ./timing.jl:356 [inlined]
 [19] macro expansion
    @ ~/AlphaZero.jl/src/report.jl:267 [inlined]
 [20] train!(env::Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}, handler::Session{Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}})
    @ AlphaZero ~/AlphaZero.jl/src/training.jl:326
 [21] resume!(session::Session{Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}})
    @ AlphaZero.UserInterface ~/AlphaZero.jl/src/ui/session.jl:316
 [22] train(e::Experiment; args::Base.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
    @ AlphaZero.Scripts ~/AlphaZero.jl/src/scripts/scripts.jl:26
 [23] train
    @ ~/AlphaZero.jl/src/scripts/scripts.jl:26 [inlined]
 [24] #train#15
    @ ~/AlphaZero.jl/src/scripts/scripts.jl:28 [inlined]
 [25] train(s::String)
    @ AlphaZero.Scripts ~/AlphaZero.jl/src/scripts/scripts.jl:28
 [26] top-level scope
    @ none:1
```

**When using Knet (2/2):**

```
ERROR: MethodError: no method matching LinearIndices(::Knet.KnetArrays.KnetVector{Float32})
Closest candidates are:
  LinearIndices(::Tuple{}) at indices.jl:451
  LinearIndices(::R) where {N, R<:Tuple{Vararg{AbstractUnitRange{Int64}, N}}} at indices.jl:448
  LinearIndices(::Tuple{Vararg{AbstractUnitRange{var"#s77"} where var"#s77"<:Integer, N}}) where N at indices.jl:452
  ...
Stacktrace:
  [1] compute_linindex
    @ ./subarray.jl:395 [inlined]
  [2] compute_offset1
    @ ./subarray.jl:387 [inlined]
  [3] compute_offset1
    @ ./subarray.jl:385 [inlined]
  [4] SubArray
    @ ./subarray.jl:38 [inlined]
  [5] SubArray
    @ ~/.julia/packages/Knet/C0PoK/src/knetarrays/dotview.jl:37 [inlined]
  [6] unsafe_view
    @ ~/.julia/packages/Knet/C0PoK/src/knetarrays/dotview.jl:21 [inlined]
  [7] view
    @ ~/.julia/packages/Knet/C0PoK/src/knetarrays/dotview.jl:16 [inlined]
  [8] dotview(A::Knet.KnetArrays.KnetMatrix{Float32}, I::Function)
    @ Knet.KnetArrays ~/.julia/packages/Knet/C0PoK/src/knetarrays/dotview.jl:10
  [9] fill!(a::Knet.KnetArrays.KnetMatrix{Float32}, x::Float32)
    @ Knet.KnetArrays ~/.julia/packages/Knet/C0PoK/src/knetarrays/abstractarray.jl:13
 [10] sum(x::Knet.KnetArrays.KnetMatrix{Float32}; dims::Vector{Any})
    @ Knet.KnetArrays ~/.julia/packages/Knet/C0PoK/src/knetarrays/reduction.jl:41
 [11] unbroadcast(x::AutoGrad.Param{Knet.KnetArrays.KnetVector{Float32}}, dx::Knet.KnetArrays.KnetMatrix{Float32})
    @ AutoGrad ~/.julia/packages/AutoGrad/TTpeo/src/unbroadcast.jl:24
 [12] back(#unused#::typeof(Base.Broadcast.broadcasted), #unused#::Type{AutoGrad.Arg{3}}, dy::Knet.KnetArrays.KnetMatrix{Float32}, 269::AutoGrad.Result{Knet.KnetArrays.KnetMatrix{Float32}}, #unused#::typeof(+), x1::AutoGrad.Result{Knet.KnetArrays.KnetMatrix{Float32}}, x2::AutoGrad.Param{Knet.KnetArrays.KnetVector{Float32}})
    @ AutoGrad ./none:0
 [13] differentiate(::Function; o::Base.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
    @ AutoGrad ~/.julia/packages/AutoGrad/TTpeo/src/core.jl:165
 [14] differentiate
    @ ~/.julia/packages/AutoGrad/TTpeo/src/core.jl:135 [inlined]
 [15] iterate
    @ ~/.julia/packages/Knet/C0PoK/src/train20/train.jl:26 [inlined]
 [16] iterate
    @ ./iterators.jl:159 [inlined]
 [17] iterate
    @ ./iterators.jl:158 [inlined]
 [18] train!(callback::AlphaZero.var"#109#111"{Vector{Float32}}, nn::ResNet, opt::Adam, loss::Function, data::Base.Iterators.Take{Base.Iterators.Stateful{Base.Iterators.Flatten{Base.Generator{Base.Iterators.Repeated{Nothing}, AlphaZero.Util.var"#12#13"{AlphaZero.var"#106#108"{ResNet}, Tuple{Matrix{Float32}, Array{Float32, 4}, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}}, Int64, Bool}}}, Tuple{Tuple{Union{Matrix{Float32}, Knet.KnetArrays.KnetMatrix{Float32}}, Union{Array{Float32, 4}, Knet.KnetArrays.KnetArray{Float32, 4}}, Union{Matrix{Float32}, Knet.KnetArrays.KnetMatrix{Float32}}, Union{Matrix{Float32}, Knet.KnetArrays.KnetMatrix{Float32}}, Union{Matrix{Float32}, Knet.KnetArrays.KnetMatrix{Float32}}}, Tuple{Nothing, Base.Generator{Vector{Tuple{Matrix{Float32}, Array{Float32, 4}, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}}}, AlphaZero.Util.var"#9#11"{AlphaZero.var"#106#108"{ResNet}}}, Int64}}}}, n::Int64)
    @ AlphaZero.KnetLib ~/AlphaZero.jl/src/networks/knet.jl:119
 [19] batch_updates!(tr::AlphaZero.Trainer, n::Int64)
    @ AlphaZero ~/AlphaZero.jl/src/learning.jl:125
 [20] macro expansion
    @ ./timing.jl:356 [inlined]
 [21] learning_step!(env::Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}, handler::Session{Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}})
    @ AlphaZero ~/AlphaZero.jl/src/training.jl:223
 [22] macro expansion
    @ ./timing.jl:356 [inlined]
 [23] macro expansion
    @ ~/AlphaZero.jl/src/report.jl:267 [inlined]
 [24] train!(env::Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}, handler::Session{Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}})
    @ AlphaZero ~/AlphaZero.jl/src/training.jl:326
 [25] resume!(session::Session{Env{AlphaZero.Examples.ConnectFour.GameSpec, ResNet, NamedTuple{(:board, :curplayer), Tuple{StaticArrays.SMatrix{7, 6, UInt8, 42}, UInt8}}}})
    @ AlphaZero.UserInterface ~/AlphaZero.jl/src/ui/session.jl:316
 [26] train(e::Experiment; args::Base.Iterators.Pairs{Union{}, Union{}, Tuple{}, NamedTuple{(), Tuple{}}})
    @ AlphaZero.Scripts ~/AlphaZero.jl/src/scripts/scripts.jl:26
 [27] train
    @ ~/AlphaZero.jl/src/scripts/scripts.jl:26 [inlined]
 [28] #train#15
    @ ~/AlphaZero.jl/src/scripts/scripts.jl:28 [inlined]
 [29] train(s::String)
    @ AlphaZero.Scripts ~/AlphaZero.jl/src/scripts/scripts.jl:28
 [30] top-level scope
    @ none:1
```