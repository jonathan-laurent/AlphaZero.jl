module Devices

using CUDA
import Base.zeros, Base.ones, Base.fill

export Device, GPU, CPU, DeviceArray, copy_to_CPU, get_device
export zeros, ones, fill

abstract type Device end
struct GPU <: Device end
struct CPU <: Device end

DeviceArray(::GPU) = CuArray
DeviceArray(::CPU) = Array

get_device(_) = @assert false "Input argument should be an `Array` or a `CuArray`."
get_device(::Array) = CPU()
get_device(::CuArray) = GPU()

copy_to_CPU(arr) = arr
copy_to_CPU(arr::CuArray) = Array(arr)

"""
A device agnostic zeros, ones & fill array.
"""
Base.zeros(T, ::CPU, dims...) = Base.zeros(T, dims...)
Base.zeros(T, ::GPU, dims...) = CUDA.zeros(T, dims...)

Base.zeros(device::Device, dims...) = zeros(Float64, device, dims)

Base.ones(T, ::CPU, dims...) = Base.ones(T, dims...)
Base.ones(T, ::GPU, dims...) = CUDA.ones(T, dims...)

Base.ones(device::Device, dims...) = ones(Float64, device, dims)

Base.fill(x, ::CPU, dims...) = Base.fill(x, dims...)
Base.fill(x, ::GPU, dims...) = CUDA.fill(x, dims...)

"""
A device agnostic parallel loop construct.
"""
function foreach end

function foreach(f, xs, ::CPU)
    # TODO: use @threads?
    for x in xs
        f(x)
    end
    return nothing
end

function foreach(f, xs, ::GPU)
    # Hopefully, CUDA provides a simple loop API
    # in the future and we do not have to use this hack with
    # `map` which involves some useless allocations.
    xs_gpu = CuArray(xs)
    map(xs_gpu) do x
        f(x)
        return nothing
    end
    return nothing
end

"""
Simpler versions of base functions that can easily be handled by the GPU compiler.
See https://github.com/JuliaGPU/CUDA.jl/issues/1548.
Ultimately, these definitions should be moved to CUDA.jl and use @device_override.
"""
module KernelFuns
using CUDA

# It isn't clear all of these are needed, except argmax
# since the version in Base does not have an `init` argument.
# TODO: do some profiling?

    function argmax(f, xs; init)
        best_x, best_y = init
        for x in xs
            y = f(x)
            if y > best_y
                best_y = y
                best_x = x
            end
        end
        return best_x
    end

    function argmax(ys; init)
        xs = 1:length(ys)
        return argmax(x -> ys[x], xs; init)
    end

    function maximum(xs; init)
        best = init
        for x in xs
            if x > best
                best = x
            end
        end
        return best
    end
    
    sum(xs::Matrix{T}; dims) where T = Base.sum(xs; dims)
    sum(xs::CuArray{T, 2}; dims) where T = CUDA.sum(xs; dims)

    function sum(xs; init)
        acc = init
        for x in xs
            acc += x
        end
        return acc
    end

    maximum(f, xs; init) = maximum((f(x) for x in xs); init)
    sum(f, xs; init) = sum((f(x) for x in xs); init)

    # A softmax implementation that does not use in place updates
    # and therefore can also be used on StaticArrays.
    function softmax(xs; eps=1e-15)
        z = zero(eltype(xs))
        xs = xs .- maximum(xs; init=z)
        ys = exp.(xs) .+ eltype(xs)(eps)
        return ys ./ sum(ys; init=z)
    end
end
end
