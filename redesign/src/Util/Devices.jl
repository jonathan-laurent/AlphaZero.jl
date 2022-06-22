module Devices

using CUDA

export Device, GPU, CPU, DeviceArray

abstract type Device end
struct GPU <: Device end
struct CPU <: Device end

DeviceArray(::GPU) = CuArray
DeviceArray(::CPU) = Array

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

    function maximum(xs; init)
        best = init
        for x in xs
            if x > best
                best = x
            end
        end
        return best
    end

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
