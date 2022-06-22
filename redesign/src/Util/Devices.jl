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

end
