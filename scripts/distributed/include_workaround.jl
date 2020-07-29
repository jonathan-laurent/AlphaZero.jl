# This fix only works on a single machine
function include_everywhere(filepath)
    fullpath = joinpath(@__DIR__, filepath)
    @sync for p in procs()
        @async remotecall_wait(include, p, fullpath)
    end
end
