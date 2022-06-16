module StaticBitArrays

using Test
using StaticArrays
using Random: MersenneTwister, randsubseq

export StaticBitArray

# Chunks are stored from lowest bits to highest bits
struct StaticBitArray{Size,NumChunks}
    chunks::SVector{NumChunks,UInt64}
end

function StaticBitArray(size::Number, num_chunks::Number)
    @assert 0 <= size <= 64 * num_chunks
    chunks = zeros(SVector{num_chunks,UInt64})
    return StaticBitArray{size,num_chunks}(chunks)
end

function StaticBitArray(size::Number)
    @assert 0 <= size
    num_chunks, rem = divrem(size, 64)
    rem > 0 && (num_chunks += 1)
    return StaticBitArray(size, num_chunks)
end

function decompose_idx(idx, ::Val{N}) where {N}
    if N == 1
        # Optimization: no need for a division if there is only one chunk
        q, rem = 0, idx - 1
    else
        q, rem = divrem(idx - 1, 64)
    end
    chunk_id = q + 1
    chunk_offset = rem
    return chunk_id, chunk_offset
end

function Base.setindex(arr::StaticBitArray{S,N}, ::Val{b}, idx) where {S,N,b}
    chunkid, offset = decompose_idx(idx, Val(N))
    chunk = arr.chunks[chunkid]
    if b
        chunk |= (UInt64(1) << offset)
    else
        chunk &= ~(UInt64(1) << offset)
    end
    return StaticBitArray{S,N}(setindex(arr.chunks, chunk, chunkid))
end

Base.setindex(arr::StaticBitArray, b::Bool, idx) = setindex(arr, Val(b), idx)

function Base.getindex(arr::StaticBitArray{S,N}, idx) where {S,N}
    chunk_id, offset = decompose_idx(idx, Val(N))
    return !iszero(arr.chunks[chunk_id] & (UInt64(1) << offset))
end

function run_tests()
    rng = MersenneTwister(0)
    @testset "static bit array operations" begin
        for size in [4, 64, 65, 1000]
            for i in 1:100
                indices = randsubseq(rng, 1:size, 0.3)
                arr = StaticBitArray(size)
                for i in indices
                    arr = setindex(arr, true, i)
                end
                @test all(arr[i] == (i ∈ indices) for i in 1:size)
                for i in 1:size
                    arr = setindex(arr, !arr[i], i)
                end
                @test all(arr[i] == (i ∉ indices) for i in 1:size)
            end
        end
    end
end

end
