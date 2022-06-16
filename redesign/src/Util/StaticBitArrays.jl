module StaticBitArrays

using Test
using StaticArrays
using Random: MersenneTwister, randsubseq

export StaticBitArray

# Chunks are stored from lowest bits to highest bits
struct StaticBitArray{Size,NumChunks}
    chunks::SVector{NumChunks,UInt64}
end

function StaticBitArray{Size,NumChunks}() where {Size,NumChunks}
    @assert 0 <= Size <= 64 * NumChunks
    chunks = zeros(SVector{NumChunks,UInt64})
    return StaticBitArray{Size,NumChunks}(chunks)
end

function StaticBitArray{Size}() where {Size}
    @assert 0 <= Size
    NumChunks, rem = divrem(Size, 64)
    rem > 0 && (NumChunks += 1)
    return StaticBitArray{Size,NumChunks}()
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

function Base.setindex(arr::StaticBitArray{S,N}, b::Bool, idx) where {S,N}
    chunkid, offset = decompose_idx(idx, Val(N))
    chunk = arr.chunks[chunkid]
    if b
        chunk |= (UInt64(1) << offset)
    else
        chunk &= ~(UInt64(1) << offset)
    end
    return StaticBitArray{S,N}(setindex(arr.chunks, chunk, chunkid))
end

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
                arr = StaticBitArray{size}()
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
