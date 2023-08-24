module SimpleResNetTests

using Flux
using Random: MersenneTwister
using Test

using ....Network


export run_simpleresnet_tests


function run_simpleresnet_tests()
    @testset "SimpleResNet: Flux.params() correctness" test_simpleresnet_functor_params()
    @testset "SimpleResNet: model conversion to gpu and back" test_simpleresnet_on_gpu()
    @testset "SimpleResNet: copy model" test_simpleresnet_copy()
    @testset "SimpleResNet: model forward" test_simpleresnet_forward()
end

function test_simpleresnet_functor_params()
    width = 52; depth_common = 2
    hp = SimpleResNetHP(width=width, depth_common=depth_common)

    input_size = 2; output_size = 4
    nn = SimpleResNet(input_size, output_size, hp)

    functor_params = Flux.params(nn)

    @testset "common" begin
        @testset "input-dense" begin
            @test nn.common[1] === Flux.flatten
            @test nn.common[2].weight === functor_params[1]
            @test size(nn.common[2].weight) == (width, input_size)
            @test nn.common[2].bias === functor_params[2]
            @test size(nn.common[2].bias) == (width,)
            @test nn.common[2].σ === Flux.relu
        end
        @testset "res-dense1" begin
            @test typeof(nn.common[3][1]) <: Flux.SkipConnection
            @test typeof(nn.common[3][1].layers) <: Flux.Dense
            @test nn.common[3][1].connection == +
            @test nn.common[3][1].layers.weight === functor_params[3]
            @test size(nn.common[3][1].layers.weight) == (width, width)
            @test nn.common[3][1].layers.bias === functor_params[4]
            @test size(nn.common[3][1].layers.bias) == (width,)
            @test nn.common[3][1].layers.σ === Flux.relu
            @test nn.common[3][2] === Flux.relu
        end
        @testset "res-dense2" begin
            @test typeof(nn.common[4][1]) <: Flux.SkipConnection
            @test typeof(nn.common[4][1].layers) <: Flux.Dense
            @test nn.common[4][1].connection == +
            @test nn.common[4][1].layers.weight === functor_params[5]
            @test size(nn.common[4][1].layers.weight) == (width, width)
            @test nn.common[4][1].layers.bias === functor_params[6]
            @test size(nn.common[4][1].layers.bias) == (width,)
            @test nn.common[4][1].layers.σ === Flux.relu
            @test nn.common[4][2] === Flux.relu
        end
    end

    @testset "vhead" begin
        @testset "res-dense1" begin
            @test typeof(nn.vhead[1][1]) <: Flux.SkipConnection
            @test typeof(nn.vhead[1][1].layers) <: Flux.Dense
            @test nn.vhead[1][1].connection == +
            @test nn.vhead[1][1].layers.weight === functor_params[7]
            @test size(nn.vhead[1][1].layers.weight) == (width, width)
            @test nn.vhead[1][1].layers.bias === functor_params[8]
            @test size(nn.vhead[1][1].layers.bias) == (width,)
            @test nn.vhead[1][1].layers.σ === Flux.relu
            @test nn.vhead[1][2] === Flux.relu
        end
        @testset "dense-output" begin
            @test nn.vhead[2].weight === functor_params[9]
            @test size(nn.vhead[2].weight) == (1, width)
            @test nn.vhead[2].bias === functor_params[10]
            @test size(nn.vhead[2].bias) == (1,)
            @test nn.vhead[2].σ === tanh
        end
    end

    @testset "phead" begin
        @testset "res-dense1" begin
            @test typeof(nn.phead[1][1]) <: Flux.SkipConnection
            @test typeof(nn.phead[1][1].layers) <: Flux.Dense
            @test nn.phead[1][1].connection == +
            @test nn.phead[1][1].layers.weight === functor_params[11]
            @test size(nn.phead[1][1].layers.weight) == (width, width)
            @test nn.phead[1][1].layers.bias === functor_params[12]
            @test size(nn.phead[1][1].layers.bias) == (width,)
            @test nn.phead[1][1].layers.σ === Flux.relu
            @test nn.phead[1][2] === Flux.relu
        end
        @testset "dense-output" begin
            @test nn.phead[2].weight === functor_params[13]
            @test size(nn.phead[2].weight) == (output_size, width)
            @test nn.phead[2].bias === functor_params[14]
            @test size(nn.phead[2].bias) == (output_size,)
        end
    end
end

function test_simpleresnet_on_gpu()
    nn = SimpleResNet(2, 4, SimpleResNetHP(width=13, depth_common=3))

    @testset "not on GPU by default" begin
        @test !on_gpu(nn)
    end

    @testset "conversion to GPU" begin
        nn = to_gpu(nn)
        @test on_gpu(nn)
    end

    @testset "conversion back to CPU" begin
        nn = to_cpu(nn)
        @test !on_gpu(nn)
    end
end

function test_simpleresnet_copy()
    nn = SimpleResNet(84, 7, SimpleResNetHP(width=34, depth_common=2))
    nn2 = copy(nn)

    function test_dense_layer_equality(layer1, layer2)
        @test layer1.weight == layer2.weight
        @test layer1.weight !== layer2.weight

        @test layer1.bias == layer2.bias
        @test layer1.bias !== layer2.bias

        @test layer1.σ == layer2.σ
    end

    function test_skip_connection_layer_equality(skip1, skip2)
        test_dense_layer_equality(skip1.layers, skip2.layers)
        @test skip1.connection == skip2.connection
    end

    @testset "common" begin
        @testset "input-dense" begin
            @test nn2.common[1] == nn.common[1]
            test_dense_layer_equality(nn2.common[2], nn.common[2])
        end
        @testset "res-dense1" begin
            test_skip_connection_layer_equality(nn2.common[3][1], nn.common[3][1])
            @test nn2.common[3][2] == nn.common[3][2]
        end
        @testset "res-dense2" begin
            test_skip_connection_layer_equality(nn2.common[4][1], nn.common[4][1])
            @test nn2.common[4][2] == nn.common[4][2]
        end
    end

    @testset "vhead" begin
        @testset "res-dense1" begin
            test_skip_connection_layer_equality(nn2.vhead[1][1], nn.vhead[1][1])
            @test nn2.vhead[1][2] == nn.vhead[1][2]
        end
        @testset "dense-output" test_dense_layer_equality(nn2.vhead[2], nn.vhead[2])
    end

    @testset "phead" begin
        @testset "res-dense1" begin
            test_skip_connection_layer_equality(nn2.phead[1][1], nn.phead[1][1])
            @test nn2.phead[1][2] == nn.phead[1][2]
        end
        @testset "dense-output" test_dense_layer_equality(nn2.phead[2], nn.phead[2])
    end
end

function test_simpleresnet_forward()
    width = 64; batch_size = 3
    nn = SimpleResNet(16, 5, SimpleResNetHP(width=width, depth_common=2))

    rng = MersenneTwister(42)
    input = rand(rng, Float32, 16, batch_size)

    nn_vhead_out, nn_phead_out = forward(nn, input, false)

    @testset "forward pass correct output size" begin
        @test size(nn_vhead_out) == (1, batch_size)
        @test size(nn_phead_out) == (5, batch_size)
    end

    @testset "manual forward pass" begin
        a1_common = nn.common[2].σ.(nn.common[2].weight * input .+ nn.common[2].bias)

        res_dense1 = nn.common[3][1].layers
        a2_dense = res_dense1.σ.(res_dense1.weight * a1_common .+ res_dense1.bias)
        a2_common = nn.common[3][2](a2_dense .+ a1_common)

        res_dense2 = nn.common[4][1].layers
        a3_dense = res_dense2.σ.(res_dense2.weight * a2_common .+ res_dense2.bias)
        common_out = nn.common[4][2](a3_dense .+ a2_common)
        @test size(common_out) == (width, batch_size)

        res_dense1_v = nn.vhead[1][1].layers
        a1_dense_v = res_dense1_v.σ.(res_dense1_v.weight * common_out .+ res_dense1_v.bias)
        a1_vhead = nn.vhead[1][2](a1_dense_v .+ common_out)

        vhead_out = nn.vhead[2].σ.(nn.vhead[2].weight * a1_vhead .+ nn.vhead[2].bias)
        @test size(vhead_out) == (1, batch_size)

        res_dense1_p = nn.phead[1][1].layers
        a1_dense_p = res_dense1_p.σ.(res_dense1_p.weight * common_out .+ res_dense1_p.bias)
        a1_phead = nn.phead[1][2](a1_dense_p .+ common_out)

        phead_out = nn.phead[2].σ.(nn.phead[2].weight * a1_phead .+ nn.phead[2].bias)
        @test size(phead_out) == (5, batch_size)

        @test nn_vhead_out ≈ vhead_out
        @test nn_phead_out ≈ phead_out
    end
end

end
