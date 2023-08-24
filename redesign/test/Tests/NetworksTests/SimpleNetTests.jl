module SimpleNetTests

using Flux
using Random: MersenneTwister
using Test

using ....Network


export run_simplenet_tests


function run_simplenet_tests()
    @testset "SimpleNet: Flux.params() correctness" test_simplenet_functor_params()
    @testset "SimpleNet: model conversion to gpu and back" test_simplenet_on_gpu()
    @testset "SimpleNet: model mode conversion" test_simplenet_modes()
    @testset "SimpleNet: copy model" test_simplenet_copy()
    @testset "SimpleNet: model forward" test_simplenet_forward()
end

function test_simplenet_functor_params()
    width = 13; depth_common = 2
    hp = SimpleNetHP(width=width, depth_common=depth_common, use_batch_norm=false)

    input_size = 2; output_size = 4
    nn = SimpleNet(input_size, output_size, hp)

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
        @testset "dense1" begin
            @test nn.common[3].weight === functor_params[3]
            @test size(nn.common[3].weight) == (width, width)
            @test nn.common[3].bias === functor_params[4]
            @test size(nn.common[3].bias) == (width,)
            @test nn.common[3].σ === Flux.relu
        end
        @testset "dense2" begin
            @test nn.common[4].weight === functor_params[5]
            @test size(nn.common[4].weight) == (width, width)
            @test nn.common[4].bias === functor_params[6]
            @test size(nn.common[4].bias) == (width,)
            @test nn.common[4].σ === Flux.relu
        end
    end

    @testset "vhead" begin
        @testset "dense1" begin
            @test nn.vhead[1].weight === functor_params[7]
            @test size(nn.vhead[1].weight) == (width, width)
            @test nn.vhead[1].bias === functor_params[8]
            @test size(nn.vhead[1].bias) == (width,)
            @test nn.vhead[1].σ === Flux.relu
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
        @testset "dense1" begin
            @test nn.phead[1].weight === functor_params[11]
            @test size(nn.phead[1].weight) == (width, width)
            @test nn.phead[1].bias === functor_params[12]
            @test size(nn.phead[1].bias) == (width,)
            @test nn.phead[1].σ === Flux.relu
        end
        @testset "dense-output" begin
            @test nn.phead[2].weight === functor_params[13]
            @test size(nn.phead[2].weight) == (output_size, width)
            @test nn.phead[2].bias === functor_params[14]
            @test size(nn.phead[2].bias) == (output_size,)
        end
    end
end

function test_simplenet_on_gpu()
    nn = SimpleNet(2, 4, SimpleNetHP(width=3, depth_common=3))

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

function test_simplenet_modes()
    nn = SimpleNet(2, 4, SimpleNetHP(width=3, depth_common=3, use_batch_norm=true))

    function test_mode(mode)
        @testset "common batch norms" begin
            @test nn.common[2][2].active === mode
            @test nn.common[3][2].active === mode
            @test nn.common[4][2].active === mode
            @test nn.common[5][2].active === mode
        end
        @testset "vhead batch norm" begin
            @test nn.vhead[1][2].active === mode
        end
        @testset "phead batch norm" begin
            @test nn.phead[1][2].active === mode
        end
    end

    @testset "nothing by default" test_mode(nothing)

    @testset "training mode" begin
        set_train_mode!(nn)
        test_mode(true)
    end

    @testset "test mode" begin
        set_test_mode!(nn)
        test_mode(false)
    end

    @testset "back to training mode" begin
        set_train_mode!(nn)
        test_mode(true)
    end
end

function test_simplenet_copy()
    nn = SimpleNet(27, 9, SimpleNetHP(width=42, depth_common=2))
    nn2 = copy(nn)

    function test_dense_layer_equality(layer1, layer2)
        @test layer1.weight == layer2.weight
        @test layer1.weight !== layer2.weight

        @test layer1.bias == layer2.bias
        @test layer1.bias !== layer2.bias

        @test layer1.σ == layer2.σ
    end

    @testset "common" begin
        @testset "input-dense" begin
            @test nn2.common[1] == nn.common[1]
            test_dense_layer_equality(nn2.common[2], nn.common[2])
        end
        @testset "dense1" test_dense_layer_equality(nn2.common[3], nn.common[3])
        @testset "dense2" test_dense_layer_equality(nn2.common[4], nn.common[4])
    end

    @testset "vhead" begin
        @testset "dense1" test_dense_layer_equality(nn2.vhead[1], nn.vhead[1])
        @testset "dense-output" test_dense_layer_equality(nn2.vhead[2], nn.vhead[2])
    end

    @testset "phead" begin
        @testset "dense1" test_dense_layer_equality(nn2.phead[1], nn.phead[1])
        @testset "dense-output" begin
            test_dense_layer_equality(nn2.phead[2], nn.phead[2])
        end
    end
end

function test_simplenet_forward()
    width = 64; batch_size = 3
    nn = SimpleNet(126, 7, SimpleNetHP(width=width, depth_common=2))

    rng = MersenneTwister(42)
    input = rand(rng, Float32, 126, batch_size)

    nn_vhead_out, nn_phead_out = forward(nn, input, false)

    @testset "forward pass correct output size" begin
        @test size(nn_vhead_out) == (1, batch_size)
        @test size(nn_phead_out) == (7, batch_size)
    end

    @testset "manual forward pass" begin
        a1_common = nn.common[2].σ.(nn.common[2].weight * input .+ nn.common[2].bias)
        a2_common = nn.common[3].σ.(nn.common[3].weight * a1_common .+ nn.common[3].bias)
        common_out = nn.common[4].σ.(nn.common[4].weight * a2_common .+ nn.common[4].bias)
        @test size(common_out) == (width, batch_size)

        a1_vhead = nn.vhead[1].σ.(nn.vhead[1].weight * common_out .+ nn.vhead[1].bias)
        vhead_out = nn.vhead[2].σ.(nn.vhead[2].weight * a1_vhead .+ nn.vhead[2].bias)
        @test size(vhead_out) == (1, batch_size)

        a1_phead = nn.phead[1].σ.(nn.phead[1].weight * common_out .+ nn.phead[1].bias)
        a2_phead = nn.phead[2].σ.(nn.phead[2].weight * a1_phead .+ nn.phead[2].bias)
        phead_out = a2_phead
        @test size(phead_out) == (7, batch_size)

        @test nn_vhead_out ≈ vhead_out
        @test nn_phead_out ≈ phead_out
    end
end

end
