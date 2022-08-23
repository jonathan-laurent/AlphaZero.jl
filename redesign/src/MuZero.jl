module MuZero

using Flux

using ..Storage
using ..TrainableEnvOracleModule

export MuZeroTrainableEnvOracle

struct MuZeroTrainableEnvOracle <: TrainableEnvOracle end

function make_target(history, state_index, target_end_unroll, train_settings)
    map(state_index:target_end_unroll) do current_index
        bootstrap_index = current_index + train_settings.td_steps
        value = if (bootstrap_index > length(history))
            0
        else
            history.values[bootstrap_index] * train_settings.discount^train_settings.td_steps
        end

        borned_bootstrap_index = min(length(history), bootstrap_index)
        for (i, reward) in enumerate(history.rewards[current_index:borned_bootstrap_index])
            value += reward * train_settings.discount^(i - 1)
        end

        reward = if (current_index > 1 && current_index <= (length(history) + 1))
            history.rewards[current_index - 1]
        else
            0
        end

        policy = (current_index > length(history)) ? [] : history.policies[current_index]
        return (value, reward, policy)
    end
end

function TrainableEnvOracleModule.make_feature_and_target(
    history::GameHistory, ::MuZeroTrainableEnvOracle, state_index, train_settings
)
    target_end_unroll = state_index + train_settings.num_unroll_steps
    # Unroll of actions has one less element
    end_unroll = min(length(history), target_end_unroll - 1)

    targets = make_target(history, state_index, target_end_unroll, train_settings)
    actions = history.actions[state_index:end_unroll]
    image = history.images[state_index]

    return (image, actions, targets)
end

function TrainableEnvOracleModule.update_weights(
    trainable_oracle::MuZeroTrainableEnvOracle, batch, train_settings
)
    # TODO
    return nothing
end

using ..BatchedMcts
function TrainableEnvOracleModule.get_EnvOracle(trainable_oracle::MuZeroTrainableEnvOracle)
    return UniformTicTacToeEnvOracle()
end

### Michael's Code ###
# Would need a (great) refacto to improve code readbility

# lossₚ(p̂, p)::Float32 = Flux.Losses.crossentropy(p̂, p) #TODO move to hyper
# lossᵥ(v̂, v)::Float32 = Flux.Losses.mse(v̂, v)
# # lossᵣ(r̂, r)::Float32 = 0f0
# lossᵣ(r̂, r)::Float32 = Flux.Losses.mse(r̂, r)

# # lossᵥ(v̂, v)::Float32 = Flux.Losses.crossentropy(v̂, v)
# # lossᵣ(r̂, r)::Float32 = Flux.Losses.crossentropy(r̂, r)

# # TODO add assertions about sizes
# function losses(nns, hyper, (X, A_mask, As, Ps, Vs, Rs))
#     prediction, dynamics, representation = nns.f, nns.g, nns.h
#     creg::Float32 = hyper.l2_regularization
#     Ksteps = hyper.num_unroll_steps
#     dimₐ = hyper.model_type == :mlp ? 2 : 3

#     # initial step, from the real observation
#     Hiddenstate = forward(representation, X)
#     P̂⁰, V̂⁰ = forward(prediction, Hiddenstate)
#     P̂⁰ = normalize_p(P̂⁰, A_mask)
#     # R̂⁰ = zero(V̂⁰)
#     # batchdim = ndims(Hiddenstate)

#     scale_initial = iszero(Ksteps) ? 1.0f0 : 0.5f0
#     Lp = scale_initial * lossₚ(P̂⁰, Ps[:, 1, :]) # scale=1
#     Lv = scale_initial * lossᵥ(V̂⁰, Vs[1:1, :])
#     Lr = zero(Lv) # starts at next step (see MuZero paper appendix)

#     scale_recurrent = iszero(Ksteps) ? nothing : 0.5f0 / Ksteps #? instead of constant scale, maybe 2^(-i+1)
#     # recurrent inference 
#     for k in 1:Ksteps
#         # targets are stored as follows: [A⁰¹ A¹² ...] [P⁰ P¹ ...] [V⁰ V¹ ...] but [R¹ R² ...]
#         # A = As[k, :]
#         # A = As[:,:,k:k,:]
#         A = selectdim(As, dimₐ, k)
#         S_A = cat(Hiddenstate, A; dims=ndims(Hiddenstate) - 1)
#         # R̂, Hiddenstate = forward(dynamics, Hiddenstate, A) # obtain next hiddenstate
#         R̂, Hiddenstate = forward(dynamics, S_A) # obtain next hiddenstate
#         P̂, V̂ = forward(prediction, Hiddenstate) #? should flip V based on players
#         # scale loss so that the overall weighting of the recurrent_inference (g,f nns)
#         # is equal to that of the initial_inference (h,f nns)
#         Lp += scale_recurrent * lossₚ(P̂, Ps[:, k + 1, :]) #? @view
#         Lv += scale_recurrent * lossᵥ(V̂, Vs[(k + 1):(k + 1), :])
#         Lr += scale_recurrent * lossᵣ(R̂, Rs[k:k, :])
#     end
#     Lreg =
#         iszero(creg) ? zero(Lv) : creg * sum(sum(w .^ 2) for w in regularized_params(nns))
#     L = Lp + Lv + Lr + Lreg # + Lr
#     # L = Lp + Lreg # + Lr
#     # Zygote.@ignore @info "Loss" loss_total=L loss_policy=Lp loss_value=Lv loss_reward=Lr loss_reg_params=Lreg relative_entropy=Lp-Flux.Losses.crossentropy(Ps, Ps) #? check if compute means inside logger is avaliable
#     return (L, Lp, Lv, Lr, Lreg)
# end

# # #TODO replace Zygote.withgradient() - new version
# # function lossgrads(f, args...)
# #   val, back = Zygote.pullback(f, args...)
# #   grad = back(Zygote.sensitivity(val))
# #   return val, grad
# # end

# # function train!(nns, opt, loss, data; cb=()->())
# function μtrain!(nns, loss, data, opt)
#     ps = Flux.params(nns)
#     losses = Float32[]
#     @progress "learning step (checkpoint)" for (i, d) in enumerate(data)
#         # for (i, d) in enumerate(data)
#         l, gs = Zygote.withgradient(ps) do
#             loss(d...)
#         end
#         push!(losses, l)
#         Flux.update!(opt, ps, gs)
#         # @info "debug" η=opt.optim.eta
#     end
#     @info "Loss" mean_loss_total = mean(losses)
# end

# struct MuTrainer
#     gspec
#     nns # MuNetwork
#     memory
#     hyper
#     opt
#     function MuTrainer(gspec, nns, memory, hyper, opt)
#         return new(gspec, Flux.trainmode!.(hyper.device.(nns)), memory, hyper, opt)
#     end
# end

# function update_weights!(tr::MuTrainer, n)
#     L(batch...) = losses(tr.nns, tr.hyper, batch)[1]
#     samples = (tr.hyper.device.(sample_batch(tr.gspec, tr.memory, tr.hyper)) for _ in 1:n)

#     return μtrain!(tr.nns, L, samples, tr.opt)
# end

end
