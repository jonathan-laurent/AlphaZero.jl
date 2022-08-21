"""
Interface that encapsulates inference (through `EnvOracle`) and training of both AlphaZero
and MuZero.
"""
module TrainableEnvOracleModule

using ..Storage

export TrainableEnvOracle, make_feature_and_target, update_weights, get_EnvOracle

abstract type TrainableEnvOracle end

"""
    make_feature_and_target(::GameHistory, ::TrainableEnvOracle, state_index)
    
Provide a single features-target pair usable to train the neural networks.

Singular logic of AlphaZero & MuZero to create the training batch (i.e. single predictive
neural networks of AlphaZero vs the three neural networks of MuZero: prediction,
representation & dynamic) should be expressed in this function.

See also [`update_weights`](@ref)
"""
function make_feature_and_target end

"""
    update_weights(trainable_oracle::TrainableEnvOracle, batch, train_settings)
    
Update the weights of the associated neural networks.

In the same way than [`make_feature_and_target`](@ref), encapsulates the core logic of
neural networks training.

See also [`make_feature_and_target`](@ref)
"""
function update_weights end

"""
    get_EnvOracle(::TrainableEnvOracle)
    
Return an Environement Oracle.

This way, the neural networks singularities of AlphaZero/ MuZero to interact with the
environement (i.e. an emulator for AlphaZero or a latent representation for MuZero) is
hidden behind this `EnvOracle`.

See [`EnvOracle`](@ref) for more details.
"""
function get_EnvOracle end

end
