"""Axissymmetric non-axissymmetric extended (ANAET) state space model.

The deterministic dynamics here correspond to a toy model of magnetohydrodynamic
instability coupled to a slow dissipative background evolution 
(Arter, 2012; https://doi.org/10.13140/RG.2.2.35052.77449), with the original ordinary
differential equation system here changed to a stochastic differential equation system

    dx₁(t) = x₂(t) dt + + β dw₁(t),
    dx₂(t) = −γᵣ x₁(t) - (μ₁ + μ₂ x₃(t)) x₁(t)³ - μ₆ x₁(t)⁶ x₂(t) dt + β dw₂(t),
    dx₃(t) = (ν₁ - ν₂ x₃(t)² - (δ₀ + δ₁ b) x₁(t)²) dt + β dw₃(t).

The state space model formulation uses a Gaussian approximation to the state transition
distributions for the SDE system based on a splitting numerical scheme which uses an 
adaptive ODE solver to solve for the deterministic component of the dynamics and 
Euler-Maruyama discretisation for the driving Wiener noise processes.
"""
module ANAET

using Distributions
using HDF5
using Random
using PDMats
using OrdinaryDiffEq
using SciMLBase
using ParticleDA

Base.@kwdef struct ANAETModelParameters{S <: Real, T <: Real}
    γ_r::S = 1.
    μ_1::S =  0.
    δ_1::S =  0.
    μ_2::S = -2.
    μ_6::S = 1e-2
    ν_1::S = 1e-3
    ν_2::S = 1e-3
    δ_0::S = 1.5e-4
    time_step::S = 1.
    observed_indices::Union{UnitRange{Int}, StepRange{Int, Int}, Vector{Int}} = 1:1
    initial_state_mean::Vector{S} = [1., 0., 0.]
    initial_state_std::Union{S, Vector{S}} = 0.01
    state_noise_std::Union{S, Vector{S}} = 0.01
    observation_noise_std::Union{T, Vector{T}} = 0.1
end

function get_params(
    P::Type{ANAETModelParameters{S, T}}, model_params_dict::Dict
) where {S <: Real, T <: Real}
    return P(; (; (Symbol(k) => v for (k, v) in model_params_dict)...)...)
end

struct ANAETModel{S <: Real, T <: Real, I <: SciMLBase.AbstractODEIntegrator}
    parameters::ANAETModelParameters{S, T}
    integrators::Vector{I}
    initial_state_distribution::MvNormal{S}
    state_noise_distribution::MvNormal{S}
    observation_noise_distribution::MvNormal{T}
end

function update_time_derivative!(
    dx_dt::Vector{S}, x::Vector{S}, p::ANAETModelParameters{S, T}, t::U
) where {S <: Real, T <: Real, U <: Real}
    dx_dt[1] = x[2]
    dx_dt[2] = -p.γ_r * x[1] - (p.μ_1 + p.μ_2 * x[3]) * x[1]^3 - p.μ_6 * x[1]^6 * x[2]
    dx_dt[3] = p.ν_1 - p.ν_2 * x[2]^2 - (p.δ_0 + p.δ_1 * x[2]) * x[1]^2
end

function init(
    parameters_dict::Dict,
    n_tasks::Int=1; 
    S::Type{<:Real}=Float64,
    T::Type{<:Real}=Float64
)
    parameters = get_params(ANAETModelParameters{S, T}, parameters_dict)
    time_span = (0, parameters.time_step)
    integrators = [
        OrdinaryDiffEq.init(
            ODEProblem(update_time_derivative!, x, time_span, parameters), 
            Tsit5();
            save_everystep=false
        ) 
        for x in eachcol(zeros(S, 3, n_tasks))
    ]
    state_dimension = 3
    observation_dimension = length(parameters.observed_indices)
    return ANAETModel(
        parameters,
        integrators,
        (
            MvNormal(m, isa(s, Vector) ? PDiagMat(s.^2) : ScalMat(length(m), s.^2))
            for (m, s) in (
                (parameters.initial_state_mean, parameters.initial_state_std), 
                (zeros(S, state_dimension), parameters.state_noise_std), 
                (zeros(T, observation_dimension), parameters.observation_noise_std), 
            )
        )...
    )
end

ParticleDA.get_state_dimension(::ANAETModel) = 3
ParticleDA.get_observation_dimension(model::ANAETModel) = length(
    model.parameters.observed_indices
)
ParticleDA.get_state_eltype(::ANAETModel{S, T}) where {S, T} = S
ParticleDA.get_observation_eltype(::ANAETModel{S, T}) where {S, T} = T

function ParticleDA.sample_initial_state!(
    state::AbstractVector{T},
    model::ANAETModel{S, T}, 
    rng::Random.AbstractRNG,
    task_index::Int=1,
) where {S, T}
    rand!(rng, model.initial_state_distribution, state)
end

function ParticleDA.update_state_deterministic!(
    state::AbstractVector{T}, 
    model::ANAETModel{S, T}, 
    time_index::Int,
    task_index::Int=1,
) where {S, T}
    reinit!(model.integrators[task_index], state)
    step!(model.integrators[task_index], model.parameters.time_step, true)
    state .= model.integrators[task_index].u
end

function ParticleDA.update_state_stochastic!(
    state::AbstractVector{T}, 
    model::ANAETModel{S, T}, 
    rng::Random.AbstractRNG,
    task_index::Int=1,
) where {S, T}
    rand!(rng, state + model.state_noise_distribution, state)
end

function ParticleDA.sample_observation_given_state!(
    observation::AbstractVector{T},
    state::AbstractVector{S}, 
    model::ANAETModel{S, T}, 
    rng::Random.AbstractRNG,
    task_index::Int=1,
) where {S <: Real, T <: Real}
    rand!(
        rng,
        view(state, model.parameters.observed_indices) 
        + model.observation_noise_distribution,
        observation
    )
end

function ParticleDA.get_log_density_observation_given_state(
    observation::AbstractVector{T},
    state::AbstractVector{S},
    model::ANAETModel{S, T},
    task_index::Int=1,
) where {S <: Real, T <: Real}
    return logpdf(
        view(state, model.parameters.observed_indices) 
        + model.observation_noise_distribution,
        observation
    )
end

function ParticleDA.write_model_metadata(file::HDF5.File, model::ANAETModel)
    group_name = "parameters"
    if !haskey(file, group_name)
        group = create_group(file, group_name)
        for field in fieldnames(typeof(model.parameters))
            value = getfield(model.parameters, field)
            attributes(group)[string(field)] = (
                isa(value, AbstractVector) ? collect(value) : value
            )
        end
    else
        @warn "Write failed, group $group_name already exists in  $(file.filename)!"
    end
end

function ParticleDA.get_observation_mean_given_state!(
    observation_mean::AbstractVector{T},
    state::AbstractVector{S},
    model::ANAETModel{S, T},
    task_index::Int=1,
) where {S <: Real, T <: Real}
    observation_mean .= view(state, model.parameters.observed_indices)
end

function ParticleDA.get_covariance_state_noise(model::ANAETModel)
    return model.state_noise_distribution.Σ
end

function ParticleDA.get_covariance_state_noise(model::ANAETModel, i::Int, j::Int)
    return model.state_noise_distribution.Σ[i, j]
end

function ParticleDA.get_covariance_observation_noise(model::ANAETModel)
    return model.observation_noise_distribution.Σ
end

function ParticleDA.get_covariance_observation_noise(
    model::ANAETModel, i::Int, j::Int
)
    return model.observation_noise_distribution.Σ[i, j]
end

function ParticleDA.get_covariance_state_observation_given_previous_state(
    model::ANAETModel, i::Int, j::Int
)
    return model.state_noise_distribution.Σ[i, model.parameters.observed_indices[j]]
end

function ParticleDA.get_covariance_observation_observation_given_previous_state(
    model::ANAETModel, i::Int, j::Int
)
    return (
        model.state_noise_distribution.Σ[
            model.parameters.observed_indices[i], model.parameters.observed_indices[j]
        ]
        + model.observation_noise_distribution.Σ[i, j]
    )
end

function ParticleDA.get_state_indices_correlated_to_observations(model::ANAETModel)
    return model.parameters.observed_indices
end

end
