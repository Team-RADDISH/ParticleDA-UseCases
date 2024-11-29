module SpeedyWeatherSSM

using ParticleDA
using Random
using SpeedyWeather

LAYERED_VARIABLES = (:vor, :div, :temp, :humid)
SURFACE_VARIABLES = (:pres,)

Base.@kwdef struct SpeedyParameters{T<:AbstractFloat, M<:SpeedyWeather.AbstractModel}
    spectral_truncation::Int = 31
    n_layers::Int = 8
    n_days::T = 0.25
    model_type::Type{M} = PrimitiveWetModel
end

struct SpeedyModel{
    T<:AbstractFloat,
    G<:SpeedyWeather.AbstractSpectralGrid,
    M<:SpeedyWeather.AbstractModel,
}
    parameters::SpeedyParameters{T, M}
    spectral_grid::G
    model::M
    prognostic_variables::PrognosticVariables{T}
    diagnostic_variables::DiagnosticVariables{T}
    n_layered_variables::Int
    n_surface_variables::Int
end


function init(parameters::SpeedyParameters{T, M}) where {
    T<:AbstractFloat, M<:SpeedyWeather.AbstractModel
}
    spectral_grid = SpectralGrid(;
        NF=T, trunc=parameters.spectral_truncation, nlayers=parameters.n_layers
    )
    model = M(; spectral_grid)
    simulation = initialize!(model)
    (; prognostic_variables, diagnostic_variables) = simulation
    n_layered_variables = count(
        SpeedyWeather.has(model, var) for var in LAYERED_VARIABLES
    )
    n_surface_variables = count(
        SpeedyWeather.has(model, var) for var in SURFACE_VARIABLES
    )
    return SpeedyModel(
        parameters,
        spectral_grid,
        model,
        prognostic_variables,
        diagnostic_variables,
        n_layered_variables,
        n_surface_variables,
    )
end

function ParticleDA.get_state_dimension(model::SpeedyModel)
    (model.parameters.spectral_truncation + 1)^2 * (
        model.parameters.n_layers * model.n_layered_variables + model.n_surface_variables
    )
end

function ParticleDA.get_observation_dimension(model::SpeedyModel)
    
end

function update_spectral_coefficients_from_vector!(
    spectral_coefficients::SpeedyWeather.LowerTriangularMatrix{Complex{T}}, 
    vector::Vector{T}
) where {T <: AbstractFloat}
    m, n = size(spectral_coefficients)
    spectral_coefficients[1:m - 1, 1] = vector[1:m - 1]
    j = m - 1
    for i in 2:n 
        spectral_coefficients[i:m - 1, i] = reinterpret(
            Complex{T}, vector[j+1:j+(m - i) * 2]
        )
        j = j + (m - i) * 2
    end
end

function update_prognostic_variables_from_state_vector!(
    prognostic_variables::PrognosticVariables{T},
    state::Vector{T},
) where {T <: AbstractFloat}
    start_index = 1
    dim_spectral = (prognostic_variables.trunc + 1)^2
    for name in LAYERED_VARIABLES
        if SpeedyWeather.has(prognostic_variables, name)
            layer_spectral_coefficients = SpeedyWeather.get_var(
                prognostic_variables, name; lf=1
            )
            for spectral_coefficients in layer_spectral_coefficients
                end_index = start_index + dim_spectral - 1
                update_spectral_coefficients_from_vector!(
                    spectral_coefficients,
                    state[start_index:end_index]
                )
                start_index = end_index + 1
            end
        end
    end
    if SpeedyWeather.has(prognostic_variables, :pres)
        update_spectral_coefficients_from_vector!(
            SpeedyWeather.get_pressure(prognostic_variables; lf=1),
            state[start_index:end]
        )
    end
end

function update_vector_from_spectral_coefficients!(
    vector::Vector{T},
    spectral_coefficients::SpeedyWeather.LowerTriangularMatrix{Complex{T}}, 
) where {T <: AbstractFloat}
    m, n = size(spectral_coefficients, as=Matrix)
    vector[1:m - 1] = spectral_coefficients[1:m - 1, 1]
    j = m - 1
    for i in 2:n 
        vector[j+1:j+(m - i) * 2] = reinterpret(T, spectral_coefficients[i:m - 1, i])
        j = j + (m - i) * 2
    end
end

function update_state_vector_from_prognostic_variables!(
    state::Vector{T},
    prognostic_variables::PrognosticVariables{T}
) where {T <: AbstractFloat}
    start_index = 1
    dim_spectral = (prognostic_variables.trunc + 1)^2
    for name in (:vor, :div, :temp, :humid)
        if SpeedyWeather.has(prognostic_variables, name)
            layer_spectral_coefficients = SpeedyWeather.get_var(
                prognostic_variables, name; lf=1
            )
            for spectral_coefficients in layer_spectral_coefficients
                end_index = start_index + dim_spectral - 1
                update_vector_from_spectral_coefficients!(
                    state[start_index:end_index],
                    spectral_coefficients
                )
                start_index = end_index + 1
            end
        end
    end
    if SpeedyWeather.has(prognostic_variables, :pres)
        update_vector_from_spectral_coefficients!(
            state[start_index:end],
            SpeedyWeather.get_pressure(prognostic_variables; lf=1)
        )
    end
end

ParticleDA.get_state_eltype(model::SpeedyModel{T}) where {T<:AbstractFloat} = T

ParticleDA.get_observation_eltype(model::SpeedyModel{T}) where {T<:AbstractFloat} = T

function ParticleDA.sample_initial_state!(
    state::Vector{T}, model::SpeedyModel{T}, rng::R
) where {T<:AbstractFloat, R<:AbstractRNG}
    initial_conditions = model.model.initial_conditions
    SpeedyWeather.initialize!(
        model.prognostic_variables, initial_conditions, model.model
    )
    update_state_vector_from_prognostic_variables!(state, model.prognostic_variables)
end

function ParticleDA.update_state_deterministic!(
    state::Vector{T}, model::SpeedyModel{T}, time_index::Int
) where {T<:AbstractFloat}
    update_prognostic_variables_from_state_vector!(model.prognostic_variables, state)
    (; clock) = model.prognostic_variables
    (; time_stepping) = model.model
    SpeedyWeather.set_period!(clock, SpeedyWeather.Day(model.parameter.n_days))
    SpeedyWeather.initialize!(clock, time_stepping)
    clock.time += clock.n_timesteps * clock.Δt * (time_index - 1)
    clock.timestep_counter += clock.n_timesteps * (time_index - 1)
    SpeedyWeather.time_stepping!(
        model.prognostic_variables, model.diagnostic_variables, model.model
    )
    update_state_vector_from_prognostic_variables!(state, model.prognostic_variables)
end

function ParticleDA.update_state_stochastic!(    
    state::Vector{T}, model::SpeedyModel{T}, rng::G
) where {T<:AbstractFloat, G<:AbstractRNG}

end

function ParticleDA.sample_observation_given_state!(
    observation::Vector{T}, state::Vector{T}, model::SpeedyModel{T}, rng::G
) where {T<:AbstractFloat, G<:AbstractRNG}
    update_prognostic_variables_from_state_vector!(model.prognostic_variables, state)
    SpeedyWeather.transform!(
        model.diagnostic_variables,
        model.prognostic_variables,
        1,
        model.model,
        true
    )
end

function ParticleDA.get_log_density_observation_given_state(observation, state, model)

end

function ParticleDA.write_model_metadata(file, model)
    
end

end
