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
    vector::AbstractVector{T}
) where {T <: AbstractFloat}
    n_row, n_col = size(spectral_coefficients, as=Matrix)
    # First column of spectral_coefficients (order = m = 0) are real-valued and we skip
    # last row (degree = l = n_row - 1) as used only for computing meridional derivative
    # for vector valued fields. LowerTriangularMatrix allows vector (flat) indexing
    # skipping zero upper-triangular entries
    spectral_coefficients[1:n_row - 1] .= vector[1:n_row - 1]
    # vector index is i, spectral coefficient (flat) index is j
    i = n_row - 1
    j = n_row
    for col_index in 2:n_col
        # Slice corresponding to column has non-zero entries from col_index row and we
        # ignore last row as used only for computing meridional derivative for vector
        # valued fields
        slice_size = n_row - col_index
        # Reinterpret real valued state coefficients to complex spectral coefficients
        spectral_coefficients[j + 1:j + slice_size] .= reinterpret(
            Complex{T}, vector[i + 1:i + 2 * slice_size]
        )
        # Zero entry corresponding to last row (degree = l = n_row - 1) as not used
        # for scalar fields
        spectral_coefficients[j + slice_size + 1] = 0
        # Update vector and spectral coefficient indices, adding 1 offset
        # to latter to skip entries corresponding to last row
        i = i + 2 * slice_size
        j = j + 1 + slice_size
    end
end

function update_prognostic_variables_from_state_vector!(
    model::SpeedyModel{T}, state::Vector{T}
) where {T <: AbstractFloat}
    start_index = 1
    dim_spectral = (model.parameters.spectral_truncation + 1)^2
    (; prognostic_variables) = model
    for name in LAYERED_VARIABLES
        if SpeedyWeather.has(model.model, name)
            # We only consider spectral coefficients for first leapfrog step (lf=1) to
            # define state
            layered_spectral_coefficients = getproperty(prognostic_variables, name)[1]
            for layer_index in 1:model.parameters.n_layers
                end_index = start_index + dim_spectral - 1
                update_spectral_coefficients_from_vector!(
                    layered_spectral_coefficients[:, layer_index],
                    view(state, start_index:end_index)
                )
                start_index = end_index + 1
            end
        end
    end
    if SpeedyWeather.has(model.model, :pres)
        update_spectral_coefficients_from_vector!(
            prognostic_variables.pres[1],
            view(state, start_index:start_index + dim_spectral - 1)
        )
    end
end

function update_vector_from_spectral_coefficients!(
    vector::AbstractVector{T},
    spectral_coefficients::SpeedyWeather.LowerTriangularMatrix{Complex{T}}, 
) where {T <: AbstractFloat}
    n_row, n_col = size(spectral_coefficients, as=Matrix)
    # First column of spectral_coefficients (order = m = 0) are real-valued and we skip
    # last row (degree = l = n_row - 1) as used only for computing meridional derivative
    # for vector valued fields. LowerTriangularMatrix allows vector (flat) indexing
    # skipping zero upper-triangular entries
    vector[1:n_row - 1] .= real(spectral_coefficients[1:n_row - 1])
    # vector index is i, spectral coefficient (flat) index is j
    i = n_row - 1
    j = n_row
    for col_index in 2:n_col
        # Slice corresponding to column has non-zero entries from col_index row and we
        # ignore last row as used only for computing meridional derivative for vector
        # valued fields
        slice_size = n_row - col_index
        # Reinterpret complex valued spectral coefficients to extract both real and
        # imaginary components
        vector[i + 1:i + 2 * slice_size] .= reinterpret(
            T, spectral_coefficients[j + 1:j + slice_size]
        )
        # Update vector and spectral coefficient indices, adding 1 offset
        # to latter to skip entries corresponding to last row
        i = i + 2 * slice_size
        j = j + 1 + slice_size
    end
end

function update_state_vector_from_prognostic_variables!(
    state::Vector{T}, model::SpeedyModel{T},
) where {T <: AbstractFloat}
    start_index = 1
    dim_spectral = (model.parameters.spectral_truncation + 1)^2
    (; prognostic_variables) = model
    for name in LAYERED_VARIABLES
        if SpeedyWeather.has(model.model, name)
            # We only consider spectral coefficients for first leapfrog step (lf=1) to
            # define state
            layered_spectral_coefficients = getproperty(prognostic_variables, name)[1]
            for layer_index in 1:model.parameters.n_layers
                end_index = start_index + dim_spectral - 1
                update_vector_from_spectral_coefficients!(
                    view(state, start_index:end_index),
                    layered_spectral_coefficients[:, layer_index]
                )
                start_index = end_index + 1
            end
        end
    end
    if SpeedyWeather.has(model.model, :pres)
        update_vector_from_spectral_coefficients!(
            view(state, start_index:start_index + dim_spectral - 1),
            prognostic_variables.pres[1]
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
    update_state_vector_from_prognostic_variables!(state, model)
end

function ParticleDA.update_state_deterministic!(
    state::Vector{T}, model::SpeedyModel{T}, time_index::Int
) where {T<:AbstractFloat}
    update_prognostic_variables_from_state_vector!(model, state)
    (; clock) = model.prognostic_variables
    (; time_stepping) = model.model
    SpeedyWeather.set_period!(clock, SpeedyWeather.Day(model.parameter.n_days))
    SpeedyWeather.initialize!(clock, time_stepping)
    clock.time += clock.n_timesteps * clock.Δt * (time_index - 1)
    clock.timestep_counter += clock.n_timesteps * (time_index - 1)
    SpeedyWeather.time_stepping!(
        model.prognostic_variables, model.diagnostic_variables, model.model
    )
    update_state_vector_from_prognostic_variables!(state, model)
end

function ParticleDA.update_state_stochastic!(    
    state::Vector{T}, model::SpeedyModel{T}, rng::G
) where {T<:AbstractFloat, G<:AbstractRNG}

end

function ParticleDA.sample_observation_given_state!(
    observation::Vector{T}, state::Vector{T}, model::SpeedyModel{T}, rng::G
) where {T<:AbstractFloat, G<:AbstractRNG}
    update_prognostic_variables_from_state_vector!(model, state)
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
