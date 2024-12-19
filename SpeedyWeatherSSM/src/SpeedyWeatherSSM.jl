module SpeedyWeatherSSM

using FillArrays
using HDF5  
using ParticleDA
using Random
using SpeedyWeather
using SpeedyWeather.RingGrids

LAYERED_VARIABLES = (:vor, :div, :temp, :humid)
SURFACE_VARIABLES = (:pres,)

function equispaced_lat_lon_grid(T, n_lat, n_lon)
    lat_interval = 180 / n_lat
    lon_interval = 360 / n_lon
    lat_range = (-90 + lat_interval/2):lat_interval:90
    lon_range = (-180 + lon_interval/2):lon_interval:180
    lat_lon_pairs = [(T(lat), T(lon)) for lat in lat_range for lon in lon_range]
    collect(reshape(reinterpret(T, lat_lon_pairs), (2, :)))
end

Base.@kwdef struct SpeedyParameters{T<:AbstractFloat, M<:SpeedyWeather.AbstractModel}
    spectral_truncation::Int = 31
    n_layers::Int = 8
    n_days::T = 0.25
    start_date::DateTime = DateTime(2000, 1, 1)
    float_type::Type{T} = Float64
    model_type::Type{M} = PrimitiveWetModel
    observed_variable::Tuple{Symbol, Symbol} = (:physics, :precip_large_scale)
    observed_coordinates::Matrix{T} = equispaced_lat_lon_grid(float_type, 6, 12)
    observation_noise_std::T = 0.1
end

struct SpeedyModel{
    T<:AbstractFloat,
    G<:SpeedyWeather.AbstractSpectralGrid,
    M<:SpeedyWeather.AbstractModel,
    I<:SpeedyWeather.RingGrids.AbstractInterpolator
}
    parameters::SpeedyParameters{T, M}
    spectral_grid::G
    model::M
    prognostic_variables::PrognosticVariables{T}
    diagnostic_variables::DiagnosticVariables{T}
    n_layered_variables::Int
    n_surface_variables::Int
    n_observed_points::Int
    observation_interpolator::I
end

function init(parameters::SpeedyParameters{T, M}) where {
    T<:AbstractFloat, M<:SpeedyWeather.AbstractModel
}
    spectral_grid = SpectralGrid(;
        NF=T, trunc=parameters.spectral_truncation, nlayers=parameters.n_layers
    )
    feedback = SpeedyWeather.Feedback(; verbose=false)
    model = M(; spectral_grid, feedback)
    model.output.active = false
    simulation = initialize!(model; time=parameters.start_date)
    (; prognostic_variables, diagnostic_variables) = simulation
    n_layered_variables = count(
        SpeedyWeather.has(model, var) for var in LAYERED_VARIABLES
    )
    n_surface_variables = count(
        SpeedyWeather.has(model, var) for var in SURFACE_VARIABLES
    )
    n_observed_points = size(parameters.observed_coordinates, 2)
    observation_interpolator = SpeedyWeather.AnvilInterpolator(
        T, spectral_grid.Grid, spectral_grid.nlat_half, n_observed_points
    )
    SpeedyWeather.RingGrids.update_locator!(
        observation_interpolator,
        parameters.observed_coordinates[1, :],
        parameters.observed_coordinates[2, :]
    )
    SpeedyWeather.set_period!(
        prognostic_variables.clock, SpeedyWeather.Day(parameters.n_days)
    )
    SpeedyWeather.initialize!(prognostic_variables.clock, model.time_stepping)
    return SpeedyModel(
        parameters,
        spectral_grid,
        model,
        prognostic_variables,
        diagnostic_variables,
        n_layered_variables,
        n_surface_variables,
        n_observed_points,
        observation_interpolator
    )
end

function ParticleDA.get_state_dimension(model::SpeedyModel)
    (model.parameters.spectral_truncation + 1)^2 * (
        model.parameters.n_layers * model.n_layered_variables + model.n_surface_variables
    )
end

function ParticleDA.get_observation_dimension(model::SpeedyModel)
    model.n_observed_points
end

function update_spectral_coefficients_from_vector!(
    spectral_coefficients::AbstractVector{Complex{T}}, 
    vector::AbstractVector{T},
    spectral_truncation::Int
) where {T <: AbstractFloat}
    n_row, n_col = spectral_truncation + 2, spectral_truncation + 1
    # First column of spectral_coefficients (order = m = 0) are real-valued and we skip
    # last row (degree = l = n_row - 1) as used only for computing meridional derivative
    # for vector valued fields. LowerTriangularMatrix allows vector (flat) indexing
    # skipping zero upper-triangular entries
    spectral_coefficients[1:n_row - 1] .= vector[1:n_row - 1]
    # Zero entry corresponding to last row as not used for scalar fields
    spectral_coefficients[n_row] = 0
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
        # Zero entry corresponding to last row as not use for scalar fields
        spectral_coefficients[j + slice_size + 1] = 0
        # Update vector and spectral coefficient indices, adding 1 offset
        # to latter to skip entries corresponding to last row
        i = i + 2 * slice_size
        j = j + 1 + slice_size
    end
end

function update_prognostic_variables_from_state_vector!(
    prognostic_variables::SpeedyWeather.PrognosticVariables{T},
    state::AbstractVector{T},
    variable_names::Tuple;
    leapfrog_step::Int = 1
) where {T <: AbstractFloat}
    start_index = 1
    spectral_truncation = prognostic_variables.trunc
    dim_spectral = (spectral_truncation + 1)^2
    for name in LAYERED_VARIABLES
        if name in variable_names
            layered_spectral_coefficients = getproperty(
                prognostic_variables, name
            )[leapfrog_step]
            for layer_index in 1:prognostic_variables.nlayers
                end_index = start_index + dim_spectral - 1
                update_spectral_coefficients_from_vector!(
                    view(layered_spectral_coefficients, :, layer_index),
                    view(state, start_index:end_index),
                    spectral_truncation
                )
                start_index = end_index + 1
            end
        end
    end
    for name in SURFACE_VARIABLES
        if name in variable_names
            spectral_coefficients = getproperty(
                prognostic_variables, name
            )[leapfrog_step]
            update_spectral_coefficients_from_vector!(
                spectral_coefficients,
                view(state, start_index:start_index + dim_spectral - 1),
                spectral_truncation
            )
        end
    end
end

function update_prognostic_variables_from_state_vector!(
    model::SpeedyModel{T}, state::AbstractVector{T}
) where {T <: AbstractFloat}
    update_prognostic_variables_from_state_vector!(
        model.prognostic_variables,
        state,
        SpeedyWeather.prognostic_variables(model.model);
        leapfrog_step=1
    )
    # Zero cofficients for second leapfrog step (corresponding to initial state)
    update_prognostic_variables_from_state_vector!(
        model.prognostic_variables,
        Zeros(ParticleDA.get_state_dimension(model)),
        SpeedyWeather.prognostic_variables(model.model);
        leapfrog_step=2
    )
end

function update_vector_from_spectral_coefficients!(
    vector::AbstractVector{T},
    spectral_coefficients::AbstractVector{Complex{T}},
    spectral_truncation::Int
) where {T <: AbstractFloat}
    n_row, n_col = spectral_truncation + 2, spectral_truncation + 1
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
    state::AbstractVector{T},
    prognostic_variables::SpeedyWeather.PrognosticVariables{T},
    variable_names::Tuple;
    leapfrog_step::Int = 1
) where {T <: AbstractFloat}
    start_index = 1
    spectral_truncation = prognostic_variables.trunc
    dim_spectral = (spectral_truncation + 1)^2
    for name in LAYERED_VARIABLES
        if name in variable_names
            layered_spectral_coefficients = getproperty(
                prognostic_variables, name
            )[leapfrog_step]
            for layer_index in 1:prognostic_variables.nlayers
                end_index = start_index + dim_spectral - 1
                update_vector_from_spectral_coefficients!(
                    view(state, start_index:end_index),
                    view(layered_spectral_coefficients, :, layer_index),
                    spectral_truncation
                )
                start_index = end_index + 1
            end
        end
    end
    for name in SURFACE_VARIABLES
        if name in variable_names
            spectral_coefficients = getproperty(
                prognostic_variables, name
            )[leapfrog_step]
            update_vector_from_spectral_coefficients!(
                view(state, start_index:start_index + dim_spectral - 1),
                spectral_coefficients,
                spectral_truncation
            )
        end
    end
end

function update_state_vector_from_prognostic_variables!(
    state::AbstractVector{T}, model::SpeedyModel{T}
) where {T <: AbstractFloat}
    update_state_vector_from_prognostic_variables!(
        state,
        model.prognostic_variables,
        SpeedyWeather.prognostic_variables(model.model)
    )
end

function update_clock_from_time_index!(clock::SpeedyWeather.Clock,time_index::Int)
    clock.time = clock.start + clock.n_timesteps * clock.Δt * (time_index - 1)
    clock.timestep_counter = clock.n_timesteps * (time_index - 1)
end

function update_clock_from_time_index!(model::SpeedyModel, time_index::Int)
    update_clock_from_time_index!(model.prognostic_variables.clock, time_index)
end

function get_observed_variable_field(
    diagnostic_variables::DiagnosticVariables, model::SpeedyModel
)
    observed_outer, observed_inner = model.parameters.observed_variable
    getfield(getfield(diagnostic_variables, observed_outer), observed_inner)
end

function update_prognostic_and_diagnostic_variables_from_state_vector!(
    model::SpeedyModel{T}, state::AbstractVector{T}
) where {T <: AbstractFloat}
    update_prognostic_variables_from_state_vector!(model, state)
    SpeedyWeather.transform!(
        model.diagnostic_variables,
        model.prognostic_variables,
        1,
        model.model,
        initialize=true
    )
end

ParticleDA.get_state_eltype(model::SpeedyModel{T}) where {T<:AbstractFloat} = T

ParticleDA.get_observation_eltype(model::SpeedyModel{T}) where {T<:AbstractFloat} = T

function ParticleDA.sample_initial_state!(
    state::AbstractVector{T}, model::SpeedyModel{T}, rng::R, task_index::Int=1
) where {T<:AbstractFloat, R<:AbstractRNG}
    initial_conditions = model.model.initial_conditions
    SpeedyWeather.initialize!(
        model.prognostic_variables, initial_conditions, model.model
    )
    update_state_vector_from_prognostic_variables!(state, model)
end

function ParticleDA.update_state_deterministic!(
    state::AbstractVector{T}, model::SpeedyModel{T}, time_index::Int, task_index::Int=1
) where {T<:AbstractFloat}
    update_prognostic_variables_from_state_vector!(model, state)
    update_clock_from_time_index!(model, time_index)
    SpeedyWeather.time_stepping!(
        model.prognostic_variables, model.diagnostic_variables, model.model
    )
    update_state_vector_from_prognostic_variables!(state, model)
end

function ParticleDA.update_state_stochastic!(    
    state::AbstractVector{T}, model::SpeedyModel{T}, rng::G, task_index::Int=1
) where {T<:AbstractFloat, G<:AbstractRNG}

end

function ParticleDA.get_observation_mean_given_state!(
    observation_mean::AbstractVector{T},
    state::AbstractVector{T},
    model::SpeedyModel{T},
    task_index::Int=1
) where {T<:AbstractFloat}
    update_prognostic_and_diagnostic_variables_from_state_vector!(model, state)
    observed_field_grid = get_observed_variable_field(model.diagnostic_variables, model)
    SpeedyWeather.interpolate!(
        observation_mean, observed_field_grid, model.observation_interpolator
    )
end

function ParticleDA.sample_observation_given_state!(
    observation::AbstractVector{T},
    state::AbstractVector{T},
    model::SpeedyModel{T},
    rng::G,
    task_index::Int=1
) where {T<:AbstractFloat, G<:AbstractRNG}
    ParticleDA.get_observation_mean_given_state!(observation, state, model, task_index)
    observation .+= (
        model.parameters.observation_noise_std 
        * randn(rng, T, ParticleDA.get_observation_dimension(model))
    )
end

function ParticleDA.get_log_density_observation_given_state(
    observation::AbstractVector{T},
    state::AbstractVector{T},
    model::SpeedyModel{T},
    task_index::Int=1
) where {T<:AbstractFloat}
    observation_mean = Vector{T}(undef, ParticleDA.get_observation_dimension(model))
    ParticleDA.get_observation_mean_given_state!(
        observation_mean, state, model, task_index
    )
    return (
        -sum((observation - observation_mean).^2) 
        / (2 * model.parameters.observation_noise_std^2)
    )

end

function ParticleDA.write_model_metadata(file::HDF5.File, model::SpeedyModel)
    group_name = "parameters"
    if !haskey(file, group_name)
        group = create_group(file, group_name)
        for field in fieldnames(typeof(model.parameters))
            value = getfield(model.parameters, field)
            isa(value, Type) && (value = string(nameof(value)))
            isa(value, Tuple{Symbol, Symbol}) && (value = join(map(String, value), "."))
            isa(value, DateTime) && (value = string(value))
            HDF5.attributes(group)[string(field)] = value
        end
    else
        @warn "Write failed, group $group_name already exists in  $(file.filename)!"
    end
end

end
