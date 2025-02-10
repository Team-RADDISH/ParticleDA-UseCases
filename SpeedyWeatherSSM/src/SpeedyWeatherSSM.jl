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

Base.@kwdef struct GaussianRandomFieldParameters{T<:AbstractFloat}
    output_scale::T = 1.
    length_scale::T = 0.1
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
    initial_state_grf_parameters::Dict{Symbol, GaussianRandomFieldParameters{T}} = Dict(
        :vor => GaussianRandomFieldParameters(; output_scale=5e-7),
        :div => GaussianRandomFieldParameters(; output_scale=5e-7),
        :temp => GaussianRandomFieldParameters(; output_scale=2e0),
        :humid => GaussianRandomFieldParameters(; output_scale=1e-5),
        :pres => GaussianRandomFieldParameters(; output_scale=2e-3),
    )
    state_noise_grf_parameters::Dict{Symbol, GaussianRandomFieldParameters{T}} = Dict(
        :vor => GaussianRandomFieldParameters(; output_scale=5e-8),
        :div => GaussianRandomFieldParameters(; output_scale=5e-8),
        :temp => GaussianRandomFieldParameters(; output_scale=2e-1),
        :humid => GaussianRandomFieldParameters(; output_scale=1e-6),
        :pres => GaussianRandomFieldParameters(; output_scale=2e-4),
    )
end

struct SpeedyModel{
    T<:AbstractFloat,
    G<:SpeedyWeather.AbstractSpectralGrid,
    M<:SpeedyWeather.AbstractModel,
    I<:SpeedyWeather.RingGrids.AbstractInterpolator,
    P<:SpeedyWeather.AbstractPrognosticVariables,
    D<:SpeedyWeather.AbstractDiagnosticVariables
}
    parameters::SpeedyParameters{T, M}
    spectral_grid::G
    model::M
    prognostic_variables::Vector{P}
    diagnostic_variables::Vector{D}
    variable_names::Tuple
    n_layered_variables::Int
    n_surface_variables::Int
    n_observed_points::Int
    observation_interpolator::I
    initial_state_grf_scale_factors::Dict{Symbol, Vector{T}}
    state_noise_grf_scale_factors::Dict{Symbol, Vector{T}}
end

function init(parameters::SpeedyParameters{T, M}, n_tasks::Int=1) where {
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
    SpeedyWeather.set_period!(
        prognostic_variables.clock, SpeedyWeather.Day(parameters.n_days)
    )
    SpeedyWeather.initialize!(prognostic_variables.clock, model.time_stepping)
    # We need separate copies of prognostic and diagnostic variables for each task to
    # allow independent parallel read-write access
    per_task_prognostic_variables = Vector{typeof(prognostic_variables)}(undef, n_tasks)
    per_task_diagnostic_variables = Vector{typeof(diagnostic_variables)}(undef, n_tasks)
    per_task_prognostic_variables[1] = prognostic_variables
    per_task_diagnostic_variables[1] = diagnostic_variables
    for t in 2:n_tasks
        per_task_prognostic_variables[t] = PrognosticVariables(spectral_grid, model)
        copy!(per_task_prognostic_variables[t], prognostic_variables)
        per_task_diagnostic_variables[t] = DiagnosticVariables(spectral_grid)
    end
    variable_names = SpeedyWeather.prognostic_variables(model)
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
    initial_state_grf_scale_factors = Dict(
        name => get_grf_coefficient_scale_factors(
            parameters.initial_state_grf_parameters[name],
            parameters.spectral_truncation,
            model.spectral_transform.norm_sphere
        )
        for name in variable_names
    )
    state_noise_grf_scale_factors = Dict(
        name => get_grf_coefficient_scale_factors(
            parameters.state_noise_grf_parameters[name],
            parameters.spectral_truncation,
            model.spectral_transform.norm_sphere
        )
        for name in variable_names
    )
    return SpeedyModel(
        parameters,
        spectral_grid,
        model,
        per_task_prognostic_variables,
        per_task_diagnostic_variables,
        variable_names,
        n_layered_variables,
        n_surface_variables,
        n_observed_points,
        observation_interpolator,
        initial_state_grf_scale_factors,
        state_noise_grf_scale_factors
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

function map_over_state_vector_slices(
    map_function::Function,
    state::AbstractVector{T},
    variable_names::Tuple,
    spectral_truncation::Int,
    n_layers::Int,
) where {T <: AbstractFloat}
    start_index = 1
    dim_spectral = (spectral_truncation + 1)^2
    for name in LAYERED_VARIABLES
        if name in variable_names
            for layer_index in 1:n_layers
                end_index = start_index + dim_spectral - 1
                map_function(
                    view(state, start_index:end_index),
                    name,
                    layer_index,
                )
                start_index = end_index + 1
            end
        end
    end
    for name in SURFACE_VARIABLES
        if name in variable_names
            map_function(
                view(state, start_index:start_index + dim_spectral - 1),
                name,
                nothing,
            )
        end
    end
end

function map_over_spectral_coefficients_and_state_vector_slices(
    map_function::Function,
    prognostic_variables::SpeedyWeather.PrognosticVariables{T},
    state::AbstractVector{T},
    variable_names::Tuple,
    leapfrog_step::Int
) where {T <: AbstractFloat}
    function outer_map_function(state_slice, name, layer_index)
        spectral_coefficients = getproperty(prognostic_variables, name)[leapfrog_step]
        if !isnothing(layer_index)
            spectral_coefficients = view(spectral_coefficients, :, layer_index)
        end
        map_function(spectral_coefficients, state_slice)
    end
    map_over_state_vector_slices(
        outer_map_function,
        state,
        variable_names,
        prognostic_variables.trunc,
        prognostic_variables.nlayers,
    )
end

function update_prognostic_variables_from_state_vector!(
    prognostic_variables::SpeedyWeather.PrognosticVariables{T},
    state::AbstractVector{T},
    variable_names::Tuple;
    leapfrog_step::Int = 1
) where {T <: AbstractFloat}
    spectral_truncation = prognostic_variables.trunc
    map_over_spectral_coefficients_and_state_vector_slices(
        (c, v) -> update_spectral_coefficients_from_vector!(c, v, spectral_truncation),
        prognostic_variables,
        state,
        variable_names,
        leapfrog_step
    )
end

function update_prognostic_variables_from_state_vector!(
    model::SpeedyModel{T}, state::AbstractVector{T}, task_index::Int
) where {T <: AbstractFloat}
    update_prognostic_variables_from_state_vector!(
        model.prognostic_variables[task_index],
        state,
        model.variable_names;
        leapfrog_step=1
    )
    # Zero coefficients for second leapfrog step (corresponding to initial state)
    update_prognostic_variables_from_state_vector!(
        model.prognostic_variables[task_index],
        Zeros(ParticleDA.get_state_dimension(model)),
        model.variable_names;
        leapfrog_step=2
    )
end

function update_vector_from_spectral_coefficients!(
    vector::AbstractVector{T},
    spectral_coefficients::AbstractVector{Complex{T}},
    spectral_truncation::Int;
    increment::Bool = false
) where {T <: AbstractFloat}
    n_row, n_col = spectral_truncation + 2, spectral_truncation + 1
    update! = increment ? (lhs, rhs) -> (lhs .+= rhs) : (lhs, rhs) -> (lhs .= rhs)
    # First column of spectral_coefficients (order = m = 0) are real-valued and we skip
    # last row (degree = l = n_row - 1) as used only for computing meridional derivative
    # for vector valued fields. LowerTriangularMatrix allows vector (flat) indexing
    # skipping zero upper-triangular entries
    @views update!(vector[1:n_row - 1], real(spectral_coefficients[1:n_row - 1]))
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
        @views update!(
            vector[i + 1:i + 2 * slice_size], 
            reinterpret(T, spectral_coefficients[j + 1:j + slice_size])
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
    spectral_truncation = prognostic_variables.trunc
    map_over_spectral_coefficients_and_state_vector_slices(
        (c, v) -> update_vector_from_spectral_coefficients!(v, c, spectral_truncation),
        prognostic_variables,
        state,
        variable_names,
        leapfrog_step
    )
end

function update_state_vector_from_prognostic_variables!(
    state::AbstractVector{T}, model::SpeedyModel{T}, task_index::Int
) where {T <: AbstractFloat}
    update_state_vector_from_prognostic_variables!(
        state,
        model.prognostic_variables[task_index],
        model.variable_names
    )
end

function update_clock_from_time_index!(clock::SpeedyWeather.Clock, time_index::Int)
    clock.time = clock.start + clock.n_timesteps * clock.Δt * (time_index - 1)
    clock.timestep_counter = clock.n_timesteps * (time_index - 1)
end

function update_clock_from_time_index!(
    model::SpeedyModel, time_index::Int, task_index::Int
)
    update_clock_from_time_index!(
        model.prognostic_variables[task_index].clock, time_index
    )
end

function get_observed_variable_field(
    diagnostic_variables::DiagnosticVariables, model::SpeedyModel
)
    observed_outer, observed_inner = model.parameters.observed_variable
    getfield(getfield(diagnostic_variables, observed_outer), observed_inner)
end

function update_prognostic_and_diagnostic_variables_from_state_vector!(
    model::SpeedyModel{T}, state::AbstractVector{T}, task_index::Int
) where {T <: AbstractFloat}
    update_prognostic_variables_from_state_vector!(model, state, task_index)
    SpeedyWeather.transform!(
        model.diagnostic_variables[task_index],
        model.prognostic_variables[task_index],
        1,
        model.model,
        initialize=true
    )
end

function add_noise_to_state_vector!(
    state::AbstractVector{T},
    spectral_truncation::Int,
    n_layers::Int,
    variable_names::Tuple,
    grf_scale_factors::Dict{Symbol, Vector{T}},
    rng::AbstractRNG
) where {T <: AbstractFloat}
    n_row, n_col = spectral_truncation + 2, spectral_truncation + 1
    spectral_coefficients = SpeedyWeather.LowerTriangularMatrix{Complex{T}}(
        undef, n_row, n_col
    )
    function map_function(state_slice, name, layer_index)
        generate_random_spectral_coefficients!(
            spectral_coefficients,
            spectral_truncation,
            grf_scale_factors[name],
            rng
        )
        update_vector_from_spectral_coefficients!(
            state_slice, spectral_coefficients, spectral_truncation; increment=true
        )
    end
    map_over_state_vector_slices(
        map_function,
        state,
        variable_names,
        spectral_truncation,
        n_layers,
    )
end

function get_grf_coefficient_scale_factors(
    parameters::GaussianRandomFieldParameters{T},
    spectral_truncation::Int,
    norm_sphere::T
) where {T <: AbstractFloat}
    el_max = spectral_truncation + 1
    (; output_scale, length_scale) = parameters
    norm_factor = norm_sphere * output_scale / sqrt(
        sum((2 * el + 1) * exp(-2 * length_scale^2 * el * (el + 1)) for el in 1:el_max)
    ) 
    scale_factors = zeros(T, el_max)
    # el = 1 case corresponds to mean - assume zero-mean
    for el in 2:el_max
        scale_factors[el] = norm_factor * exp(-length_scale^2 * el * (el + 1))
    end
    return scale_factors
end

function generate_random_spectral_coefficients!(
    spectral_coefficients::AbstractVector{Complex{T}},
    spectral_truncation::Int,
    scale_factors::AbstractVector{T},
    rng::AbstractRNG
) where {T <: AbstractFloat}
    el_m = 0
    @inbounds for m in 1:spectral_truncation + 1
        for el in m:spectral_truncation + 2
            el_m += 1
            # Don't generate coefficients in last row (used only for meridional deriv.)
            (el == spectral_truncation + 2) && continue 
            spectral_coefficients[el_m] = scale_factors[el] * randn(rng, Complex{T})
        end
    end
end

ParticleDA.get_state_eltype(model::SpeedyModel{T}) where {T<:AbstractFloat} = T

ParticleDA.get_observation_eltype(model::SpeedyModel{T}) where {T<:AbstractFloat} = T

function ParticleDA.sample_initial_state!(
    state::AbstractVector{T}, model::SpeedyModel{T}, rng::R, task_index::Int=1
) where {T<:AbstractFloat, R<:AbstractRNG}
    SpeedyWeather.initialize!(
        model.prognostic_variables[task_index],
        model.model.initial_conditions,
        model.model
    )
    update_state_vector_from_prognostic_variables!(state, model, task_index)
    add_noise_to_state_vector!(
        state,
        model.parameters.spectral_truncation,
        model.parameters.n_layers,
        model.variable_names,
        model.initial_state_grf_scale_factors,
        rng
    )
end

function ParticleDA.update_state_deterministic!(
    state::AbstractVector{T}, model::SpeedyModel{T}, time_index::Int, task_index::Int=1
) where {T<:AbstractFloat}
    update_prognostic_variables_from_state_vector!(model, state, task_index)
    update_clock_from_time_index!(model, time_index, task_index)
    SpeedyWeather.time_stepping!(
        model.prognostic_variables[task_index],
        model.diagnostic_variables[task_index],
        model.model
    )
    update_state_vector_from_prognostic_variables!(state, model, task_index)
end

function ParticleDA.update_state_stochastic!(    
    state::AbstractVector{T}, model::SpeedyModel{T}, rng::G, task_index::Int=1
) where {T<:AbstractFloat, G<:AbstractRNG}
    add_noise_to_state_vector!(
        state,
        model.parameters.spectral_truncation,
        model.parameters.n_layers,
        model.variable_names,
        model.state_noise_grf_scale_factors,
        rng
    )
end

function ParticleDA.get_observation_mean_given_state!(
    observation_mean::AbstractVector{T},
    state::AbstractVector{T},
    model::SpeedyModel{T},
    task_index::Int=1
) where {T<:AbstractFloat}
    update_prognostic_and_diagnostic_variables_from_state_vector!(
        model, state, task_index
    )
    observed_field_grid = get_observed_variable_field(
        model.diagnostic_variables[task_index], model
    )
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

const HDF5FileOrGroup = Union{HDF5.File, HDF5.Group}

function write!(
    group::HDF5FileOrGroup, key::String, value::Union{Number, String, Array}
)
    attributes(group)[key] = value
end

function write!(group::HDF5FileOrGroup, key::String, value::Union{Symbol, DateTime})
    attributes(group)[key] = string(value)
end

function write!(group::HDF5FileOrGroup, key::String, value::Type)
    attributes(group)[key] = string(nameof(value))
end

function write!(
    group::HDF5FileOrGroup, key::String, value::NTuple{N, T}
) where {N, T <: Union{Number, String, Symbol, Type}}
    subgroup = create_group(group, key)
    for (index, val) in enumerate(value)
        write!(subgroup, string(index), val)
    end
end

function write!(group::HDF5FileOrGroup, key::String, value::Dict)
    subgroup = create_group(group, key)
    for (k, v) in value
        write!(subgroup, string(k), v)
    end
end

function write!(group::HDF5FileOrGroup, key::String, value)
    subgroup = create_group(group, key)
    for name in fieldnames(typeof(value))
        write!(subgroup, string(name), getfield(value, name))
    end
end

function ParticleDA.write_model_metadata(file::HDF5.File, model::SpeedyModel)
    write!(file, "parameters", model.parameters)
end

end
