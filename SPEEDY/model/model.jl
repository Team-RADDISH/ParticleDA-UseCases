module Model

using ParticleDA

using Random, Distributions, Base.Threads, GaussianRandomFields, HDF5
# using Default_params
using DelimitedFiles
using MPI
using Dates
using FortranFiles
using PDMats

# using .Default_params
include("speedy.jl")
using .SPEEDY

include("grf.jl")
using .Periodic_Cov
"""
    ModelParameters()

Parameters for the model. Keyword arguments:

* `IDate::String` : Start date for the simulations in the format: YYYYmmddHH
* `endDate::String` : End date for the simulations in the format: YYYYmmddHH
* `Hinc::Int` : Hourly increment
* `obs_network::String` : Location of observations (real or uniform)
* `nobs::Int` : Number of observation stations
* `lambda::AbstractFloat` : Length scale for Matérn covariance kernel in background noise
* `nu::AbstractFloat` : Smoothess parameter for Matérn covariance kernel in background noise
* `sigma::AbstractFloat` : Marginal standard deviation for Matérn covariance kernel in background noise
* `lambda_initial_state::AbstractFloat` : Length scale for Matérn covariance kernel in initial state of particles
* `nu_initial_state::AbstractFloat` : Smoothess parameter for Matérn covariance kernel in initial state of particles
* `sigma_initial_state::AbstractFloat` : Marginal standard deviation for Matérn covariance kernel in initial state of particles
* `padding::Int` : Min padding for circulant embedding gaussian random field generator
* `primes::Int`: Whether the size of the minimum circulant embedding of the covariance matrix can be written as a product of small primes (2, 3, 5 and 7). Default is `true`.
* `obs_noise_std::Float`: Standard deviation of noise added to observations of the true state
* `particle_dump_file::String`: file name for dump of particle state vectors
* `particle_dump_time::Int`: list of (one more more) time steps to dump particle states
* `SPEEDY::String` : Path to the SPEEDY directory
* `output_folder::String` : Output folder
* `station_filename::String` : Path to the station file which defines the observation locations
* `nature_dir::String`: Path to the directory with the ouputs of the nature run
* `observed_indices::Vector`: Vector containing the indices of the observed values in the state vector
* `nlon::Int`: Number of points in the longitude direction
* `nlat::Int`: Number of points in the latitude direction
* `lon_length::AbstractFloat`: Domain size in the lon direction
* `lat_length::AbstractFloat`: Domain size in the lat direction
* `dx::AbstractFloat` : Distance between grid points in the lon direction
* `dy::AbstractFloat` : Distance between grid points in the lat direction
* `nlev::Int`: Number of points in the vertical direction
* `n_state_var::Int`: Number of variables in the state vector
"""

Base.@kwdef struct ModelParameters{T<:AbstractFloat}
    # Initial date
    IDate::String=""
    # Incremental date
    endDate::String=""
    # Hour interval
    Hinc::Int = 6
    # Ensemble initialisation
    ensDate::String=""
    # Choose observation network (choose "real" or "uniform")
    obs_network::String = "uniform"
    # Number of obs stations
    nobs::Int = 50

    lambda::Vector{T} = [1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0]
    nu::Vector{T} = [2.5, 2.5, 2.5, 2.5, 2.5]
    sigma::Vector{T} = [0.1, 0.1, 0.1, 0.00001, 1000.0]

    lambda_initial_state::Vector{T} = [1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0]
    nu_initial_state::Vector{T} = [2.5, 2.5, 2.5, 2.5, 2.5]
    sigma_initial_state::Vector{T} = [0.1, 0.1, 0.1, 0.0001, 1000.0]
    
    padding::Int = 100
    primes::Bool = true

    state_prefix::String = "data"
    title_avg::String = "avg"
    title_var::String = "var"
    title_syn::String = "syn"
    title_grid::String = "grid"
    title_stations::String = "stations"
    title_params::String = "params"
    particle_dump_file::String = "particle_dump.h5"
    particle_dump_time = [-1]
    #Path to the the local speedy directory
    SPEEDY::String = ""
    station_filename::String = joinpath(SPEEDY, "obs", "networks", obs_network * ".txt")
    nature_dir::String = joinpath(SPEEDY, "DATA", "nature")
    # Output folders
    output_folder::String = joinpath(pwd())
    guess_folder::String = joinpath(output_folder, "DATA", "ensemble", "gues")
    anal_folder::String = joinpath(output_folder, "DATA", "ensemble", "anal")

    obs_noise_std::Vector{T} = [1000.0]
    # Observed indices
    observed_state_var_indices::Vector{Int} = [33]
    n_assimilated_var::Int = length(observed_state_var_indices)
    # Grid dimensions
    nlon::Int = 96
    nlat::Int = 48
    nlev::Int = 8
    lon_length::T = 360.0
    lat_length::T = 180.0
    z_length::T = 1.0
    dx::T = lon_length / (nlon - 1)
    dy::T = lat_length / (nlat - 1)
    dz::T = z_length / (nlev - 1)

    nij0::Int = nlon*nlat
    n_2d::Int = 2
    n_3d::Int = 4
    n_state_var::Int = n_3d*nlev + n_2d

end
const SPEEDY_DATE_FORMAT = "YYYYmmddHH"
get_float_eltype(::Type{<:ModelParameters{T}}) where {T} = T
get_float_eltype(p::ModelParameters) = get_float_eltype(typeof(p))

struct RandomField{F<:GaussianRandomField}
    grf::F
    # xi::Array{T, 3}
    # w::Array{Complex{T}, 3}
    # z::Array{T, 3}
end

struct StateFieldMetadata
    name::String
    unit::String
    description::String
end

const STATE_FIELDS_METADATA = [
    StateFieldMetadata("u", "m/s", "Ocean surface velocity x-component"),
    StateFieldMetadata("v", "m/s", "Ocean surface velocity y-component"),
    StateFieldMetadata("T", "K", "Temperature"),
    StateFieldMetadata("Q", "kg/kg", "Specific Humidity"),
    StateFieldMetadata("ps", "Pa", "Surface Pressure"),
    StateFieldMetadata("rain", "mm/hr", "Rain"),
]

struct ModelData{T <: Real, U <: Real, G <: GaussianRandomField, S <: DateTime}
    model_params::ModelParameters{T}
    station_grid_indices::Matrix{Int}
    field_buffer::Array{T, 4}
    observation_buffer::Matrix{U}
    initial_state_grf::Vector{RandomField{G}}
    state_noise_grf::Vector{RandomField{G}}
    model_matrices::SPEEDY.Matrices{T}
    dates::Vector{S}
end

function ParticleDA.get_params(T::Type{ModelParameters}, user_input_dict::Dict)

    for key in ("lambda", "nu", "sigma", "lambda_initial_state", "nu_initial_state", "sigma_initial_state")
        if haskey(user_input_dict, key) && !isa(user_input_dict[key], Vector)
            user_input_dict[key] = fill(user_input_dict[key], 3)
        end
    end

    user_input = (; (Symbol(k) => v for (k,v) in user_input_dict)...)
    params = T(;user_input...)

end

function step_datetime(idate::String,dtdate::String)
    new_idate = Dates.format(DateTime(idate, SPEEDY_DATE_FORMAT) + Dates.Hour(6), SPEEDY_DATE_FORMAT)
    new_dtdate = Dates.format(DateTime(dtdate, SPEEDY_DATE_FORMAT) + Dates.Hour(6), SPEEDY_DATE_FORMAT)
    return new_idate,new_dtdate
end

function step_ens(ensdate::String, Hinc::Int, rng::Random.AbstractRNG)
    ens_end = Dates.format(DateTime(ensdate, SPEEDY_DATE_FORMAT) + Dates.Month(1), SPEEDY_DATE_FORMAT)
    diff = length(DateTime(ensdate, SPEEDY_DATE_FORMAT):Hour(Hinc):DateTime(ens_end, SPEEDY_DATE_FORMAT))
    rand_int = rand(rng, 0:diff-1)*Hinc
    new_idate = Dates.format(DateTime(ensdate, SPEEDY_DATE_FORMAT) + Dates.Hour(rand_int), SPEEDY_DATE_FORMAT)
    return new_idate
end

function get_obs!(
    observations::AbstractVector{T},
    state::AbstractArray{T, 3},
    params::ModelParameters,
    station_grid_indices::AbstractMatrix,
) where T
    get_obs!(
        observations, state, params.observed_state_var_indices, station_grid_indices
    )
end

# Return observation data at stations from given model state
function get_obs!(
    observations::AbstractVector{T},
    state::AbstractArray{T, 3},
    observed_state_var_indices::AbstractVector{Int},
    station_grid_indices::AbstractMatrix{Int},
) where T
    @assert (
        length(observations) 
        == size(station_grid_indices, 1) * length(observed_state_var_indices) 
    )
    n = 1
    for k in observed_state_var_indices
        for (i, j) in eachrow(station_grid_indices)
            observations[n] = state[i, j, k]
            n += 1
        end
    end
end

function speedy_update!(SPEEDY::String,
    output::String,
    YMDH::String,
    TYMDH::String,
    rank::String,
    particle::String)

    # Path to the bash script which carries out the forecast
    forecast = joinpath(@__DIR__, "dafcst.sh")
    # Bash script call to speedy
    run(`$forecast $SPEEDY $output $YMDH $TYMDH $rank $particle`)
end

function get_axes(params::ModelParameters)

    return get_axes(params.nlon, params.nlat, params.nlev)

end
function get_axes(ix::Int, iy::Int, iz::Int)
    x = LinRange(0, 2*π, ix)
    y = LinRange(0, π, iy)
    z = LinRange(0, 1, iz)

    return x,y,z
end


function init_gaussian_random_field_generator(params::ModelParameters)

    x, y, z = get_axes(params)
    return init_gaussian_random_field_generator(params.lambda,params.nu, params.sigma, x, y)

end

# Initialize a gaussian random field generating function using the Matern covariance kernel
# and circulant embedding generation method
# TODO: Could generalise this
function init_gaussian_random_field_generator(lambda::Vector{T},
                                              nu::Vector{T},
                                              sigma::Vector{T},
                                              xpts::AbstractVector{T},
                                              ypts::AbstractVector{T}) where T

    # Let's limit ourselves to two-dimensional fields
    dim = 2
    function _generate(l, n, s)
        cov = Periodic_Cov.CovarianceFunction(dim, Periodic_Cov.CustomDistanceCovarianceStructure(Periodic_Cov.SphericalDistance(), Matern(l, n, σ = s)))
        grf = GaussianRandomField(cov, Spectral(), xpts, ypts)
        v = grf.data.eigenval
        xi = Array{eltype(grf.cov)}(undef, size(v)..., nthreads())
        w = Array{complex(float(eltype(grf.cov)))}(undef, size(v)..., nthreads())
        z = Array{eltype(grf.cov)}(undef, length.(grf.pts)..., nthreads())
        RandomField(grf)
    end

    return [_generate(l, n, s) for (l, n, s) in zip(lambda, nu, sigma)]
end

# Get a random sample from random_field_generator using random number generator rng
function sample_gaussian_random_field!(field::AbstractArray{T,2},
    random_field_generator::RandomField,
    rng::Random.AbstractRNG) where T

    field .= GaussianRandomFields.sample(random_field_generator.grf)
end

# Add a gaussian random field to the height in the state vector of all particles
function add_random_field!(
    state_fields::AbstractArray{T, 3},
    field_buffer::AbstractMatrix{T},
    generators::Vector{<:RandomField},
    rng::Random.AbstractRNG,) where T
    # @show length(generators)
    sample_gaussian_random_field!(field_buffer, generators[1], rng)
    state_fields[:, :, 33] .+= field_buffer
    # for ivar in 1:33
    #     if ivar < 9
    #         sample_gaussian_random_field!(field_buffer, generators[1], rng)
    #         state_fields[:, :, ivar] .+= field_buffer
    #     elseif 9 <= ivar < 17
    #         sample_gaussian_random_field!(field_buffer, generators[2], rng)
    #         state_fields[:, :, ivar] .+= field_buffer 
    #     elseif 17 <= ivar < 25 
    #         sample_gaussian_random_field!(field_buffer, generators[3], rng)
    #         state_fields[:, :, ivar] .+= field_buffer 
    #     elseif 25 <= ivar < 33
    #         sample_gaussian_random_field!(field_buffer, generators[4], rng)
    #         state_fields[:, :, ivar] .+= field_buffer 
    #     elseif ivar == 33
    #         sample_gaussian_random_field!(field_buffer, generators[5], rng)
    #         state_fields[:, :, ivar] .+= field_buffer 
    #     end
    # end
end


function add_noise!(vec::AbstractVector{T}, rng::Random.AbstractRNG, cov::AbstractPDMat{T}) where T
    vec .+= rand(rng, MvNormal(cov))
end

function ParticleDA.sample_initial_state!(
    state::AbstractVector{T},
    model_data::ModelData, 
    rng::Random.AbstractRNG,
) where T
    state_fields = reshape(
        state, 
        (
            model_data.model_params.nlon, 
            model_data.model_params.nlat, 
            model_data.model_params.n_state_var, 
        )
    )
    # Set true initial state
    # Read in the initial nature run - just surface pressure
    # read_grd!(@view(state_fields[:, :, :]), joinpath(model_data.model_params.nature_dir, model_data.model_params.IDate * ".grd"), model_data.model_params.nlon, model_data.model_params.nlat, model_data.model_params.nlev)
    ### Read in arbitrary nature run files for the initial conditions
    dummy_date = step_ens(model_data.model_params.ensDate, model_data.model_params.Hinc, rng)
    read_grd!(@view(state_fields[:, :, :]), joinpath(model_data.model_params.nature_dir, dummy_date * ".grd"), model_data.model_params.nlon, model_data.model_params.nlat, model_data.model_params.nlev)
    
    return state
end



function get_station_grid_indices(params::ModelParameters)
    return get_station_grid_indices(
        params.station_filename,
        params.nobs
    )
end

function get_station_grid_indices(
    filename::String,
    nobs::T
) where T
    station_grid_indices = Matrix{Int}(undef, nobs, 2)
    open(filename,"r") do f
        readline(f)
        readline(f)
        count = 0
        ind = 1 
        for line in eachline(f)
            if count%10 == 0
                station_grid_indices[ind, 1] = parse(Int64,split(line)[1])
                station_grid_indices[ind, 2] = parse(Int64,split(line)[2])
                ind += 1
            end
            count = count + 1
        end
    end
    return station_grid_indices
end


ParticleDA.get_state_dimension(d::ModelData) = (
    d.model_params.nlon * d.model_params.nlat * d.model_params.n_state_var
)

ParticleDA.get_observation_dimension(d::ModelData) = (
    size(d.station_grid_indices, 1) * length(d.model_params.observed_state_var_indices)
)

ParticleDA.get_state_eltype(::Type{<:ModelData{T, U, G}}) where {T, U, G} = T
ParticleDA.get_state_eltype(d::ModelData) = ParticleDA.get_state_eltype(typeof(d))

ParticleDA.get_observation_eltype(::Type{<:ModelData{T, U, G}}) where {T, U, G} = U
ParticleDA.get_observation_eltype(d::ModelData) = ParticleDA.get_observation_eltype(typeof(d))

function ParticleDA.get_covariance_observation_noise(
    d::ModelData, state_index_1::CartesianIndex, state_index_2::CartesianIndex
)
    x_index_1, y_index_1, var_index_1 = state_index_1.I
    x_index_2, y_index_2, var_index_2 = state_index_2.I

    if (x_index_1 == x_index_2 && y_index_1 == y_index_2)
        return (d.model_params.obs_noise_std[var_index_1]^2)
    else
        return 0.
    end
end

function ParticleDA.get_covariance_observation_noise(d::ModelData)
    return PDiagMat(
        ParticleDA.get_observation_dimension(d), 
        repeat(d.model_params.obs_noise_std.^2, inner=ParticleDA.get_observation_dimension(d))
    )
end


function flat_state_index_to_cartesian_index(
    model_params::ModelParameters, flat_index::Integer
)
    n_grid = model_params.nlon * model_params.nlat
    state_var_index, flat_grid_index = fldmod1(flat_index, n_grid)
    grid_y_index, grid_x_index = fldmod1(flat_grid_index, model_params.nlon)
    return CartesianIndex(grid_x_index, grid_y_index, state_var_index)
end

function grid_index_to_grid_point(
    model_params::ModelParameters, grid_index::Tuple{T, T}
) where {T <: Integer}
    return [
        (grid_index[1] - 1) * model_params.dx, (grid_index[2] - 1) * model_params.dy
    ]
end

function observation_index_to_cartesian_state_index(
    model_params::ModelParameters, station_grid_indices::AbstractMatrix, observation_index::Integer
)
    n_station = size(station_grid_indices,1)
    state_var_index, station_index = fldmod1(observation_index, n_station)
    return CartesianIndex(
        station_grid_indices[station_index, :]..., state_var_index
    )
end

function ParticleDA.get_covariance_state_noise(
    model_data::ModelData, state_index_1::Integer, state_index_2::Integer
)
    return ParticleDA.get_covariance_state_noise(
        model_data, 
        flat_state_index_to_cartesian_index(model_data.model_params, state_index_1),
        flat_state_index_to_cartesian_index(model_data.model_params, state_index_2),
    )
end

function ParticleDA.get_covariance_state_noise(
    model_data::ModelData, state_index_1::CartesianIndex, state_index_2::CartesianIndex
)
    x_index_1, y_index_1, var_index_1 = state_index_1.I
    x_index_2, y_index_2, var_index_2 = state_index_2.I
    if var_index_1 == var_index_2
        grid_point_1 = grid_index_to_grid_point(
            model_data.model_params, (x_index_1, y_index_1)
        )
        grid_point_2 = grid_index_to_grid_point(
            model_data.model_params, (x_index_2, y_index_2)
        )
        covariance_structure = model_data.state_noise_grf[var_index_1].grf.cov.cov.cov
        return covariance_structure.σ^2 * apply(
            covariance_structure, abs.(grid_point_1 .- grid_point_2)
        )
    else
        return 0.
    end
end

function ParticleDA.get_covariance_observation_observation_given_previous_state(
    model_data::ModelData, observation_index_1::Integer, observation_index_2::Integer
)
    observation_1 = observation_index_to_cartesian_state_index(
            model_data.model_params, 
            model_data.station_grid_indices, 
            observation_index_1
        )

    observation_2 = observation_index_to_cartesian_state_index(
        model_data.model_params, 
            model_data.station_grid_indices, 
            observation_index_2
    )
    return ParticleDA.get_covariance_state_noise(
        model_data,
        observation_1,
        observation_2,
    ) + ParticleDA.get_covariance_observation_noise(
        model_data, observation_1, observation_2
    )
end

function ParticleDA.get_covariance_state_observation_given_previous_state(
    model_data::ModelData, state_index::Integer, observation_index::Integer
)
    return ParticleDA.get_covariance_state_noise(
        model_data,
        flat_state_index_to_cartesian_index(model_data.model_params, state_index),
        observation_index_to_cartesian_state_index(
            model_data.model_params, model_data.station_grid_indices, observation_index
        ),
    )
end
                                                         
function ParticleDA.get_state_indices_correlated_to_observations(model_data::ModelData)
    n_grid = model_data.model_params.nlon * model_data.model_params.nlat
    return vcat(
        (
            (i - 1) * n_grid + 1 : i * n_grid 
            for i in model_data.model_params.observed_state_var_indices
        )...
    )
end

function init(model_params_dict::Dict)

    model_params = ParticleDA.get_params(
        ModelParameters, get(model_params_dict, "speedy", Dict())
    )
    
    station_grid_indices = get_station_grid_indices(model_params)
     
    T = get_float_eltype(model_params)
    n_stations = size(station_grid_indices, 1)
    n_observations = n_stations * length(model_params.observed_state_var_indices)
    
    # Buffer array to be used in the tsunami update
    field_buffer = Array{T}(undef, model_params.nlon, model_params.nlat, 2, nthreads())
    
    # Buffer array to be used in computing observation mean
    observation_buffer = Array{T}(undef, n_observations, nthreads())
    
    # Gaussian random fields for generating intial state and state noise
    x, y = get_axes(model_params)
    initial_state_grf = init_gaussian_random_field_generator(
        model_params.lambda_initial_state,
        model_params.nu_initial_state,
        model_params.sigma_initial_state,
        x,
        y
    )
    state_noise_grf = init_gaussian_random_field_generator(
        model_params.lambda,
        model_params.nu,
        model_params.sigma,
        x,
        y
    )

    # Set up tsunami model
    model_matrices = SPEEDY.setup(
        model_params.nlon,
        model_params.nlat,
        model_params.nlev
    )
    create_folders(
        model_params.output_folder, 
        model_params.anal_folder,
        model_params.guess_folder
    )
    dates = collect(DateTime(model_params.IDate, SPEEDY_DATE_FORMAT):Dates.Hour(6):DateTime(model_params.endDate, SPEEDY_DATE_FORMAT))

    return ModelData(
        model_params, 
        station_grid_indices, 
        field_buffer,
        observation_buffer,
        initial_state_grf,
        state_noise_grf, 
        model_matrices,
        dates
    )
end

function ParticleDA.get_observation_mean_given_state!(
    observation_mean::AbstractVector, state::AbstractVector, model_data::ModelData
)
    return get_obs!(
        observation_mean, 
        reshape(
            state, 
            (
                model_data.model_params.nlon, 
                model_data.model_params.nlat, 
                model_data.model_params.n_state_var
            )
        ), 
        model_data.model_params, 
        model_data.station_grid_indices
    )
end

function ParticleDA.sample_observation_given_state!(
    observation::AbstractVector{S},
    state::AbstractVector{T},
    model_data::ModelData, 
    rng::AbstractRNG
) where{S, T}
    ParticleDA.get_observation_mean_given_state!(observation, state, model_data)
    add_noise!(
        observation, 
        rng, 
        ParticleDA.get_covariance_observation_noise(model_data),
    )
    return observation
end

function ParticleDA.get_log_density_observation_given_state(
    observation::AbstractVector, 
    state::AbstractVector, 
    model_data::ModelData,
)
    observation_mean = view(model_data.observation_buffer, :, threadid())
    ParticleDA.get_observation_mean_given_state!(observation_mean, state, model_data)
    return -invquad(
        ParticleDA.get_covariance_observation_noise(model_data), 
        observation - observation_mean
    ) / 2 
end

function ParticleDA.update_state_deterministic!(
    state::AbstractVector, d::ModelData, time_index::Int
)
    state_fields = reshape(
        state, 
        (
            d.model_params.nlon, 
            d.model_params.nlat, 
            d.model_params.n_state_var, 
        )
    )
    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    prt = rand(1:100000)
    # Check if the subfolders have been generated
    prt_anal_folder = joinpath(d.model_params.anal_folder, string(my_rank), string(prt))
    prt_guess_folder = joinpath(d.model_params.guess_folder, string(my_rank), string(prt))
    prt_rank_tmp = joinpath(d.model_params.output_folder, "DATA", "tmp", string(my_rank), string(prt))
 
    while isdir(prt_anal_folder) == true
        prt = rand(1:100000)
        prt_anal_folder = joinpath(d.model_params.anal_folder, string(my_rank), string(prt))
        prt_guess_folder = joinpath(d.model_params.guess_folder, string(my_rank), string(prt))
        prt_rank_tmp = joinpath(d.model_params.output_folder, "DATA", "tmp", string(my_rank), string(prt))
    end
    mkdir(prt_anal_folder)
    mkdir(prt_guess_folder)
    mkdir(prt_rank_tmp)

    idate = Dates.format(d.dates[time_index],SPEEDY_DATE_FORMAT)
    dtdate = Dates.format(d.dates[time_index+1],SPEEDY_DATE_FORMAT)
    #Write to file
    anal_file = joinpath(prt_anal_folder, idate * ".grd")
    write_fortran(anal_file,d.model_params.nlon, d.model_params.nlat, d.model_params.nlev, state_fields[:, :, :])
    # Update the dynamics
    speedy_update!(d.model_params.SPEEDY,d.model_params.output_folder,idate, dtdate, string(my_rank), string(prt))
    # Read back in the data and update the states
    guess_file = joinpath(prt_guess_folder, dtdate * ".grd")
    read_grd!(@view(state_fields[:, :, :]), guess_file, d.model_params.nlon, d.model_params.nlat, d.model_params.nlev)
    # Remove old files
    rm(prt_anal_folder; recursive=true)
    rm(prt_guess_folder; recursive=true)
    rm(prt_rank_tmp; recursive=true)
    
    # d.dates[1,threadid()], d.dates[2,threadid()] = step_datetime(d.dates[1,threadid()],d.dates[2,threadid()])
end

function ParticleDA.update_state_stochastic!(
    state::AbstractVector, model_data::ModelData, rng::AbstractRNG
)
    state_fields = reshape(
        state, 
        (
            model_data.model_params.nlon, 
            model_data.model_params.nlat, 
            model_data.model_params.n_state_var, 
        )
    )
    # Add state noise
    add_random_field!(
        state_fields,
        view(model_data.field_buffer, :, :, 1, threadid()),
        model_data.state_noise_grf,
        rng,
    )
end


function create_folders(output_folder::String, anal_folder::String, gues_folder::String)
    MPI.Init()
    comm = MPI.COMM_WORLD
    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    data = joinpath(output_folder, "DATA")
    ens = joinpath(data, "ensemble")
    tmp = joinpath(data, "tmp")
    if my_rank == 0
        rm(data; recursive=true)
        mkdir(data)
        mkdir(ens)
        mkdir(tmp)
        mkdir(anal_folder)
        mkdir(gues_folder)
    end
    MPI.Barrier(comm)
    rank_tmp = joinpath(tmp,string(my_rank))
    rank_anal = joinpath(anal_folder,string(my_rank))
    rank_gues = joinpath(gues_folder,string(my_rank))
    mkdir(rank_tmp)
    mkdir(rank_anal)
    mkdir(rank_gues)
end

function read_grd!(truth::AbstractArray{T}, filename::String, nlon::Int, nlat::Int, nlev::Int) where T

    nij0 = nlon*nlat
    iolen = 4
    nv3d = 4
    nv2d = 2
    v3d = Array{Float32, 4}(undef, nlon, nlat, nlev, nv3d)
    v2d = Array{Float32, 3}(undef, nlon, nlat, nv2d)

    f = FortranFile(filename, access="direct", recl=nij0*iolen)

    irec = 1

    for n = 1:nv3d
        for k = 1:nlev
            v3d[:,:,k,n] .= read(f, (Float32, nlon, nlat), rec=irec)
            irec += 1
        end
    end

    for n = 1:nv2d
        v2d[:,:,n] .= read(f, (Float32, nlon, nlat), rec = irec)
        irec += 1
    end
    truth[:,:,1:(nv3d*nlev)] .= reshape(v3d, size(truth[:,:,1:(nv3d*nlev)]))
    truth[:,:,(nv3d*nlev)+1:(nv3d*nlev)+2] .= v2d

    close(f)

end

function write_fortran(filename::String,nlon::Int, nlat::Int, nlev::Int,dataset::AbstractArray{T}) where T

    nij0 = nlon*nlat
    iolen = 4
    nv3d = 4
    nv2d = 2
    iolen = 4
    v3d = Array{Float32, 4}(undef, nlon, nlat, nlev, nv3d)
    v2d = Array{Float32, 3}(undef, nlon, nlat, nv2d)
    v3d .= reshape(dataset[:,:,1:(nv3d*nlev)], size(v3d))
    v2d .= dataset[:,:,(nv3d*nlev)+1:(nv3d*nlev)+2]

    f = FortranFile(filename, "w", access="direct", recl=(nij0*iolen))
    irec = 1
    for n = 1:nv3d
        for k = 1:nlev
            write(f,v3d[:,:,k,n],rec=irec)
            irec=irec+1
        end
    end
    for n = 1:nv2d
        write(f,v2d[:,:,n],rec=irec)
        irec=irec+1
    end
    close(f)
end


### Model IO

function write_params(output_filename, params)

    file = h5open(output_filename, "cw")

    if !haskey(file, params.title_params)

        group = create_group(file, params.title_params)

        fields = fieldnames(typeof(params));

        for field in fields

            attributes(group)[string(field)] = getfield(params, field)

        end

    else

        @warn "Write failed, group $(params.title_params) already exists in $(file.filename)!"

    end

    close(file)

end

function write_grid(output_filename, params)

    h5open(output_filename, "cw") do file

        if !haskey(file, params.title_grid)

            # Write grid axes
            group = create_group(file, params.title_grid)
            x,y = get_axes(params)
            #TODO: use d_write instead of create_dataset when they fix it in the HDF5 package
            ds_x,dtype_x = create_dataset(group, "x", collect(x))
            ds_y,dtype_x = create_dataset(group, "y", collect(x))
            ds_x[1:params.nlon] = collect(x)
            ds_y[1:params.nlat] = collect(y)
            attributes(ds_x)["Unit"] = "°"
            attributes(ds_y)["Unit"] = "°"

        else

            @warn "Write failed, group $(params.title_grid) already exists in $(file.filename) !"

        end

    end

end

function write_stations(
    output_filename, station_grid_indices::AbstractMatrix, params::ModelParameters
) where T

    h5open(output_filename, "cw") do file

        if !haskey(file, params.title_stations)
            group = create_group(file, params.title_stations)
            x = (station_grid_indices[:, 1] .- 1) .* params.dx
            y = (station_grid_indices[:, 2] .- 1) .* params.dy
            for (dataset_name, val) in zip(("x", "y"), (x, y))
                ds, dtype = create_dataset(group, dataset_name, val)
                ds[:] = val
                attributes(ds)["Description"] = "Station $dataset_name coordinate"
                attributes(ds)["Unit"] = "°"
            end
        else
            @warn "Write failed, group $(params.title_stations) already exists in  $(file.filename)!"
        end
    end
end

function write_weights(file::HDF5.File, weights::AbstractVector, it::Int, params::ModelParameters)

    group_name = "weights"
    dataset_name = "t" * lpad(string(it),4,'0')

    group, subgroup = ParticleDA.create_or_open_group(file, group_name)

    if !haskey(group, dataset_name)
        #TODO: use d_write instead of create_dataset when they fix it in the HDF5 package
        ds,dtype = create_dataset(group, dataset_name, weights)
        ds[:] = weights
        attributes(ds)["Description"] = "Particle Weights"
        attributes(ds)["Unit"] = ""
        attributes(ds)["Time step"] = it
        # attributes(ds)["Time (s)"] = it * params.time_step
    else
        @warn "Write failed, dataset $group_name/$dataset_name  already exists in $(file.filename) !"
    end

end

function ParticleDA.write_snapshot(output_filename::AbstractString,
                                   model_data::ModelData,
                                   states::AbstractMatrix{T},
                                   avg::AbstractArray{T},
                                   var::AbstractArray{T},
                                   weights::AbstractVector{T},
                                   it::Int) where T

    if it == 0
        # These are written only at the initial state it == 0
        write_grid(output_filename, model_data.model_params)
        write_params(output_filename, model_data.model_params)
        write_stations(
            output_filename, model_data.station_grid_indices, model_data.model_params
        )
    end

    if any(model_data.model_params.particle_dump_time .== it)
        write_particles(
            model_data.model_params.particle_dump_file, 
            states, 
            it, 
            model_data.model_params
        )
    end

    return ParticleDA.write_snapshot(
        output_filename, avg, var, weights, it, model_data.model_params
    )
end

function ParticleDA.write_snapshot(output_filename::AbstractString,
                                   avg::AbstractArray{T},
                                   var::AbstractArray{T},
                                   weights::AbstractVector{T},
                                   it::Int,
                                   params::ModelParameters) where T

    println("Writing output at timestep = ", it)
    
    avg = reshape(avg, (params.nlon, params.nlat, params.n_state_var, :))
    var = reshape(var, (params.nlon, params.nlat, params.n_state_var, :))

    h5open(output_filename, "cw") do file

        for (i, metadata) in enumerate(STATE_FIELDS_METADATA)

            write_field(file, @view(avg[:,:,i]), it, params.title_avg, metadata, params)
            write_field(file, @view(var[:,:,i]), it, params.title_var, metadata, params)

        end

        write_weights(file, weights, it, params)
    end

end

function write_obs(file::HDF5.File, observation::AbstractVector, it::Int, params::ModelParameters)

    group_name = "obs"
    dataset_name = "t" * lpad(string(it),4,'0')

    group, subgroup = ParticleDA.create_or_open_group(file, group_name)

    if !haskey(group, dataset_name)
        #TODO: use d_write instead of create_dataset when they fix it in the HDF5 package
        ds,dtype = create_dataset(group, dataset_name, observation)
        ds[:] = observation
        attributes(ds)["Description"] = "Observations"
        attributes(ds)["Unit"] = ""
        attributes(ds)["Time step"] = it
        # attributes(ds)["Time (s)"] = it * params.time_step
    else
        @warn "Write failed, dataset $group_name/$dataset_name  already exists in $(file.filename) !"
    end

end

function ParticleDA.write_state_and_observations(state::AbstractArray{T}, 
                                      observation::AbstractArray{T},
                                      it::Int, 
                                      model_data::ModelData) where T

    println("Writing output at timestep = ", it)
    params = model_data.model_params
    state = reshape(state, (params.nlon, params.nlat, params.n_state_var, :))
    h5open(params.truth_obs_filename, "cw") do file
        for (i, metadata) in enumerate(STATE_FIELDS_METADATA)
            write_field(file, @view(state[:,:,i]), it, params.title_syn, metadata, params)
        end
        if it > 0
            # These are written only after the initial state
            write_obs(file, observation, it, params)
        end
    end
end

function write_particles(
    output_filename::AbstractString,
    states::AbstractMatrix{T},
    it::Int,
    params::ModelParameters
) where T

    println("Writing particle states at timestep = ", it)
    
    nprt = size(states, 2)
    state_fields = reshape(states, (params.nlon, params.nlat, params.n_state_var, nprt))

    h5open(output_filename, "cw") do file
        for p = 1:nprt
            group_name = "particle" * string(p)
            for (i, metadata) in enumerate(STATE_FIELDS_METADATA)
                write_field(
                    file, @view(state_fields[:, :, i, p]), it, group_name, metadata, params
                )
            end
        end
    end

end

function write_field(
    file::HDF5.File,
    field::AbstractMatrix{T},
    it::Int,
    group::String,
    metadata::StateFieldMetadata,
    params::ModelParameters
) where T

    group_name = params.state_prefix * "_" * group
    subgroup_name = "t" * lpad(string(it), 4, '0')
    dataset_name = metadata.name

    group, subgroup = ParticleDA.create_or_open_group(file, group_name, subgroup_name)

    if !haskey(subgroup, dataset_name)
        #TODO: use d_write instead of create_dataset when they fix it in the HDF5 package
        ds, _ = create_dataset(subgroup, dataset_name, field)
        ds[:,:] = field
        attributes(ds)["Description"] = metadata.description
        attributes(ds)["Unit"] = metadata.unit
        attributes(ds)["Time step"] = it
    else
        @warn "Write failed, dataset $group_name/$subgroup_name/$dataset_name already exists in $(file.filename) !"
    end
end

end
