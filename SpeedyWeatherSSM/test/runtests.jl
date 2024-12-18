using ParticleDA
using Random
using Test
using ReadOnlyArrays
using SpeedyWeather
using SpeedyWeatherSSM


@testset (
    "equispaced_lat_lon_grid with T=$T n_lat=$n_lat, n_lon=$n_lon"
 ) for T in (Float32, Float64), n_lat in (1, 3), n_lon in (1, 2)
     lat_lon_pairs = SpeedyWeatherSSM.equispaced_lat_lon_grid(T, n_lat, n_lon)
     @test lat_lon_pairs isa Matrix{T}
     @test size(lat_lon_pairs) == (2, n_lat * n_lon)
     @test all(-90 .<= lat_lon_pairs[1, :] .<= 90)
     @test all(-180 .<= lat_lon_pairs[2, :] .<= 180)
end

function check_consistency_vector_and_spectral_coefficients(
    vector::Vector{T}, spectral_coefficients::LowerTriangularMatrix{Complex{T}}
) where {T <: AbstractFloat}
    n_row, n_col = size(spectral_coefficients, as=Matrix)
    # Test first (n_row - 1) entries corresponding to real component of first column
    @test all(vector[1:n_row - 1] == real(spectral_coefficients[1:n_row - 1]))
    # Test remaining entries correspond to interleaved real and imaginary components
    # scanning across columns
    vector_index = n_row
    for col_index in 2:n_col
        for row_index in col_index:(n_row - 1)
            @test vector[vector_index] == real(
                spectral_coefficients[row_index, col_index]
            )
            @test vector[vector_index + 1] == imag(
                spectral_coefficients[row_index, col_index]
            )
            vector_index += 2
        end
    end
end

@testset (
    "update_spectral_coefficients_from_vector with T = $T, trunc = $spectral_truncation"
) for T in (Float32, Float64), spectral_truncation in (1, 7, 15)
    rng = Random.Xoshiro(1234)
    vector = randn(T, (spectral_truncation + 1)^2)
    n_row, n_col = spectral_truncation + 2, spectral_truncation + 1
    spectral_coefficients = SpeedyWeather.LowerTriangularMatrix{Complex{T}}(
        undef, n_row, n_col
    )
    # Set coefficients in last row to NaN - these should be replaced with zero
    for col_index in 1:n_col
        spectral_coefficients[n_row, col_index] = NaN
    end
    # Wrap vector as read-only to check not being mutated 
    SpeedyWeatherSSM.update_spectral_coefficients_from_vector!(
        spectral_coefficients, ReadOnlyVector(vector), spectral_truncation
    )
    # Check last row entries all zero
    for col_index in 1:n_col
        @test spectral_coefficients[n_row, col_index] == 0
    end
    check_consistency_vector_and_spectral_coefficients(vector, spectral_coefficients)     
end

@testset (
    "update_vector_from_spectral_coefficients with T = $T, trunc = $spectral_truncation"
) for T in (Float32, Float64), spectral_truncation in (1, 7, 15)
    rng = Random.Xoshiro(1234)
    vector = Vector{T}(undef, (spectral_truncation + 1)^2)
    n_row, n_col = spectral_truncation + 2, spectral_truncation + 1
    spectral_coefficients = SpeedyWeather.LowerTriangularMatrix{Complex{T}}(
        undef, n_row, n_col
    )
    # First column (order = m = 0) coefficients real-valued
    spectral_coefficients[1:n_row] = randn(rng, T, n_row)
    # Remainder of coefficients complex valued
    spectral_coefficients[n_row + 1:end] = randn(
        rng, Complex{T}, length(spectral_coefficients) - n_row
    )
    # Set coefficients in last row to NaN - these should be ignored
    for col_index in 1:n_col
        spectral_coefficients[n_row, col_index] = NaN
    end
    # Wrap spectral_coefficient as read-only to check not being mutated 
    SpeedyWeatherSSM.update_vector_from_spectral_coefficients!(
        vector, ReadOnlyVector(spectral_coefficients), spectral_truncation
    )
    # Check no NaNs from last row ended up in vector
    @test !any(isnan.(vector))
    check_consistency_vector_and_spectral_coefficients(vector, spectral_coefficients)          
end

@testset (
    "Updating prognostic variables to/from state vector with \
    T=$T, spectral_truncation=$spectral_truncation, n_layers=$n_layers"
 ) for T in (Float32, Float64), spectral_truncation in (7, 15), n_layers in (1, 8)
    rng = Random.Xoshiro(1234)
    spectral_grid = SpectralGrid(;
        NF=T, trunc=spectral_truncation, nlayers=n_layers
    )
    layered_variables = (:vor, :div, :temp, :humid)
    surface_variables = (:pres,)
    all_variables = (layered_variables..., surface_variables...)
    state_dimension = (spectral_truncation + 1)^2 * (
        n_layers * length(layered_variables) + length(surface_variables)
    )
    state = rand(rng, T, state_dimension)
    prognostic_variables_1 = SpeedyWeather.PrognosticVariables(spectral_grid)
    prognostic_variables_1_init = SpeedyWeather.PrognosticVariables(spectral_grid)
    SpeedyWeather.copy!(prognostic_variables_1_init, prognostic_variables_1)
    prognostic_variables_2 = SpeedyWeather.PrognosticVariables(spectral_grid)
    SpeedyWeatherSSM.update_prognostic_variables_from_state_vector!(
        prognostic_variables_1, state, all_variables
    )
    SpeedyWeatherSSM.update_prognostic_variables_from_state_vector!(
        prognostic_variables_2, state, all_variables
    )
    # Check all variables changed from initial values
    for variable_name in all_variables
        @test all(
            getproperty(prognostic_variables_1, variable_name)[1]
            != getproperty(prognostic_variables_1_init, variable_name)[1]
        )
    end
    # Check variables updates from same state vector match
    for variable_name in all_variables
        @test all(
            getproperty(prognostic_variables_1, variable_name)[1]
            == getproperty(prognostic_variables_2, variable_name)[1]
        )
    end
    # Check state vector reconstructed from prognostic variables matches original
    state_2 = Vector{T}(undef, state_dimension)
    SpeedyWeatherSSM.update_state_vector_from_prognostic_variables!(
        state_2, prognostic_variables_1, all_variables
    )
    @test all(state == state_2)
end
    
@testset "Generic model interface unit tests" begin
    seed = 1234
    model = SpeedyWeatherSSM.init(SpeedyWeatherSSM.SpeedyParameters())
    ParticleDA.run_unit_tests_for_generic_model_interface(model, seed)
end
