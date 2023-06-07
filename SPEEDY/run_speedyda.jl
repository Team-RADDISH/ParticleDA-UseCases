# Load ParticleDA
using ParticleDA
#
# # Save some variables for later use
test_dir = joinpath(@__DIR__)
module_src = joinpath(test_dir, "model", "model.jl")
input_file = joinpath(test_dir, "inputs", "ensemble_assimilation.yaml")
obs_file = joinpath(test_dir, "results", "real_obs_Jan_Aug.h5")
# # Instantiate the test environment
using Pkg
Pkg.activate(test_dir)
Pkg.instantiate()

# Include the sample model source code and load it
include(module_src)
using .Model

# Simulate observations from the model to use 
simulate_observations_from_model(Model.init, input_file, obs_file)
# Run the (optimal proposal) particle filter with simulated observations computing the
# mean of the particle ensemble. 
run_particle_filter(Model.init, input_file, obs_file, ParticleDA.OptimalFilter, ParticleDA.NaiveMeanSummaryStat)
