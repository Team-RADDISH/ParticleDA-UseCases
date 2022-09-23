# Load ParticleDA
using ParticleDA
#
# # Save some variables for later use
test_dir = joinpath(@__DIR__)
module_src = joinpath(test_dir, "model", "model.jl")
input_file = joinpath(test_dir,"speedy_ps.yaml")
obs_file = joinpath(test_dir, "results", "flatten", "observations.h5")
# # Instantiate the test environment
using Pkg
Pkg.activate(test_dir)
Pkg.instantiate()

# Include the sample model source code and load it
include(module_src)
using .Model
run_particle_filter(Model.init, input_file, obs_file, ParticleDA.OptimalFilter, ParticleDA.NaiveMeanSummaryStat)
