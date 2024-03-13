# ANAET state space model


Example integration of ParticleDA with a toy model of magnetohydrodynamic instability
coupled to a slow dissipative background evolution, produced as part of the 
[ExCALIBUR fusion use case project NEPTUNE](https://excalibur.ac.uk/projects/excalibur-fusion-use-case-project-neptune-neutrals-plasma-turbulence-numerics-for-the-exascale/).

The deterministic dynamics here correspond to the _axissymmetric non-axissymmetric
extended_ (ANAET) model described by [Arter
(2012)](https://doi.org/10.13140/RG.2.2.35052.77449), with the original ordinary
differential equation (ODE) system here changed to a stochastic differential equation
(SDE) system

$$
  \mathrm{d}\boldsymbol{x}(t)  =
  \begin{pmatrix} 
  x_2(t) \\ 
  -\gamma_r x_1(t) - (\mu_1 + \mu_2 x_3(t)) x_1(t)^3 - \mu_6 x_1(t)^6 x_2(t) \\
  \nu_1 - \nu_2 x_3(t)^2 - (\delta_0 + \delta_1 b) x_1(t)^2
  \end{pmatrix}\mathrm{d}t + \beta \mathrm{d} \boldsymbol{w}(t).
$$

The state space model formulation uses a Gaussian approximation to the state transition
distributions for the SDE system based on a splitting numerical scheme which uses an
adaptive ODE solver to solve for the deterministic component of the dynamics and
Euler-Maruyama discretisation for the driving Wiener noise processes.

## Prerequisites

The example requires the following to be installed:

- Julia (tested with version 1.9) - [installation instructions](https://julialang.org/downloads/).

## Installation

To install the `ANAET` package and its dependencies from a Julia REPL run

```Julia
using Pkg
Pkg.add(
    url="https://github.com/Team-RADDISH/ParticleDA-UseCases",
    subdir="ANAET"
)
```

## Example usage

### Simulating from model

To generate a simulated observation sequence from the model we can use the
[`simulated_observations_from_model`
function](https://team-raddish.github.io/ParticleDA.jl/stable/#ParticleDA.simulate_observations_from_model)
exported by `ParticleDA`. We need to initialise an instance of the model using the
`init` method specified in the `ANAET` module, passing in a dictionary of any model
parameters we wish to adjust from their default values. The code example below simulates
the model for 100 (equispaced) observation times, writing the resulting simulated
observation sequence to a HDF5 file `observations.h5` in the current working directory.

```Julia
using ParticleDA
using HDF5
using ANAET

num_observation_times = 100

model_parameters_dict = Dict()

model = ANAET.init(model_parameters_dict)

observation_sequence = h5open("observations.h5", "w") do output_file
    simulate_observations_from_model(
        model, num_observation_times; output_file
    )
end
```

### Filtering

To perform filtering, that is estimation of the distribution of the model state at each
observed time given the observations up to that time, we can use the
[`run_particle_filter` function](https://team-raddish.github.io/ParticleDA.jl/stable/#ParticleDA.run_particle_filter)
exported by ParticleDA. ParticleDA supports parallelising filtering run using [Julia's
multithreading support](https://docs.julialang.org/en/v1/manual/multi-threading/), with
each thread sampling new values for one or more state particles in the ensemble from the
proposal distributions in parallel. To laThe code example below would perform filtering
for the ANAET state space model with 500 particles for a sequence of observations
recorded in a HDF5 file `observations.h5`, and write the filtering results to a HDF5
file `filter_results.h5`. We specify to use the locally optimal proposal filter type and
to compute both mean and variance summary statistics from the ensemble at each
observation time.

```Julia
using ParticleDA
using HDF5
using ANAET

observation_sequence = h5open("observations.h5", "r") do observation_file
    ParticleDA.read_observation_sequence(observation_file)
end

filter_parameters = ParticleDA.FilterParameters(
    nprt=500,
    output_filename="filter_results.h5",
    verbose=true,
)

model_parameters_dict = Dict()

final_state, final_statistics = run_particle_filter(
    ANAET.init,
    filter_parameters,
    model_parameters_dict,
    observation_sequence,
    OptimalFilter,
    MeanAndVarSummaryStat
)
```

## Acknowledgements

This work was funded by a grant from the ExCALIBUR programme.
