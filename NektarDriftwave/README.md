# NetkarDriftwave state space model

Example integration of ParticleDA with a [Nektar++](https://www.nektar.info/) solver for a plasma physics model, produced as part of the 
[ExCALIBUR fusion use case project NEPTUNE](https://excalibur.ac.uk/projects/excalibur-fusion-use-case-project-neptune-neutrals-plasma-turbulence-numerics-for-the-exascale/).

This example combines the [`nektar-driftwave` solver](https://github.com/ExCALIBUR-NEPTUNE/nektar-driftwave) with simulation of Gaussian random fields with Matern covariance functions using Nektar++ to form a state space model suitable for applying a particle filter to. The deterministic dynamics solve the [two-dimensional Hasegawa-Wakatani equations](https://github.com/ExCALIBUR-NEPTUNE/nektar-driftwave/blob/master/doc.ipynb) with spatially correlated additive state noise simulated by solving a Helmholtz equation driven by a Gaussian white noise process. 

The system is simulated on a rectangular spatial domain with periodic boundary conditions, and a regular quadrilateral mesh, using the Nektar++ spectral element method implementation. The Nektar++ mesh file is generated using [the Gmsh Julia interface](https://github.com/JuliaFEM/Gmsh.jl) and the [Nektar++ NekMesh utility](https://doc.nektar.info/userguide/5.4.0/user-guidech5.html), allowing the mesh dimensions and spatial size to be specified programmatically.

## Prerequisites

The example requires the following to be installed:

- Nektar++ with HDF5 support (tested with [version 5.4](https://gitlab.nektar.info/nektar/nektar/-/tree/v5.4.0)) - [installation instructions](https://www.nektar.info/getting-started/installation/).
- `nektar-driftwave` solver (tested with [commit `5a6e941`](https://github.com/ExCALIBUR-NEPTUNE/nektar-driftwave/tree/5a6e94131926116590a988058b651e653ac423b7)) - [installation instructions](https://github.com/ExCALIBUR-NEPTUNE/nektar-driftwave?tab=readme-ov-file#compilation--installation).
- Julia (tested with version 1.9) - [installation instructions](https://julialang.org/downloads/).

## Installation

To install the `NektarDriftwave` package and its dependencies from a Julia REPL run

```Julia
using Pkg
Pkg.add(
    url="https://github.com/Team-RADDISH/ParticleDA-UseCases",
    subdir="NektarDriftwave"
)
```

## Known issues

### Non-reproducibility of filtering results

Nektar++ does not appear to allow fixing the seed for the (Boost) pseudo-random number generators used to simulate Gaussian white noise fields using the `awgn` function in expressions in the XML conditions files, therefore currently even with a fixed seed set in the filter parameters, the Gaussian random fields used to perturb the state at each observation time will not be consistent across different particle filter runs and so the filter results will not be entirely reproducible.

### Model incompatible with `OptimalFilter` filter type

Currently the `NektarDriftwave` module only implements methods for [the subset of ParticleDA functions required to run a bootstrap particle filter algorithm](https://team-raddish.github.io/ParticleDA.jl/stable/#Interfacing-the-model) (specified with a filter type `BootstrapFilter`). The model here does meet the conditional linear-Gaussianity assumptions underlying the more statistically efficient locally optimal proposal filter (specified with a filter type `OptimalFilter`), however the approach used for simulating Gaussian random fields using Nektar++, does not allow for easy explicit evaluation of the covariance between pairs of state dimensions (degrees of freedom), which is required to implement methods for the `get_covariance_state_noise`, `get_covariance_state_observation_given_previous_state` and `get_covariance_observation_observation_given_previous_state` functions required by the locally optimal proposal filter implementation.

### Building and running example with MPI on ARCHER2

When using the Cray MPICH (tested with version 8.1.23) MPI implementation on ARCHER2, if Nektar++ (and the `nektar-driftwave` solver) are built with MPI support, running a `ParticleDA` particle filter using MPI hangs when calling the Nektar++ solvers, we believe due to the nested use of MPI at the particle filter and Nektar++ levels. This issue is potentially specific to the Cray MPICH library, as the issue does not occur when building and running using OpenMPI on another system. The implementation here executes the Nektar++ solvers and utilities directly using system calls but similar behaviour was observed when using `MPI_Comm_spawn` to spawn new processes to run the solver instances in within the ParticleDA wrapper. 

Building Nektar++ with HDF5 support but without MPI, requires manually commenting out / removing [the lines in the `cmake/ThirdPartyHDF5.cmake` file in the Nektar++ source tree which raise an error if trying to build with HDF5 but no MPI support](https://gitlab.nektar.info/nektar/nektar/-/blob/8bc4b3095361e868b26219eff826d4f1902763df/cmake/ThirdPartyHDF5.cmake#L12-16). After changing these lines in a local clone of the Nektar++ repository, we successively built Nektar++ with HDF5 but no MPI support on ARCHER2 (using the pre-built Cray HDF5 library on ARCHER2) by running the following from the root of the repository

```sh
module load cpe/22.12
module swap PrgEnv-cray PrgEnv-gnu 
module load cray-fftw
module load cray-hdf5-parallel
module load cmake 
mkdir build && cd build
CC=cc CXX=CC cmake -DNEKTAR_USE_MPI=OFF -DNEKTAR_USE_HDF5=ON -DNEKTAR_USE_FFTW=ON -DTHIRDPARTY_BUILD_BOOST=ON -DNEKTAR_USE_SYSTEM_HDF5=ON ..
make install
```

and then building and installing the `nektar-driftwave` solver following [the instructions in it's repository](https://github.com/ExCALIBUR-NEPTUNE/nektar-driftwave?tab=readme-ov-file#compilation--installation) with `-DNektar++_DIR` option set to point to the `dist/lib64/nektar++/cmake` subdirectory in the directory Nektar++ was built.

## Model parameters

The `NektarDriftwaveModelParameters` struct exposes various parameters which can be used to specify the model behaviour and setup: 

- `nektar_bin_directory` (default = `""`): Path to directory containing built Nektar++ binaries (specifically the example here uses `NekMesh`, `FieldConvert` and `ADRSolver`). Needs to be explicitly set unless these binaries are already on the search path.
- `driftwave_solver_bin_directory` (default = `""`): Path to directory containing built `DriftWaveSolver` binary. Needs to be explicitly set unless the binary is already on the search path.
- `alpha` (default = `2.`): Hasegawa-Wakatani system parameter α (adiabiacity operator).
- `kappa` (default = `1.`): Hasegawa-Wakatani system parameter κ (background density gradient scale-length).
- `mesh_dims` (default = `[32, 32]`): Number of quadrilateral elements along each axis in mesh.
- `mesh_size` (default = `[40., 40.]`): Size (extents) of rectangular spatial domain mesh is defined on.
- `num_modes` (default = `4`): Number of modes in spectral element expansion (one higher than the polynomial order).
- `time_step` (default = `0.0005`): Time step for numerical integration in time.
- `num_steps_per_observation_time` (default = `1000`): Number of time integrations steps to perform between each observation of state.
- `observed_points` (default = `[[-10., -10.], [0., -10.], [10., -10.], [-10., 0.], [0., 0.], [10., 0.], [-10., 10.], [0., 10.], [10., 10.]]`): Points at which state is observed in two-dimensional spatial domain.
- `observed_variables` (default = `["zeta"]`): Which of field variables are observed (a subset of `{"phi", "zeta", "n"}`).
- `observation_noise_std` (default = `0.1`): Scale parameter (standard deviation) of independent Gaussian noise in observations.
- `state_grf_length_scale` (default = `1.`): Length scale parameter for Gaussian random fields used for state noise and initialization.
- `state_grf_smoothness` (default = `2`): Positive integer smoothness parameter for Gaussian random fields used for state noise and initialization.
- `initial_state_grf_output_scale` (default = `0.05`): Output scale parameter for initial state Gaussian random field.
- `state_noise_grf_output_scale` (default = `0.05`): Output scale parameter for additive state noise Gaussian random field.
- `initial_state_mean_length_scale` (default = `2`): Length scale parameter for bump functions used for initial state field means.


## Example usage

### Simulating from model

To generate a simulated observation sequence from the model we can use the [`simulated_observations_from_model` function](https://team-raddish.github.io/ParticleDA.jl/stable/#ParticleDA.simulate_observations_from_model) exported by `ParticleDA`. We need to initialise an instance of the model using the `init` method specified in the `NektarDriftwave` module, passing in a dictionary of any model parameters we wish to adjust from their default values - in most cases this will include setting the parameters `nektar_bin_directory` and `driftwave_solver_bin_directory` to point to the system specific directories in which the built Nektar++ and `nektar-driftwave` binaries can be found. The code example below simulates the model for 100 (equispaced) observation times, writing the resulting simulated observation sequence to a HDF5 file `observations.h5` in the current working directory.

```Julia
using ParticleDA
using HDF5
using NektarDriftwave

num_observation_times = 100

model_parameters_dict = Dict(
    "nektar_bin_directory" => "/path/to/nektar/build/dist/bin",
    "driftwave_solver_bin_directory" => "/path/to/nektar-driftwave/build/dist",
)

model = NektarDriftwave.init(model_parameters_dict)

observation_sequence = h5open("observations.h5", "w") do output_file
    simulate_observations_from_model(
        model, num_observation_times; output_file
    )
end
```

### Distributed filtering using MPI

To perform filtering, that is estimation of the distribution of the model state at each observed time given the observations up to that time, we can use the [`run_particle_filter` function](https://team-raddish.github.io/ParticleDA.jl/stable/#ParticleDA.run_particle_filter) exported by ParticleDA. ParticleDA supports distributing the filtering run using a _message passing interface_ (MPI) implementation, with each process sampling new values for one or more state particles in the ensemble from the proposal distributions in parallel, with MPI communication operations used to perform the resampling step which involves synchronization across the processes. The code example below would perform filtering for the NektarDriftwave state space model for a sequence of observations recorded in a HDF5 file `observations.h5`, and write the filtering results to a HDF5 file `filter_results.h5`. It is assumed the code is launched using [the `mpiexecjl` Julia wrapper](https://juliaparallel.org/MPI.jl/latest/usage/#Julia-wrapper-for-mpiexec)  to run over multiple MPI ranks (processes) with each rank hosting a fixed number of particles (by default here one particle per rank). We specify to use the bootstrap filter type and to compute both mean and variance summary statistics from the ensemble at each observation time.

```Julia
using ParticleDA
using HDF5
using MPI
using NektarDriftwave

MPI.Init()
num_ranks = MPI.Comm_size(MPI.COMM_WORLD)

observation_sequence = h5open("observations.h5", "r") do observation_file
    ParticleDA.read_observation_sequence(observation_file)
end

num_particles_per_rank = 1

filter_parameters = ParticleDA.FilterParameters(
    nprt=num_ranks * num_particles_per_rank,
    output_filename="filter_results.h5",
    verbose=true,
)

model_parameters_dict = Dict(
    "nektar_bin_directory" => "/path/to/nektar/build/dist/bin",
    "driftwave_solver_bin_directory" => "/path/to/nektar-driftwave/build/dist",
)

final_state, final_statistics = run_particle_filter(
    NektarDriftwave.init,
    filter_parameters,
    model_parameters_dict,
    observation_sequence,
    BootstrapFilter,
    MeanAndVarSummaryStat
)
```

### Visualizing state trajectories using Paraview

The `NetkarDriftwave` module includes a helper function `generate_paraview_vtu_stack` to extract a sequence of state fields (either simulating directly from the model or for example the ensemble mean estimate of the state from a filtering run) from a HD5 file output by ParticleDA and output to a stack of VTU (Visualization Toolkit unstructured grid file) files which can then be loaded in to [ParaView](https://www.paraview.org/) for visualization purposes. The code example below, loads the sequence of states recorded under the `state` key in a HDF5 file `observations.h5` (as would be output by the [_Simulating from the model_ example](#simulating-from-model) above) and writes a stack of VTU files to a subdirectory `vtus` in the current working directory (this directory is assumed to already exist). As the utility function needs to be able to generate a mesh file to perform the conversion using `NekMesh`, a model instance needs to be initialized and passed to the `generate_paraview_vtu_stack` function, with the model being used to access the `NekMesh` utility binary and generate the mesh file (it is assumed the parameters specifying the mesh dimensions and size passed to this model match those used in simulating the state sequence being extracted). The stack of `.vtu` files can then be directly [loaded as a file series in ParaView](https://docs.paraview.org/en/latest/UsersGuide/dataIngestion.html), with the time step included in the file names ensuring they should be displayed in the correct order in animations.


```Julia
using HDF5
using NektarDriftwave

model_parameters_dict = Dict(
    "nektar_bin_directory" => "/path/to/nektar/build/dist/bin",
    "driftwave_solver_bin_directory" => "/path/to/nektar-driftwave/build/dist",
)

model = NektarDriftwave.init(model_parameters_dict)

h5open("observations.h5", "r") do observation_file
    NektarDriftwave.generate_paraview_vtu_stack(
        observation_file, "state", model, "vtus"
    )
end
```

## Acknowledgements

This work was funded by a grant from the ExCALIBUR programme.
