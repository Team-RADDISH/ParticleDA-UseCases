"""Nektar++ driftwave system state space model.

Deterministic dynamics solve two-dimensional Hasegawa-Wakatani equations with spatially
correlated additive state noise simulated by solving a Helmholtz equation driven by
a Gaussian white noise process. System is simulated on a rectangular spatial domain with
periodic boundary conditions, and a regular quadrilaterial mesh using Nektar++ spectral
element method implementation.
"""
module NektarDriftwave

using Base.Threads
using Distributions
using HDF5
using Random
using PDMats
using Gmsh: gmsh
using CSV
using Tables
using XML
using ParticleDA

"""
    NektarDriftwaveModelParameters()

Parameters for Nektar++ driftwave system state space model.
"""
Base.@kwdef struct NektarDriftwaveModelParameters{S <: Real, T <: Real}
    "Path to directory containing Nektar++ binaries"
    nektar_bin_directory::String = ""
    "Path to directory containing DriftWaveSolver binary"  
    driftwave_solver_bin_directory::String = ""
    "Hasegawa-Wakatani system parameter α - adiabiacity operator"
    alpha::S = 2.
    "Hasegawa-Wakatani system parameter κ - background density gradient scale-length"
    kappa::S = 1.
    "Number of quadrilateral elements along each axis in mesh"
    mesh_dims::Vector{Int} = [32, 32]
    "Size (extents) of rectangular spatial domain mesh is defined on"
    mesh_size::Vector{Float64} = [40., 40.]
    "Number of modes in expansion (one higher than the polynomial order)"
    num_modes::Int = 4
    "Time step for numerical integraton in time"
    time_step::S = 0.0005
    "Number of time integrations steps to perform between each observation of state"
    num_steps_per_observation_time::Int = 1000
    "Points at which state is observed in two-dimensional spatial domain"
    observed_points::Vector{Vector{Float64}} = map(
        collect, vec(collect(Iterators.product(-10.:10.:10., -10.:10.:10.)))
    )
    "Which of field variables are observed (subset of {phi, zeta, n})"
    observed_variables::Vector{String} = ["zeta"]
    "Scale parameter (standard deviation) of independent Gaussian noise in observations"
    observation_noise_std::T = 0.1
    "Length scale parameter for Gaussian random fields used for state noise and initialisation"
    state_grf_length_scale::S = 1.
    "Positive integer smoothness parameter for Gaussian random fields used for state noise and initialisation"
    state_grf_smoothness::Int = 2
    "Output scale parameter for initial state Gaussian random field"
    initial_state_grf_output_scale::S = 0.05
    "Output scale parameter for additive state noise Gaussian random field"
    state_noise_grf_output_scale::S = 0.05
    "Length scale parameter for bump functions used for initial state field means"
    initial_state_mean_length_scale::S = 2.
end

function get_params(
    P::Type{NektarDriftwaveModelParameters{S, T}}, model_params_dict::Dict
) where {S <: Real, T <: Real}
    return P(; (; (Symbol(k) => v for (k, v) in model_params_dict)...)...)
end

function make_gmsh_quadrilateral_mesh(output_path, mesh_dim, mesh_size)
    point_dim, line_dim, surface_dim = 0, 1, 2
    point_tag, (bottom_tag, top_tag, left_tag, right_tag, surface_tag) = 1, 1:5
    gmsh.initialize()
    # Create point at bottom-left corner
    gmsh.model.geo.addPoint(-mesh_size[1] / 2, -mesh_size[2] / 2, 0)
    # Extrude point to a segmented line rightwards along x-axis forming bottom edge
    gmsh.model.geo.extrude([(point_dim, point_tag)], mesh_size[1], 0, 0, [mesh_dim[1]])
    # Extrude segmented line to a quadrilateralized surface upwards along y-axis
    gmsh.model.geo.extrude([(line_dim, bottom_tag)], 0, mesh_size[2], 0, [mesh_dim[2]], [], true)
    gmsh.model.geo.synchronize()
    # Add physical groups for quadrilateralized surface and four boundaries
    gmsh.model.addPhysicalGroup(surface_dim, [surface_tag], 0)
    gmsh.model.addPhysicalGroup(line_dim, [bottom_tag], 1)
    gmsh.model.addPhysicalGroup(line_dim, [right_tag], 2)
    gmsh.model.addPhysicalGroup(line_dim, [top_tag], 3)
    gmsh.model.addPhysicalGroup(line_dim, [left_tag], 4)
    # Generate mesh and write to file
    gmsh.model.mesh.generate(surface_dim)
    gmsh.write(output_path)
    gmsh.finalize()
end

function change_extension(path, new_extension)
    path_stem, old_extension = splitext(path)
    return "$(path_stem).$(new_extension)"
end

struct MeshFilePaths
    with_expansions::String
    no_expansions::String
end

function make_mesh_files(parameters, output_directory, nek_mesh_path)
    mesh_file_path = joinpath(output_directory, "mesh.xml")
    mesh_no_expansions_file_path = joinpath(output_directory, "mesh_no_expansions.xml")
    gmsh_mesh_file_path = change_extension(mesh_file_path, "msh")
    make_gmsh_quadrilateral_mesh(gmsh_mesh_file_path, parameters.mesh_dims, parameters.mesh_size)
    run(`$(nek_mesh_path) -f $(gmsh_mesh_file_path) $(mesh_file_path)`)
    # Output additional copy of mesh file which removes the default EXPANSIONS element
    # to avoid interfering with solver specific element in conditions files when
    # conditions files is passed as a preceding argument (necessary with solver commands
    # for session name inferred from names of XML files passed in to be specific to the
    # conditions file rather than named after the mesh file)
    mesh_root_node = read(mesh_file_path, Node)
    filter!(node -> tag(node) != "EXPANSIONS", children(mesh_root_node[end]))
    XML.write(mesh_no_expansions_file_path, mesh_root_node)
    return MeshFilePaths(mesh_file_path, mesh_no_expansions_file_path)
end

collections_element() = XML.Element("COLLECTIONS"; DEFAULT="auto")

function expansions_element(variables, num_modes=4, composite="C[0]", type="MODIFIED")
    return XML.Element(
        "EXPANSIONS", 
        XML.Element(
            "E";
            COMPOSITE=composite,
            NUMMODES=string(num_modes),
            TYPE=type,
            FIELDS=join(variables, ",")
        )
    )
end

function solver_info_element(properties)
    return XML.Element(
        "SOLVERINFO",
        (XML.Element("I"; PROPERTY=p, VALUE=v) for (p, v) in pairs(properties))...
    )
end

function parameters_element(parameters)
    return XML.Element(
        "PARAMETERS",
        (XML.Element("P", XML.Text("$p = $v")) for (p, v) in pairs(parameters))...
    )
end

function variables_element(variables)
    return XML.Element(
        "VARIABLES",
        (XML.Element("V", XML.Text(v); ID=string(i - 1)) for (i, v) in enumerate(variables))...
    )
end

function boundary_regions_element(boundary_regions)
    return XML.Element(
        "BOUNDARYREGIONS",
        (XML.Element("B", XML.Text(r); ID=string(i - 1)) for (i, r) in enumerate(boundary_regions))...
    )
end

abstract type AbstractVariableBoundaryCondition end
    
struct PeriodicVariableBoundaryCondition <: AbstractVariableBoundaryCondition
    variable::String
    with_region_id::Int
end

struct DirichletVariableBoundaryCondition <: AbstractVariableBoundaryCondition
    variable::String
    value::String
end

struct NeumannVariableBoundaryCondition <: AbstractVariableBoundaryCondition
    variable::String
    value::String
end

struct BoundaryCondition
    composite::String
    variable_boundary_conditions::Vector{AbstractVariableBoundaryCondition}
end

variable_boundary_condition_element(vbc::PeriodicVariableBoundaryCondition) = XML.Element(
    "P"; VAR=vbc.variable, VALUE="[$(vbc.with_region_id - 1)]"
)

variable_boundary_condition_element(vbc::DirichletVariableBoundaryCondition) = XML.Element(
    "D"; VAR=vbc.variable, VALUE=vbc.value
)

variable_boundary_condition_element(vbc::NeumannVariableBoundaryCondition) = XML.Element(
    "N"; VAR=vbc.variable, VALUE=vbc.value
)

function region_boundary_condition_element(region_boundary_condition, id)
    return XML.Element(
        "REGION",
        (
            variable_boundary_condition_element(vbc) 
            for vbc in region_boundary_condition.variable_boundary_conditions
        )...;
        REF=string(id - 1),
    ) 
end

function boundary_conditions_element(region_boundary_conditions)
    return XML.Element(
        "BOUNDARYCONDITIONS",
        (
            region_boundary_condition_element(rbc, i) 
            for (i, rbc) in enumerate(region_boundary_conditions)
        )...
    ) 
end

abstract type AbstractFieldDefinition end
    
struct ExpressionFieldDefinition <: AbstractFieldDefinition
    variables::String
    expression::String
end

struct FileFieldDefinition <: AbstractFieldDefinition
    variables::String
    file::String
end

struct FieldFunction{T <: AbstractFieldDefinition}
    name::String 
    field_definitions::Vector{T}
end

field_definition_element(f::FileFieldDefinition) = XML.Element("F"; VAR=f.variables, FILE=f.file)
field_definition_element(f::ExpressionFieldDefinition) = XML.Element("E"; VAR=f.variables, VALUE=f.expression)

function function_element(field_function::FieldFunction)
    return XML.Element(
        "FUNCTION",
        (field_definition_element(f) for f in field_function.field_definitions)...;
        NAME=field_function.name
    )
end

function conditions_element(
    solver_properties, parameters, variables, boundary_conditions, functions
)
    return XML.Element(
        "CONDITIONS",
        solver_info_element(solver_properties),
        parameters_element(parameters),
        variables_element(variables),
        boundary_regions_element((bc.composite for bc in boundary_conditions)),
        boundary_conditions_element(boundary_conditions),
        (function_element(func) for func in functions)...
    )
end

function nektar_element(
    variables, num_modes, solver_properties, parameters, boundary_conditions, functions
)
    return XML.Element(
        "NEKTAR",
        collections_element(),
        expansions_element(variables, num_modes),
        conditions_element(
            solver_properties, parameters, variables, boundary_conditions, functions
        )
    )
end

function make_nektar_conditions_file(
    output_path; variables, num_modes, solver_properties, parameters, boundary_conditions, functions
)
    document = XML.Document(
        XML.Declaration(; version="1.0", encoding="utf-8"),
        nektar_element(
            variables, num_modes, solver_properties, parameters, boundary_conditions, functions,
        )
    )
    XML.write(output_path, document)
end

function periodic_boundary_conditions(variables)
    return [
        BoundaryCondition("C[1]", [PeriodicVariableBoundaryCondition(v, 3) for v in variables]),
        BoundaryCondition("C[2]", [PeriodicVariableBoundaryCondition(v, 4) for v in variables]),
        BoundaryCondition("C[3]", [PeriodicVariableBoundaryCondition(v, 1) for v in variables]),
        BoundaryCondition("C[4]", [PeriodicVariableBoundaryCondition(v, 2) for v in variables])
    ]
end

function make_driftwave_conditions_file(output_path, previous_state_path, parameters)
    variables = ["zeta", "n", "phi"]
    make_nektar_conditions_file(
        output_path,
        variables=variables,
        num_modes=parameters.num_modes,
        solver_properties=(; 
            EQTYPE = "DriftWaveSystem",
            Projection = "DisContinuous",
            TimeIntegrationMethod = "ClassicalRungeKutta4",
        ),
        parameters=(; 
            NumSteps = parameters.num_steps_per_observation_time,
            TimeStep = parameters.time_step,
            kappa = parameters.kappa,
            alpha = parameters.alpha,
            IO_InfoSteps = 0,
            IO_CheckSteps = 0,
        ),
        boundary_conditions=periodic_boundary_conditions(variables),
        functions=[
            FieldFunction(
                "InitialConditions", 
                [FileFieldDefinition(join(variables, ","), previous_state_path)]
            )
        ]
    )
end

function make_helmholtz_conditions_file(output_path, forcing_field_path, parameters)
    variables = ["u"]
    forcing_field_definition = (
        isnothing(forcing_field_path) 
        ? ExpressionFieldDefinition(only(variables), "awgn(1)")
        : FileFieldDefinition(only(variables), forcing_field_path) 
    )
    make_nektar_conditions_file(
        output_path,
        variables=variables,
        num_modes=parameters.num_modes,
        solver_properties=(; EQTYPE = "Helmholtz", Projection = "Continuous"),
        parameters=(; lambda = 1 / parameters.state_grf_length_scale^2),
        boundary_conditions=periodic_boundary_conditions(variables),
        functions=[FieldFunction("Forcing", [forcing_field_definition])]
    )
end

function make_poisson_conditions_file(output_path, forcing_field_path, parameters)
    # TODO: Identify why we get numerical issues when using discontinuous projection
    variables = ["u"]
    make_nektar_conditions_file(
        output_path,
        variables=variables,
        num_modes=parameters.num_modes,
        solver_properties=(; EQTYPE = "Poisson", Projection = "Continuous"),
        parameters=(;),
        boundary_conditions=periodic_boundary_conditions(variables),
        functions=[
            FieldFunction(
                "Forcing", [FileFieldDefinition(only(variables), forcing_field_path)]
            )
        ]
    )
end

struct NektarExecutablePaths
    adr_solver::String
    driftwave_solver::String
    field_convert::String
    nek_mesh::String
end

function NektarExecutablePaths(parameters::NektarDriftwaveModelParameters)
    return NektarExecutablePaths(
        joinpath(parameters.nektar_bin_directory, "ADRSolver"),
        joinpath(parameters.driftwave_solver_bin_directory, "DriftWaveSolver"),
        joinpath(parameters.nektar_bin_directory, "FieldConvert"),
        joinpath(parameters.nektar_bin_directory, "NekMesh"),
    )
end

struct NektarConditionsFilePaths
    driftwave::String
    grf::String
    grf_recursion::String
    poisson::String
end

function NektarConditionsFilePaths(parent_directory::String)
    return NektarConditionsFilePaths(
        joinpath(parent_directory, "driftwave.xml"),
        joinpath(parent_directory, "grf.xml"),
        joinpath(parent_directory, "grf_recursion.xml"),
        joinpath(parent_directory, "poisson.xml")
    )
end

get_field_file_path(conditions_file_path) = change_extension(conditions_file_path, "fld")

struct NektarDriftwaveModel{S <: Real, T <: Real}
    parameters::NektarDriftwaveModelParameters{S, T}
    executable_paths::NektarExecutablePaths
    root_working_directory::String
    mesh_file_paths::MeshFilePaths
    observed_points_file_path::String
    task_working_directories::Vector{String}
    task_conditions_file_paths::Vector{NektarConditionsFilePaths}
    observation_noise_distribution::MvNormal{T}
end

function make_observed_points_file(parameters, root_working_directory)
    observed_points_file_path = joinpath(root_working_directory, "observed_points.csv")
    observed_points_table = Tables.table(
        # In Julia 1.9+ we can use stack(...; dims=1) but we use hcat for compatibility
        reduce(hcat, parameters.observed_points)'
    )
    CSV.write(observed_points_file_path, observed_points_table; header=["# x", "y"])
    return observed_points_file_path
end

function generate_gaussian_random_field_file(model, task_index, variable, noise_scale, mean_expression=nothing)
    conditions_file_paths = model.task_conditions_file_paths[task_index]
    grf_field_file_path = get_field_file_path(conditions_file_paths.grf)
    grf_recursion_field_file_path = get_field_file_path(conditions_file_paths.grf_recursion)
    variable_field_path = joinpath(model.task_working_directories[task_index], "$(variable).fld")
    # Whittle-Matérn Gaussian random field variance for spatial dimension 2 is 
    # Γ(ν) / (Γ(ν + 1) * κ^2ν * 4π) = 1 / (ν * κ^2ν * 4π)
    # Therefore multiply noise_scale by sqrt(ν * κ^2ν * 4π) so that noise_scale = 1
    # corresponds to unit variance
    ν = model.parameters.state_grf_smoothness * 2 - 1
    κ = 1 / model.parameters.state_grf_length_scale
    noise_scale *= sqrt(ν * κ^2ν * 4π)
    if isnothing(mean_expression)
        field_expression_string = "$(noise_scale) * u"
    else
        field_expression_string = "$(mean_expression) + $(noise_scale) * u"
    end
    cd(model.task_working_directories[task_index]) do
        run(`$(model.executable_paths.adr_solver) -f -i Hdf5 $(conditions_file_paths.grf) $(model.mesh_file_paths.no_expansions)`)
        for i in 1:(model.parameters.state_grf_smoothness - 1)
            run(`$(model.executable_paths.adr_solver) -f -i Hdf5 $(conditions_file_paths.grf_recursion) $(model.mesh_file_paths.no_expansions)`)
            mv(grf_recursion_field_file_path, grf_field_file_path; force=true)
        end
        run(`$(model.executable_paths.field_convert) -f -m fieldfromstring:fieldstr="$(field_expression_string)":fieldname="$(variable)" $(model.mesh_file_paths.with_expansions) $(grf_field_file_path) $(variable_field_path):fld:format=Hdf5`)
        run(`$(model.executable_paths.field_convert) -f -m removefield:fieldname="u" $(model.mesh_file_paths.with_expansions) $(variable_field_path) $(variable_field_path):fld:format=Hdf5`)
    end
    return variable_field_path
end

function concatenate_fields(model, field_file_paths, concatenated_field_file_path)
    run(`$(model.executable_paths.field_convert) -f $(model.mesh_file_paths.with_expansions) $(field_file_paths) $(concatenated_field_file_path):fld:format=Hdf5`)
end

function add_fields(model, field_file_path_1, field_file_path_2, output_field_file_path)
    run(`$(model.executable_paths.field_convert) -f -m addfld:fromfld=$(field_file_path_1) $(model.mesh_file_paths.with_expansions) $(field_file_path_2) $(output_field_file_path):fld:format=Hdf5`)
end

function update_phi(model, task_index)
    conditions_file_paths = model.task_conditions_file_paths[task_index]
    poisson_field_file_path = get_field_file_path(conditions_file_paths.poisson)
    driftwave_field_file_path = get_field_file_path(conditions_file_paths.driftwave)
    cd(model.task_working_directories[task_index]) do
        run(`$(model.executable_paths.adr_solver) -f -i Hdf5 $(conditions_file_paths.poisson) $(model.mesh_file_paths.no_expansions)`)
        run(`$(model.executable_paths.field_convert) -f $(model.mesh_file_paths.with_expansions) $(driftwave_field_file_path) $(poisson_field_file_path) $(driftwave_field_file_path):fld:format=Hdf5`)
        run(`$(model.executable_paths.field_convert) -f -m fieldfromstring:fieldstr="u":fieldname="phi" $(model.mesh_file_paths.with_expansions) $(driftwave_field_file_path) $(driftwave_field_file_path):fld:format=Hdf5`)
        run(`$(model.executable_paths.field_convert) -f -m removefield:fieldname="u" $(model.mesh_file_paths.with_expansions) $(driftwave_field_file_path) $(driftwave_field_file_path):fld:format=Hdf5`)
    end
end

function check_fields(field_file)
    field_info_keys = setdiff(keys(field_file["NEKTAR"]), ["DATA", "DECOMPOSITION", "ELEMENTIDS", "Metadata"])
    @assert length(field_info_keys) == 1 "Multiple field information groups present in HDF5 file"
    @assert read_attribute(field_file["NEKTAR"][field_info_keys[1]], "FIELDS") == ["zeta", "n", "phi"] "Field variables in unexpected order"
end

function write_state_to_field_file(field_file_path, state)
    h5open(field_file_path, "r+") do field_file
        check_fields(field_file)
        field_file["NEKTAR"]["DATA"][:] = state
    end
end

function read_state_from_field_file!(state, field_file_path)
    h5open(field_file_path, "r") do field_file
        check_fields(field_file)
        state .= field_file["NEKTAR"]["DATA"][:]
    end    
end

function interpolate_field_at_observed_points(model, task_index, field_file_path)
    interpolated_field_file_path = joinpath(model.task_working_directories[task_index], "interpolated_field.csv")
    cd(model.task_working_directories[task_index]) do
        run(`$(model.executable_paths.field_convert) -f -m interppoints:fromxml=$(model.mesh_file_paths.with_expansions):fromfld=$(field_file_path):topts=$(model.observed_points_file_path) $(interpolated_field_file_path)`)
    end
    interpolated_field_data = CSV.File(interpolated_field_file_path; select=(i, name) -> String(name) in model.parameters.observed_variables)
    return Tables.matrix(interpolated_field_data)
end

function distribution_observation_given_state(state, model, task_index)
    conditions_file_paths = model.task_conditions_file_paths[task_index]
    driftwave_field_file_path = get_field_file_path(conditions_file_paths.driftwave)
    write_state_to_field_file(driftwave_field_file_path, state)
    field_at_observed_points = interpolate_field_at_observed_points(model, task_index, driftwave_field_file_path)
    return field_at_observed_points[:] + model.observation_noise_distribution
end

function init(
    parameters_dict::Dict,
    n_tasks::Int=1;
    S::Type{<:Real}=Float64,
    T::Type{<:Real}=Float64
)
    parameters = get_params(NektarDriftwaveModelParameters{S, T}, parameters_dict)
    executable_paths = NektarExecutablePaths(parameters)
    root_working_directory = mktempdir(; prefix="jl_ParticleDA_nektar_driftwave_")
    task_working_directories = [
        joinpath(root_working_directory, "task_$(t)") for t in 1:n_tasks
    ]
    task_conditions_file_paths = [
        NektarConditionsFilePaths(task_working_directory)
        for task_working_directory in task_working_directories
    ]
    mesh_file_paths = make_mesh_files(parameters, root_working_directory, executable_paths.nek_mesh)
    observed_points_file_path = make_observed_points_file(parameters, root_working_directory)
    for (task_working_directory, conditions_file_paths) in zip(task_working_directories, task_conditions_file_paths)
        mkdir(task_working_directory)
        driftwave_field_file_path = get_field_file_path(conditions_file_paths.driftwave)
        grf_field_file_path = get_field_file_path(conditions_file_paths.grf)
        make_driftwave_conditions_file(conditions_file_paths.driftwave, driftwave_field_file_path, parameters)
        make_helmholtz_conditions_file(conditions_file_paths.grf, nothing, parameters)
        make_helmholtz_conditions_file(conditions_file_paths.grf_recursion, grf_field_file_path, parameters)
        make_poisson_conditions_file(conditions_file_paths.poisson, "$(driftwave_field_file_path):zeta", parameters)
    end
    observation_dimension = length(parameters.observed_points)
    observation_noise_distribution = MvNormal(
        zeros(T, observation_dimension),
        ScalMat(observation_dimension, parameters.observation_noise_std^2)
    )
    return NektarDriftwaveModel(
        parameters,
        executable_paths,
        root_working_directory,
        mesh_file_paths,
        observed_points_file_path,
        task_working_directories,
        task_conditions_file_paths,
        observation_noise_distribution
    )
end

ParticleDA.get_state_dimension(model::NektarDriftwaveModel) = (
    model.parameters.mesh_dims[1] * model.parameters.mesh_dims[1] * model.parameters.num_modes^2 * 3
)
ParticleDA.get_observation_dimension(model::NektarDriftwaveModel) = length(
    model.parameters.observed_points
)
ParticleDA.get_state_eltype(::NektarDriftwaveModel{S, T}) where {S, T} = S
ParticleDA.get_observation_eltype(::NektarDriftwaveModel{S, T}) where {S, T} = T

function ParticleDA.sample_initial_state!(
    state::AbstractVector{T},
    model::NektarDriftwaveModel{S, T}, 
    rng::Random.AbstractRNG,
    task_index::Integer=1
) where {S, T}
    conditions_file_paths = model.task_conditions_file_paths[task_index]
    driftwave_field_file_path = get_field_file_path(conditions_file_paths.driftwave)
    s = model.parameters.initial_state_mean_length_scale
    variable_mean_expressions = [
        "zeta" => "4*exp((-x*x-y*y)/($(s^2)))*(-$(s^2)+x*x+y*y)/$(s^4)",
        "n" => "exp((-x*x-y*y)/$(s^2))",
    ]
    variable_field_file_paths = [
        generate_gaussian_random_field_file(
            model, task_index, variable, model.parameters.initial_state_grf_output_scale, mean_expression
        )
        for (variable, mean_expression) in variable_mean_expressions
    ]
    concatenate_fields(model, variable_field_file_paths, driftwave_field_file_path)
    update_phi(model, task_index)
    read_state_from_field_file!(state, driftwave_field_file_path)
end

function ParticleDA.update_state_deterministic!(
    state::AbstractVector{T}, 
    model::NektarDriftwaveModel{S, T}, 
    time_index::Integer,
    task_index::Integer=1
) where {S, T}
    conditions_file_paths = model.task_conditions_file_paths[task_index]
    driftwave_field_file_path = get_field_file_path(conditions_file_paths.driftwave)
    write_state_to_field_file(driftwave_field_file_path, state)
    cd(model.task_working_directories[task_index]) do
        run(`$(model.executable_paths.driftwave_solver) -f -i Hdf5 $(conditions_file_paths.driftwave) $(model.mesh_file_paths.no_expansions)`)
    end
    read_state_from_field_file!(state, driftwave_field_file_path)
end

function ParticleDA.update_state_stochastic!(
    state::AbstractVector{T}, 
    model::NektarDriftwaveModel{S, T}, 
    rng::Random.AbstractRNG,
    task_index::Integer=1
) where {S, T}
    conditions_file_paths = model.task_conditions_file_paths[task_index]
    driftwave_field_file_path = get_field_file_path(conditions_file_paths.driftwave)
    noise_field_file_path = joinpath(model.task_working_directories[task_index], "noise.fld")
    write_state_to_field_file(driftwave_field_file_path, state)
    variable_field_file_paths = [
        generate_gaussian_random_field_file(
            model, task_index, variable, model.parameters.state_noise_grf_output_scale
        )
        for variable in ["zeta", "n"]
    ]
    concatenate_fields(model, variable_field_file_paths, noise_field_file_path)
    add_fields(model, driftwave_field_file_path, noise_field_file_path, driftwave_field_file_path)
    update_phi(model, task_index)
    read_state_from_field_file!(state, driftwave_field_file_path)
end
    
function ParticleDA.sample_observation_given_state!(
    observation::AbstractVector{T},
    state::AbstractVector{S}, 
    model::NektarDriftwaveModel{S, T}, 
    rng::Random.AbstractRNG,
    task_index::Integer=1 
) where {S <: Real, T <: Real}
    return rand!(rng, distribution_observation_given_state(state, model, task_index), observation)
end

function ParticleDA.get_log_density_observation_given_state(
    observation::AbstractVector{T},
    state::AbstractVector{S},
    model::NektarDriftwaveModel{S, T},
    task_index::Integer=1
) where {S <: Real, T <: Real}
    return logpdf(distribution_observation_given_state(state, model, task_index), observation)
end

function ParticleDA.write_model_metadata(file::HDF5.File, model::NektarDriftwaveModel)
    group_name = "parameters"
    if !haskey(file, group_name)
        group = create_group(file, group_name)
        for field in fieldnames(typeof(model.parameters))
            value = getfield(model.parameters, field)
            HDF5.attributes(group)[string(field)] = (
                isa(value, Vector{Vector{Float64}}) ? hcat(value...)' : value
            )
        end
    else
        @warn "Write failed, group $group_name already exists in  $(file.filename)!"
    end
end

function ParticleDA.write_state(
    file::HDF5.File,
    state::AbstractVector,
    time_index::Int,
    group_name::String,
    model
)
    time_stamp = ParticleDA.time_index_to_hdf5_key(time_index)
    group, _ = ParticleDA.create_or_open_group(file, group_name)
    conditions_file_paths = model.task_conditions_file_paths[1]
    driftwave_field_file_path = get_field_file_path(conditions_file_paths.driftwave)
    h5open(driftwave_field_file_path, "r+") do field_file
        check_fields(field_file)
        field_file["NEKTAR"]["DATA"][:] = state
        if !haskey(group, time_stamp)
            time_stamp_group = create_group(group, time_stamp)
            copy_object(field_file["NEKTAR"], time_stamp_group, "NEKTAR")
        else
            @warn "Write failed, timestamp $time_stamp already exists in $group"
        end
    end
end

end
