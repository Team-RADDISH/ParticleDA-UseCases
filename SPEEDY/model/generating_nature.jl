using FortranFiles
using Dates
using HDF5

using ParticleDA
const SPEEDY_DATE_FORMAT = "YYYYmmddHH"


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


function step_datetime(idate::String,dtdate::String)
    new_idate = Dates.format(DateTime(idate, SPEEDY_DATE_FORMAT) + Dates.Hour(6), SPEEDY_DATE_FORMAT)
    new_dtdate = Dates.format(DateTime(dtdate, SPEEDY_DATE_FORMAT) + Dates.Hour(6), SPEEDY_DATE_FORMAT)
    return new_idate,new_dtdate
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

function write_state_and_observations(filename::String, observation::AbstractArray{T},
    it::Int) where T

    println("Writing output at timestep = ", it)
    h5open(filename, "cw") do file
        if it >= 0
            # These are written only after the initial state
            write_state(file, observation, it)
        end
    end
end



function write_obs(file::HDF5.File, observation::AbstractVector, it::Int)

    group_name = "observations"
    dataset_name = "t" * lpad(string(it),4,'0')

    group, subgroup = ParticleDA.create_or_open_group(file, group_name)

    if !haskey(group, dataset_name)
        #TODO: use d_write instead of create_dataset when they fix it in the HDF5 package
        ds,dtype = create_dataset(group, dataset_name, observation)
        ds[:] = observation
        attributes(ds)["Description"] = "Observations"
        attributes(ds)["Unit"] = ""
        attributes(ds)["Time step"] = it
    else
        @warn "Write failed, dataset $group_name/$dataset_name  already exists in $(file.filename) !"
    end

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


function write_state(
    file::HDF5.File,
    state::AbstractArray{T},
    time_index::Int,
) where T
    # model_params = model_data.model_params
    group_name = "data_nature"
    subgroup_name = "t" * lpad(string(time_index), 4, '0')
    _, subgroup = ParticleDA.create_or_open_group(file, group_name, subgroup_name)
    state_fields_metadata = [
        (name="u1", level = "1", unit="m/s", description="Velocity zonal-component"),
        (name="u2", level = "2", unit="m/s", description="Velocity zonal-component"),
        (name="u3", level = "3", unit="m/s", description="Velocity zonal-component"),
        (name="u4", level = "4", unit="m/s", description="Velocity zonal-component"),
        (name="u5", level = "5", unit="m/s", description="Velocity zonal-component"),
        (name="u6", level = "6", unit="m/s", description="Velocity zonal-component"),
        (name="u7", level = "7", unit="m/s", description="Velocity zonal-component"),
        (name="u8", level = "8", unit="m/s", description="Velocity zonal-component"),
        (name="v1", level = "1", unit="m/s", description="Velocity meridional-component"),
        (name="v2", level = "2", unit="m/s", description="Velocity meridional-component"),
        (name="v3", level = "3", unit="m/s", description="Velocity meridional-component"),
        (name="v4", level = "4", unit="m/s", description="Velocity meridional-component"),
        (name="v5", level = "5", unit="m/s", description="Velocity meridional-component"),
        (name="v6", level = "6", unit="m/s", description="Velocity meridional-component"),
        (name="v7", level = "7", unit="m/s", description="Velocity meridional-component"),
        (name="v8", level = "8", unit="m/s", description="Velocity meridional-component"),
        (name="T1", level = "1", unit="K", description="Temperature"),
        (name="T2", level = "2", unit="K", description="Temperature"),
        (name="T3", level = "3", unit="K", description="Temperature"),
        (name="T4", level = "4", unit="K", description="Temperature"),
        (name="T5", level = "5", unit="K", description="Temperature"),
        (name="T6", level = "6", unit="K", description="Temperature"),
        (name="T7", level = "7", unit="K", description="Temperature"),
        (name="T8", level = "8", unit="K", description="Temperature"),
        (name="Q1", level = "1", unit="kg/kg", description="Specific Humidity"),
        (name="Q2", level = "2", unit="kg/kg", description="Specific Humidity"),
        (name="Q3", level = "3", unit="kg/kg", description="Specific Humidity"),
        (name="Q4", level = "4", unit="kg/kg", description="Specific Humidity"),
        (name="Q5", level = "5", unit="kg/kg", description="Specific Humidity"),
        (name="Q6", level = "6", unit="kg/kg", description="Specific Humidity"),
        (name="Q7", level = "7", unit="kg/kg", description="Specific Humidity"),
        (name="Q8", level = "8", unit="kg/kg", description="Specific Humidity"),
        (name="ps", level = "1", unit="Pa", description="Surface Pressure"),
        (name="rain", level = "1", unit="mm/hr", description="Rain")
    ]
    for (field, metadata) in zip(eachslice(state, dims=3), state_fields_metadata)
        if !haskey(subgroup, metadata.name) && !haskey(subgroup, metadata.level)
            subgroup[metadata.name] = field
            dataset_attributes = attributes(subgroup[metadata.name])
            dataset_attributes["Description"] = metadata.description
            dataset_attributes["Level"] = metadata.level
            dataset_attributes["Unit"] = metadata.unit
            dataset_attributes["Time step"] = time_index
        else
            @warn "Write failed, dataset $group_name/$subgroup_name/$(metadata.name) already exists in $(file.filename) !"
        end
    end
end

IDate="1982010100"
dtDate="1982010103"
endDate = "1982080100"
SPEEDY= ""
SPEEDY_DATE_FORMAT = "YYYYmmddHH"
delta = (DateTime(endDate, SPEEDY_DATE_FORMAT)- DateTime(IDate, SPEEDY_DATE_FORMAT))
num_timesteps = Dates.Hour(delta)/3

nature_dir = joinpath(SPEEDY, "DATA", "nature")
array = zeros(96,48,34)
dates = [IDate,dtDate]

for time in 0:num_timesteps.value
    truth_file = joinpath(nature_dir, dates[1] * ".grd")
    read_grd!(array, truth_file, 96, 48, 8)
    write_state_and_observations("", array, time)
    dates[1], dates[2] = step_datetime(dates[1],dates[2])
end





