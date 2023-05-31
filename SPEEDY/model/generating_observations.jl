using FortranFiles
using Dates
using HDF5
using Random
using Distributions
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
        if it > 0
            # These are written only after the initial state
            write_obs(file, observation, it)
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


IDate="1982010106"
dtDate="1982010112"
endDate = "1982080100"
SPEEDY= ""
obs_network ="uniform"
SPEEDY_DATE_FORMAT = "YYYYmmddHH"
dt = 6
delta = (DateTime(endDate, SPEEDY_DATE_FORMAT) - DateTime(IDate, SPEEDY_DATE_FORMAT))
num_timesteps = Dates.Hour(delta)/dt
nobs = 50

station_filename = joinpath(SPEEDY, "obs", "networks", obs_network * ".txt") 
nature_dir = joinpath(SPEEDY, "DATA", "nature")
obs_indices = [33]
array = zeros(96,48,34)


station_grid_indices = get_station_grid_indices(station_filename, nobs)
indices = hcat(station_grid_indices[:,1], station_grid_indices[:,2])

obs_dim = (size(station_grid_indices, 1) * length(obs_indices))
observation = zeros(obs_dim)
dates = [IDate,dtDate]

for time in 1:num_timesteps.value+1
    truth_file = joinpath(nature_dir, dates[1] * ".grd")
    read_grd!(array, truth_file, 96, 48, 8)
    for k in obs_indices
        for i in 1:nobs
            observation[i] = array[indices[i,1], indices[i,2], k]
        end
    end
    dates[1], dates[2] = step_datetime(dates[1],dates[2])
    write_state_and_observations("", observation, time)

end





