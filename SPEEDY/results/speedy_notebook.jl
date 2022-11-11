### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ c7383ff4-ba4c-11eb-1977-b31b330b20d0
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
	using Plots
	using HDF5
	using Unitful
	using UnitfulRecipes
	using PlutoUI
	using Statistics
	using Plots.PlotMeasures
	using GeoDatasets
	using Dates
end

# ╔═╡ 577046b8-9d3e-429f-ab04-476f2b9f946c
md"# ParticleDA"

# ╔═╡ d58501b3-c2d2-49d9-a834-7efb1ad83774
md"## Model Set Up"

# ╔═╡ dd4cf55d-0d1f-4436-9ba3-30a0c641ccc1
md"To recap we assume the following dynamics, for $t \geq 1$,

$\begin{align}
x_{t} &= F(x_{t-1}) + \xi_t,  \quad x_0\sim p(dx_0); \\
y_{t} &= A\,x_{t} + \zeta_{t},
\end{align}$
 
To showcase the generality of the platform and the agnostic nature to underlying PDE. We have integrated it with SPEEDY (Simplified Parameterizations, primitivE-Equation DYnamics), which is a simplified atmposheric circulation model
"

# ╔═╡ 1a86d391-6063-42c4-bb3f-a6aadd1f923f
md"## SPEEDY"

# ╔═╡ bc29f40d-5761-495d-b0b1-9a4d0a76263b
md"- Spectral primitive-equation dynamic core along with a set of physical parameterization schemes 
- Similar to current state-of-the-art atmospheric general circulation models but requires drastically less computational resources
- Model has been utilised to carry out large ensemble and/or data assimilation experiments
- The tsunami model was written within the Julia platform which allowed for easy integration (test case) with the Particle Filtering but SPEEDY is written in Fortran
- The software considerations include:
	1. Data transfer between the codes
	2. Calling the SPEEDY model from Julia"

# ╔═╡ 4b70238f-00b3-4803-9615-7c3ca4b80fc4
md"## Data assimilation experiment"

# ╔═╡ e866aade-b966-4061-81eb-43197bbf9065
md"- The prognostic variables of the model consist of the zonal and meridional wind velocity components (u,v), temperature (T), specific humidity (q) and surface pressure (ps)
- Spatial resolution is 96 × 48 x 8 grid
- Initial conditions are randomised states of the atmosphere coming from a long term nature run
- 50 observation stations
- Assimilating the surface pressure every 6 hours
- Number of particles: N = 128
"

# ╔═╡ 33ad3104-1f4c-42b6-8fed-02ce618eb348
md"## Results"

# ╔═╡ fdfd7e5d-be64-44d9-b151-d130a1fdcc02
field = "ps"

# ╔═╡ 8f67e2b3-9a01-42a3-a56a-11b84776a5e1
md"#### Scatter plot of particle weights"

# ╔═╡ e5335e14-bdb6-432c-94ab-c666c304efc6
md"#### Time series of Estimated Sample Size"

# ╔═╡ 145b6140-b315-45dc-aca7-838d2c675ada
md"# Future Work"

# ╔═╡ 979d0ff9-d4ed-43f1-a270-c49d036cf71f
md"1. Theory
	- Extend the optimal filter to other settings (unstructured meshes)
	- Add localisation features (Couple with LETKF)
2.  Usability
	- Integration with other models"

# ╔═╡ d09cdbb1-46ce-48d7-9c3a-90febdf4935b
md"Function Defintions"

# ╔═╡ 7eda4250-a24e-4ce1-b966-5e1284885322
md"### Error Calculations
Root mean square error over the whole domain"

# ╔═╡ 9579c953-046f-4bb9-bc8e-0aedb0cff7a7
md"### Read in the Output File and the Ground Truth"

# ╔═╡ 680290e4-1201-47f9-8041-c49b0d1159a0
begin 
	speedy_filename = "prt_128_Jan_Aug.h5"
	speedy = h5open(speedy_filename, "r");
end 

# ╔═╡ e1acfe11-0ead-4ada-946e-eff674e6d44e
begin
	timestamp = keys(speedy["state_avg"])
	# levels = [1,2,3,4,5,6,7,8]
	md"""
	Select the timestamp
	$(@bind timestamp_idx Slider(1:length(timestamp)))
	"""
end

# ╔═╡ 424431dc-12d4-4362-a121-52d1017488ff
begin
	const SPEEDY_DATE_FORMAT = "YYYYmmddHH"
	int_time = parse(Int64,split(timestamp[timestamp_idx],'t')[2]);
	time = timestamp[timestamp_idx]
	idate = "1982010100";
    date = DateTime(idate, SPEEDY_DATE_FORMAT) + Dates.Hour(6*int_time);
end

# ╔═╡ a35f5895-5066-4e1d-b252-6172888aa92d
begin
	weights = read(speedy["weights"][time])

	pwe = Plots.scatter(weights, marker=:star, label="Particles")
	#p2 = scatter(weights, marker=:star, yscale=:log10)

	#for plt in (p1, p2)
	Plots.plot!(pwe; xlabel="Particle ID", ylabel="Weight")
	#end

	Plots.plot(pwe, label="Particles")
end

# ╔═╡ 343a1d50-38f8-4457-81dc-5d962a2acb4a
Plots.plot([1 / sum(read(w) .^ 2) for w in speedy["weights"]];
     label="", marker=:o, xlabel="Time step", ylabel="Estimated Sample Size (1 / sum(weight^2))")

# ╔═╡ 6c501b02-eb73-48c1-88ae-0d72b53ef600
begin
	field_unit = read(speedy["state_avg"][time][field]["Unit"])
	x_unit = read(speedy["grid_coordinates"]["lon"]["Unit"])
	y_unit = read(speedy["grid_coordinates"]["lat"]["Unit"])
	x_st_unit = read(speedy["station_coordinates"]["lon"]["Unit"])
	y_st_unit = read(speedy["station_coordinates"]["lat"]["Unit"])
	
	field_desc = read(speedy["state_avg"][time][field]["Description"])
	
	x = read(speedy["grid_coordinates"]["lon"]) .* uparse(x_unit) .|> u"°"
	y = read(speedy["grid_coordinates"]["lat"]) .* uparse(y_unit) .|> u"°"
	z_avg = read(speedy["state_avg"][time][field]) .* uparse(field_unit)
	x_st = read(speedy["station_coordinates"]["lon"]) .* uparse(x_st_unit) .|> u"°"
	y_st = read(speedy["station_coordinates"]["lat"]) .* uparse(y_st_unit) .|> u"°"
end

# ╔═╡ 78a6d939-19f0-46a7-9bba-e1045051af53
function plot_mean_error_rmse(x, y, z_t, z_avg, rmse, date, timestamp, lon, lat, data)
    z_err = ((z_t-z_avg)./z_t).*100
    p1 = heatmap(x, y, transpose(z_avg); title="Mean assimilated Surface Pressure", annotations = (10, 10, Plots.text(string(date), :left)))
    p1 = Plots.contour!(lon,lat, (data.*10000)', c = :blues)
    p2 = heatmap(x, y, transpose(z_err); title="Surface Pressure % error", colorbar_title="% Error", annotations = (10, 10, Plots.text(string(date), :left)))
    p2 = Plots.contour!(lon,lat, data', c = :blues)
    p3 = Plots.plot(rmse, label="", marker=:o, xlabel="Time step", ylabel="RMSE", title="RMSE of Surface Pressure over domain")
    p3 = vline!([timestamp], label=date)
    special_gauges = [47 12; 50 38; 14 41; 29 26; 3 44]
    for (i, plt) in enumerate((p1, p2, p3))
        # Set labels
        i ∈ (1,2) && Plots.plot!(plt; xlabel="Lon", ylabel="Lat", labelfontsize=18)
        i ∈ (3) && Plots.plot!(plt; xlabel="Timestep", ylabel="RMSE", labelfontsize=18)
        # Add the positions of the stations
        i ∈ (1,2) && Plots.scatter!(plt, x_st, y_st, color=:black, marker=:star, label="Observation Locations", labelfontsize=20)
    end
    l = @layout[grid(3, 1)]
    Plots.plot(p1, p2, p3, layout = l, titlefontsize=24, guidefontsize=24, colorbar_titlefontsize = 12, legendfontsize=12, legendtitlefontsize=12, tickfontsize = 12, left_margin = [10mm 10mm], bottom_margin = 30px)
    Plots.plot!(size=(1200,1800))
end

# ╔═╡ b5f4c549-0946-4fc4-af1e-b1cdf8c848d0
begin 
	filename = "nature_Jan_Aug.h5";
	th = h5open(filename, "r");
	z_truth = read(th["data_nature"][time][field]) .* uparse(field_unit);
end

# ╔═╡ 6d53ebcb-f783-4cac-8dd0-5a3ebeeed2b1
function error_calculation(z_truth, z_avg)
	rmse = Float64[].* uparse(field_unit)
	for it = 0:length(keys(speedy["state_avg"]))-1
	    timestamps = "t" * lpad(string(it),4,'0')
	    z_truth = read(th["data_nature"][timestamps][field]) .* uparse(field_unit)
	    z_avg = read(speedy["state_avg"][timestamps][field]) .* uparse(field_unit)
	    error = sqrt.(mean((z_truth[:,:].-z_avg[:,:]).^2))
	    push!(rmse, error)
	end
	return rmse
end

# ╔═╡ c211578c-540e-4487-b6c4-fa7b223f6561
begin
	lon,lat,data = GeoDatasets.landseamask(;resolution='c',grid=10)
	coast = zeros(size(data))
	coast[1:1080, 1:1080] .= data[1081:2160,1:1080]
	coast[1081:2160,1:1080] .= data[1:1080, 1:1080]
	lon = lon .+ 180
	lat = lat .+ 90
	x_plot = range(0,360,96)
	y_plot = range(0,180,48)
	rmse = error_calculation(z_truth, z_avg)
	plot_mean_error_rmse(x_plot, y_plot, z_truth[:,:], z_avg[:,:], rmse, date, int_time, lon, lat, coast)
end

# ╔═╡ Cell order:
# ╟─577046b8-9d3e-429f-ab04-476f2b9f946c
# ╟─c7383ff4-ba4c-11eb-1977-b31b330b20d0
# ╟─d58501b3-c2d2-49d9-a834-7efb1ad83774
# ╟─dd4cf55d-0d1f-4436-9ba3-30a0c641ccc1
# ╟─1a86d391-6063-42c4-bb3f-a6aadd1f923f
# ╟─bc29f40d-5761-495d-b0b1-9a4d0a76263b
# ╟─4b70238f-00b3-4803-9615-7c3ca4b80fc4
# ╟─e866aade-b966-4061-81eb-43197bbf9065
# ╟─33ad3104-1f4c-42b6-8fed-02ce618eb348
# ╟─e1acfe11-0ead-4ada-946e-eff674e6d44e
# ╟─424431dc-12d4-4362-a121-52d1017488ff
# ╟─fdfd7e5d-be64-44d9-b151-d130a1fdcc02
# ╟─c211578c-540e-4487-b6c4-fa7b223f6561
# ╟─8f67e2b3-9a01-42a3-a56a-11b84776a5e1
# ╟─a35f5895-5066-4e1d-b252-6172888aa92d
# ╟─e5335e14-bdb6-432c-94ab-c666c304efc6
# ╟─343a1d50-38f8-4457-81dc-5d962a2acb4a
# ╟─145b6140-b315-45dc-aca7-838d2c675ada
# ╟─979d0ff9-d4ed-43f1-a270-c49d036cf71f
# ╟─d09cdbb1-46ce-48d7-9c3a-90febdf4935b
# ╟─78a6d939-19f0-46a7-9bba-e1045051af53
# ╟─7eda4250-a24e-4ce1-b966-5e1284885322
# ╟─6d53ebcb-f783-4cac-8dd0-5a3ebeeed2b1
# ╟─9579c953-046f-4bb9-bc8e-0aedb0cff7a7
# ╠═680290e4-1201-47f9-8041-c49b0d1159a0
# ╠═6c501b02-eb73-48c1-88ae-0d72b53ef600
# ╠═b5f4c549-0946-4fc4-af1e-b1cdf8c848d0
