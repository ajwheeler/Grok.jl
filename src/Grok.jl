module Grok
using HDF5, Optim, FITSIO, Interpolations, Korg, ProgressMeter
using DSP: gaussian, conv  # used for vsini and continuum adjustment
using SparseArrays: spzeros # used for crazy continuum adjustment

_data_dir = joinpath(@__DIR__, "../data") 

# TODO addprocs

#####################################################################
# TODO these should be excised from the module
# read the mask that is aplied to all spectra
const ferre_mask = parse.(Bool, readlines(joinpath(_data_dir, "ferre_mask.dat")));
const ferre_wls = (10 .^ (4.179 .+ 6e-6 * (0:8574)))
const regions = [
    (15152.0, 15800.0),
    (15867.0, 16424.0),
    (16484.0, 16944.0), 
]
const region_inds = map(regions) do (lb, ub)
    findfirst(ferre_wls .> lb) : (findfirst(ferre_wls .> ub) - 1)
end
#####################################################################

"""
We don't use Korg's `apply_rotation` because this is specialed for the case where wavelenths are 
log-uniform.  We can use FFT-based convolution to apply the rotation kernel to the spectrum in this 
case.
"""
function apply_rotation(flux, vsini; ε=0.6, log_lambda_step=1.3815510557964276e-5)
    if vsini == 0
        return flux
    end

    # calculate log-lamgda step from the wavelength grid
    # log_lambda_step = mean(diff(log.(ferre_wls)))

    # half-width of the rotation kernel in Δlnλ
    Δlnλ_max = vsini * 1e5 / Korg.c_cgs
    # half-width of the rotation kernel in pixels
    p = Δlnλ_max / log_lambda_step
    # Δlnλ detuning for each pixel in kernel
    Δlnλs = [-Δlnλ_max ; (-floor(p) : floor(p))*log_lambda_step ; Δlnλ_max]
    
    # kernel coefficients
    c1 = 2(1-ε)
    c2 = π * ε / 2
    c3 = π * (1-ε/3)
    
    x = @. (1 - (Δlnλs/Δlnλ_max)^2)
    rotation_kernel = @. (c1*sqrt(x) + c2*x)/(c3 * Δlnλ_max)
    rotation_kernel ./= sum(rotation_kernel)
    offset = (length(rotation_kernel) - 1) ÷ 2

    conv(rotation_kernel, flux)[offset+1 : end-offset]
end

function fill_chip_gaps!(flux)
    Δ = 10 # insurance
    # set the off-chip flux values to be reasonable, so as to not have crazy edge effects
    flux[1:region_inds[1][1]] .= flux[region_inds[1][1] + Δ]
    flux[region_inds[end][end]:end] .= flux[region_inds[end][end] - Δ]
    for (i1, i2) in [(region_inds[1][end], region_inds[2][1]), (region_inds[2][end], region_inds[3][1])]
        flux[i1:i2] .= range(start=flux[i1-Δ], stop=flux[i2+Δ], length=i2-i1 + 1)
    end
end

function apply_smoothing_filter!(flux, kernel_width=150;
                                kernel_pixels=301)
                                
    @assert isodd(kernel_pixels) # required for offset math to work
    sampled_kernel = gaussian(kernel_pixels, kernel_width/kernel_pixels)
    sampled_kernel ./= sum(sampled_kernel) # normalize (everything should cancel in the end, but I'm paranoid)
    offset = (length(sampled_kernel) - 1) ÷ 2

    map(region_inds) do r
        buffered_region = r[1]-offset : r[end]+offset
        flux[buffered_region] = conv(sampled_kernel, flux[buffered_region])[offset+1 : end-offset]
    end
end

"""
This loads a grok model grid.
"""
function load_grid(grid_path; fix_vmic=nothing, fill_non_finite=10.0, fix_off_by_one=false, 
                   v_sinis=nothing, ε=0.6)
    model_spectra = h5read(grid_path, "spectra")

    if !isnothing(fill_non_finite)
        model_spectra[isnan.(model_spectra)] .= fill_non_finite
    end

    # TODO make it possible to not hardcode the specific parameters?
    teffs = h5read(grid_path, "Teff_vals")
    loggs = h5read(grid_path, "logg_vals")
    v_mics = h5read(grid_path, "vmic_vals")
    m_hs = h5read(grid_path, "metallicity_vals")

    if !isnothing(fix_vmic)
        index = findfirst(v_mics .== fix_vmic)
        model_spectra = model_spectra[:, :, :, index, :]
        labels = ["teff", "logg", "m_h"]
        grid_points = [teffs, loggs, m_hs]
    else
        labels = ["teff", "logg", "v_micro", "m_h"]
        grid_points = [teffs, loggs, v_mics, m_hs]
    end

    if fix_off_by_one
        # TODO audit this
        model_spectra[2:end, :, :, :, :] .= model_spectra[1:end-1, :, :, :, :]
    end

    if !isnothing(v_sinis)

        # make it a matrix with each row a spectrum
        model_spectra = reshape(model_spectra, (size(model_spectra, 1), :))

        _model_spectra = Array{Float64}(undef, (size(model_spectra, 1),  
                                                prod(collect(size(model_spectra)[2:end])),
                                                length(v_sinis)))

        # apply each kernel to model spectra to add a new dimention to the grid
        for (i, v_sini) in enumerate(v_sinis)
            for j in 1:size(model_spectra, 2)
                rotF = apply_rotation(model_spectra[:, j], v_sini, ε=ε)
                if all(rotF .== 0)
                    println("all null")
                end
                _model_spectra[:, j, i] = rotF #apply_rotation(model_spectra[:, j], v_sini, ε=ε)
            end
            # apply the kernel to each spectrum
            #_model_spectra[:, :, i] = kernel * model_spectra
        end

        push!(grid_points, v_sinis)
        push!(labels, "v_sini")
        
        model_spectra = reshape(_model_spectra, (size(model_spectra, 1), length.(grid_points)...))
    end
    (labels, grid_points, model_spectra)
end

function _normal_pdf(Δ, σ; cuttoff=3) 
    if abs(Δ) > cuttoff * 3σ
        0.0
    else
        exp(-0.5 * Δ^2 / σ^2)
    end
end

"""
fluxes and ivar should be vectors or lists of the same length whose elements are vectors of length 
8575.
"""
function get_best_nodes(fluxes, ivars, grid)
    fluxes = Vector{Float64}.(collect.(fluxes))
    ivars = Vector{Float64}.(collect.(ivars))
    labels, grid_points, model_spectra = grid 

    # reshape the model spectra to be a 2D array w/ first dimension corresponding to wavelength
    stacked_model_spectra = reshape(model_spectra, (size(model_spectra, 1), :))
    convolved_inv_model_flux = 1 ./ stacked_model_spectra
    @showprogress desc="conv'ing inverse models" for i in 1:size(stacked_model_spectra, 2)
        apply_smoothing_filter!(view(convolved_inv_model_flux, :, i))
    end
    #convolved_model_flux = reshape(convolved_inv_model_flux, size(model_spectra))
    masked_model_spectra = (stacked_model_spectra .* convolved_inv_model_flux)[ferre_mask, :]
    # reshape back
    masked_model_spectra = reshape(masked_model_spectra, (size(masked_model_spectra, 1), size(model_spectra)[2:end]...))

    # iterate over the obbserved spectra
    out = Array{Any}(undef, length(fluxes))
    p = Progress(length(fluxes); dt=1.0, desc="finding best-fit nodes")
    for (i, (flux, ivar)) in enumerate(zip(fluxes, ivars))
        convF = copy(flux)
        fill_chip_gaps!(convF)
        apply_smoothing_filter!(convF)

        # rather than comparing the observed spectrum to every single model in the grid, we 
        # subsample at every 2^n nodes, turning n down from n_refinements to 0.
        # TODO limit this to max refinements
        overlap = 0.5
        n_refinements = 4

        start_indices = ones(Int, length(size(model_spectra)[2:end]))
        end_indices = size(model_spectra)[2:end]

        chi2 = Array{Float64}(undef, size(masked_model_spectra)[2:end])
        rel_index = nothing # index into the sub-array of chi2 values
        index = nothing # index into the (not actually constructed) full array of chi2 values
        slicer = nothing
        # S is the "step size"
        for S in (2 .^ (n_refinements:-1:0))
            slicer = [si:S:ei for (si, ei) in zip(start_indices, end_indices)]
        
            chi2 = sum((view(masked_model_spectra, :, slicer...) .* convF[ferre_mask] .- flux[ferre_mask, :]).^2 .* ivar[ferre_mask], dims=1)
            
            rel_index = collect(Tuple(argmin(chi2)))[2:end]
            index = (start_indices .- 1) + S * rel_index

            offset = Int(ceil((1 + overlap) * S))
            start_indices = max.(index .- offset, 1)
            end_indices = min.(index .+ offset, collect(size(model_spectra)[2:end]))
        end
        best_fit_node = getindex.(grid_points, index)
        splines = map(1:length(best_fit_node)) do index_index
            # the best-fit index for each dimension, but a colon for the one that this spline is over
            slice = [rel_index[1:index_index-1] ; : ; rel_index[index_index+1:end]]

            Korg.CubicSplines.CubicSpline(grid_points[index_index][slicer[index_index]], chi2[1, slice...])
        end

        out[i] = (best_fit_node,
         minimum(chi2),
         model_spectra[:, index...],
         splines
         )

         next!(p)
    end
    finish!(p)
    out
end

#=
   STUFF TO INTERPOLATE SPECTRA
"""
Convert an array to a range if it is possible to do so.
"""
function rangify(a::AbstractRange)
    a
end
function rangify(a::AbstractVector)
    minstep, maxstep = extrema(diff(a))
    @assert minstep ≈ maxstep
    range(a[1], a[end]; length=length(a))
end

# set up a spectrum interpolator
interpolate_spectrum = let 
    T_pivot = 3000 # TODO change this to 4000 when the grid is updated
    coolmask = all_vals[1] .< T_pivot

    # uncomment this to use grids that go below the temperature pivot
    #xs = rangify.((1:sum(global_ferre_mask), all_vals[1][coolmask], all_vals[2:end]...))
    #cool_itp = cubic_spline_interpolation(xs, model_spectra[:, coolmask, :, :, :])

    xs = rangify.((1:sum(global_ferre_mask), all_vals[1][.! coolmask], all_vals[2:end]...))
    hot_itp = cubic_spline_interpolation(xs, model_spectra[:, .!coolmask, :, :]) 
    
    function itp(Teff, logg, metallicity) 
        if Teff < T_pivot
            #cool_itp(1:sum(global_ferre_mask), Teff, logg, vmic, metallicity)
        else
            hot_itp(1:sum(global_ferre_mask), Teff, logg, metallicity)
        end
    end
end
=#

#=
     STUFF FOR LIVE SYNTHESIS
const synth_wls = ferre_wls[1] - 10 : 0.01 : ferre_wls[end] + 10
const LSF = Korg.compute_LSF_matrix(synth_wls, ferre_wls, 22_500)
const linelist = Korg.get_APOGEE_DR17_linelist();
const elements_to_fit = ["Mg", "Na", "Al"] # these are what will be fit
=#


end
