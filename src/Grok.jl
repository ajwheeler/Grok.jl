module Grok
using HDF5, Optim, FITSIO, Interpolations, Korg, ProgressMeter
using DSP: conv  # used for vsini
using SparseArrays: spzeros # used for crazy continuum adjustment
using Distributed: addprocs, pmap

# TODO addprocs

#####################################################################
# TODO these should be excised from the module
#include("element_windows.jl")
# read the mask that is aplied to all spectra
const ferre_mask = parse.(Bool, readlines("ferre_mask.dat"));
const ferre_wls = (10 .^ (4.179 .+ 6e-6 * (0:8574)))
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

function calculate_filter()
    #TODO modify this to act chipwise?
    n_wls = length(ferre_wls)
    oversampled_kernel = _normal_pdf.((1 - n_wls) : n_wls, 51)
    filter = spzeros(n_wls, n_wls)
    @showprogress desc="calculating smoothing filter" for i in 1:n_wls
        k = oversampled_kernel[i+n_wls-1 : -1 : i]
        filter[:, i] .= k ./ sum(k)
    end
    filter
end

"""
fluxes and ivar should be vectors or lists of the same length whose elements are vectors of length 
8575.
"""
function get_best_nodes(fluxes, ivars, filter, grid)
    fluxes = Vector{Float64}.(collect.(fluxes))
    ivars = Vector{Float64}.(collect.(ivars))
    labels, grid_points, model_spectra = grid 

    # reshape the model spectra to be a 2D array w/ first dimension corresponding to wavelength
    stacked_model_spectra = reshape(model_spectra, (size(model_spectra, 1), :))
    convolved_inv_model_flux = similar(stacked_model_spectra)
    @showprogress desc="conv'ing inverse models" for i in 1:size(stacked_model_spectra, 2)
        convolved_inv_model_flux[:, i] = filter * (1 ./ stacked_model_spectra[:, i])
    end
    convolved_model_flux = reshape(convolved_inv_model_flux, size(model_spectra))
    println(size(convolved_model_flux))
    # convolve and reshape back to have a dimension per parameter
    #convolved_inv_model_flux = reshape(filter * (1 ./ stacked_model_spectra), size(model_spectra))

    masked_model_spectra = model_spectra[ferre_mask, (1:s for s in size(model_spectra)[2:end])...]

    @showprogress desc="finding best-fit nodes" pmap(fluxes, ivars) do flux, ivar
        #convolved_flux = filter * flux
        #quote_unquote_continuum = convolved_flux .* convolved_inv_model_flux

        chi2 = sum((masked_model_spectra .- flux[ferre_mask]).^2 .* ivar[ferre_mask], dims=1)
        best_inds = collect(Tuple(argmin(chi2)))[2:end]
        best_fit_node = getindex.(grid_points, best_inds) # the label values of the best-fit node
    end
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
