import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy import interpolate
import os
from astropy.io import fits
from typing import Optional, Tuple, Union
from scipy.ndimage import median_filter
from tqdm import tqdm


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))

import numpy as np
from scipy import sparse, stats
from typing import Sequence, Optional

_fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))


λ = 10**(4.179 + 6e-6 * np.arange(8575))


def rotational_broadening_kernel(λ: Sequence[float], vsini: float, epsilon: float):
    """
    Construct a sparse matrix to convolve fluxes at input wavelengths (λ) with a rotational broadening kernel
    with a given vsini and epsilon.

    :param λ:
        A N-length array of input wavelength values.

    :param vsini:
        The projected rotational velocity of the star in km/s.

    :param epsilon:
        The limb darkening coefficient.

    :returns:
        A (N, N) sparse array representing a convolution kernel.
    """

    # Let's pre-calculate some things that are needed in the hot loop.
    denominator = np.pi * vsini * (1.0 - epsilon / 3.0)
    c1 = 2.0 * (1.0 - epsilon) / denominator
    c2 = 0.5 * np.pi * epsilon / denominator

    vsini_c = vsini / 299792.458
    scale = vsini_c / (λ[1] - λ[0])  # assume uniform sampling
    N = λ.size

    data, row_index, col_index = ([], [], [])
    for i, λ_i in enumerate(λ):
        n_pix = int(np.ceil(λ_i * scale))
        si, ei = (max(0, i - n_pix), min(i + n_pix + 1, N))  # ignoring edge effects

        λ_delta_max = λ_i * vsini_c
        λ_delta = λ[si:ei] - λ_i
        λ_ratio_sq = (λ_delta / λ_delta_max) ** 2.0
        ϕ = c1 * np.sqrt(1.0 - λ_ratio_sq) + c2 * (1.0 - λ_ratio_sq)
        ϕ[λ_ratio_sq >= 1.0] = 0.0  # flew too close to the sun
        ϕ /= np.sum(ϕ)

        data.extend(ϕ)
        row_index.extend(list(range(si, ei)))
        col_index.extend([i] * (ei - si))

    return sparse.csr_matrix((data, (row_index, col_index)), shape=(λ.size, λ.size))
    

def load_grid(
    path, 
    fix_vmic=None, 
    fill_non_finite=10.0, 
    fix_off_by_one=False,
    v_sinis=None,
    epsilon=0.6,
):
    
    with h5.File(expand_path(path), "r") as grid:
        
        model_spectra = grid["spectra"][:]
        if fill_non_finite is not None:
            model_spectra[~np.isfinite(model_spectra)] = 10.0

        teffs = grid["Teff_vals"][:]
        loggs = grid["logg_vals"][:]
        m_hs = grid["metallicity_vals"][:]
        v_mics = grid["vmic_vals"][:]
        
        if fix_vmic is not None:
            index = list(v_mics).index(fix_vmic)
            model_spectra = model_spectra[:, index]
            labels = ["m_h", "logg", "teff"]
            grid_points = [m_hs, loggs, teffs]
            
        else:
            labels = ["m_h", "v_micro", "logg", "teff"]
            grid_points = [m_hs, v_mics, loggs, teffs]
            
        if fix_off_by_one:
            model_spectra.T[1:] = model_spectra.T[:-1]
            
        if v_sinis is not None:
            λ = 10**(4.179 + 6e-6 * np.arange(8575))

            v_sinis = np.hstack([v_sinis]).flatten()
            
            kernels = []
            for v_sini in tqdm(v_sinis, desc="Constructing kernels"):
                if v_sini == 0:
                    kernels.append(None)
                else:                    
                    kernels.append(
                        rotational_broadening_kernel(
                            λ,
                            v_sini,
                            epsilon=epsilon
                        )
                    )
            
            # Convolve model_spectra with K
            N = np.prod(model_spectra.shape[:-1])
            _model_spectra = np.empty((v_sinis.size, *model_spectra.shape))
            
            with tqdm(desc="Applying vsini convolution", total=N * v_sinis.size) as pb:                
                for i, (v_sini, kernel) in enumerate(zip(v_sinis), kernels)):
                    if kernel is None:
                        _model_spectra[i, :] = model_spectra
                        pb.update(N)
                    else:
                        for j in range(N):
                            jdx = np.unravel_index(j, model_spectra.shape[:-1])
                            idx = tuple([i] + list(jdx))
                            _model_spectra[idx] = kernel @ model_spectra[jdx]
                            
                            assert np.all(_model_spectra[idx]) > 0
                            pb.update()
                        
            labels.insert(0, "v_sini")
            grid_points.insert(0, v_sinis)
            model_spectra = _model_spectra
        
        
    return (labels, grid_points, model_spectra)



def inflate_errors_at_bad_pixels(
    flux,
    e_flux,
    bitfield,
    skyline_sigma_multiplier=100,
    bad_pixel_flux_value=1e-4,
    bad_pixel_error_value=1e10,
    spike_threshold_to_inflate_uncertainty=3,
    min_sigma_value=0.05,
):
    # Inflate errors around skylines,
    skyline_mask = (bitfield & 4096) > 0 # significant skyline
    e_flux[skyline_mask] *= skyline_sigma_multiplier

    # Sometimes FERRE will run forever.
    if spike_threshold_to_inflate_uncertainty > 0:

        flux_median = np.nanmedian(flux)
        flux_stddev = np.nanstd(flux)
        e_flux_median = np.median(e_flux)

        delta = (flux - flux_median) / flux_stddev
        is_spike = (delta > spike_threshold_to_inflate_uncertainty)
        #* (
        #    sigma_ < (parameters["spike_threshold_to_inflate_uncertainty"] * e_flux_median)
        #)
        #if np.any(is_spike):
        #    sum_spike = np.sum(is_spike)
            #fraction = sum_spike / is_spike.size
            #log.warning(
            #    f"Inflating uncertainties for {sum_spike} pixels ({100 * fraction:.2f}%) that were identified as spikes."
            #)
            #for pi in range(is_spike.shape[0]):
            #    n = np.sum(is_spike[pi])
            #    if n > 0:
            #        log.debug(f"  {n} pixels on spectrum index {pi}")
        e_flux[is_spike] = bad_pixel_error_value

    # Set bad pixels to have no useful data.
    if bad_pixel_flux_value is not None or bad_pixel_error_value is not None:                            
        bad = (
            ~np.isfinite(flux)
            | ~np.isfinite(e_flux)
            | (flux < 0)
            | (e_flux < 0)
            | ((bitfield & 16639) > 0) # any bad value (level = 1)
        )

        flux[bad] = bad_pixel_flux_value
        e_flux[bad] = bad_pixel_error_value        

    if min_sigma_value is not None:
        e_flux = np.clip(e_flux, min_sigma_value, np.inf)

    return (flux, e_flux)







def read_apstar(path, inflate_errors=True, use_ferre_mask=True):
    with fits.open(expand_path(path)) as image:
        flux = image[1].data[0]
        e_flux = image[2].data[0]
        pixel_flags = image[3].data[0]
        
    if inflate_errors:
        flux, e_flux = inflate_errors_at_bad_pixels(
            flux, 
            e_flux,
            pixel_flags,
        )
    
    if use_ferre_mask:
        ferre_mask = np.loadtxt(expand_path("~/Downloads/ferre_mask.dat"))
        use_pixel = (ferre_mask == 1)        
        e_flux[~use_pixel] = np.inf

    wl = 10**(4.179 + 6e-6 * np.arange(8575))
    
    return (wl, flux, e_flux, pixel_flags)


from scipy.ndimage import gaussian_filter1d

def do_median_filter(
    index,
    wl,
    flux,
    model_flux,
    median_filter_width = 151,
    bad_minimum_flux = 0.01,
    valid_continuum_correction_range: Optional[Tuple[float]] = (0.1, 1e8),#, +np.inf),#0.1, 10.0),
    mode: Optional[str] = "mirror",
    regions: Optional[Tuple[Tuple[float, float]]] = (
        (15152, 15800),
        (15867, 16424),
        (16484, 16944),
    ),
    fill_value: Optional[Union[int, float]] = 0,
    func="median"
):
    slices = []
    for lower, upper in regions:
        si, ei = wl.searchsorted([lower, upper])
        slices.append((si, 1 + ei))

    continuum = fill_value * np.ones_like(wl)
    for si, ei in slices:
        
        flux_region = flux[si:ei].copy()
        ratio = flux_region / model_flux[si:ei]

        is_bad_pixel = (
            (flux_region < bad_minimum_flux)
        |   (flux_region > (np.nanmedian(flux_region) + 3 * np.nanstd(flux_region)))
        |   (~np.isfinite(ratio))
        |   (ratio < valid_continuum_correction_range[0])
        |   (ratio > valid_continuum_correction_range[1])
        )        
        if not np.all(is_bad_pixel):
            x = np.arange(ratio.size)
            ratio[is_bad_pixel] = np.interp(x[is_bad_pixel], x[~is_bad_pixel], ratio[~is_bad_pixel])            
            if func == "median":            
                continuum[si:ei] = median_filter(ratio, size=median_filter_width, mode=mode)
            else:
                continuum[si:ei] = gaussian_filter1d(ratio, median_filter_width, mode=mode)

    return (index, continuum)


if __name__ == "__main__":
    
    try:
        model_flux
    except NameError:
        labels, grid_points, model_flux = load_grid(
            "~/Downloads/korg_grid_old.h5",
            fix_off_by_one=True,
            fill_non_finite=10.0,
            fix_vmic=1.0,
            v_sinis=[0, 5, 10, 20, 30, 50]
        )    
    
        regions = [
            (15152, 15800),
            (15867, 16424),
            (16484, 16944),        
        ]
        λ = 10**(4.179 + 6e-6 * np.arange(8575))

        slices = [λ.searchsorted(region) + [0, 1] for region in regions]
        
        sigma, mode = (51, "mirror")
                
        # Convolve the grid once and convolve data once
        convolved_inv_model_flux = np.zeros_like(model_flux)
        for si, ei in slices:
            # todo: generalize this slicing no matter how many grid dimensions we have
            if model_flux.ndim == 4:
                convolved_inv_model_flux[:, :, :, si:ei] = gaussian_filter1d(1/model_flux[:, :, :, si:ei], sigma, mode=mode, axis=-1)
            elif model_flux.ndim == 5:
                convolved_inv_model_flux[:, :, :, :, si:ei] = gaussian_filter1d(1/model_flux[:, :, :, :, si:ei], sigma, mode=mode, axis=-1)            
            elif model_flux.ndim == 6:
                convolved_inv_model_flux[:, :, :, :, :, si:ei] = gaussian_filter1d(1/model_flux[:, :, :, :, :, si:ei], sigma, mode=mode, axis=-1)            
            else:
                raise a
        
        no_continuum = np.all(convolved_inv_model_flux == 0, axis=tuple(range(convolved_inv_model_flux.ndim - 1)))
        
    else:
        print("Warning, using pre-loaded grid. `del model_flux` to reload")
        
    
    # load the shit
    wl, flux, e_flux, pixel_flags = read_apstar("~/Downloads/apStar-dr17-2M00000068+5710233.fits") 
    #wl, flux, e_flux, pixel_flags = read_apstar("/Users/andycasey/Downloads/apStar-dr17-2M05372530-0633125.fits")
    #wl, flux, e_flux, pixel_flags = read_apstar("/Users/andycasey/Downloads/apStar-dr17-2M09485609+1344395.fits")
    #wl, flux, e_flux, pixel_flags = read_apstar("/Users/andycasey/Downloads/apStar-dr17-2M18061464-2434528.fits")
    #wl, flux, e_flux, pixel_flags = read_apstar("/Users/andycasey/research/20240208_kepler_subset_spectra/apStar-dr17-2M19441934+4716319.fits")
    
    
    def fast_analyze_spectrum(flux, e_flux, N=1000, full_output=False):
        
        convolved_flux = np.copy(flux)
        for si, ei in slices:
            region = convolved_flux[si:ei]
            
            is_bad_pixel = (
                (region < 0.01)
            |   (region > (np.nanmedian(region) + 3 * np.nanstd(region)))
            |   (~np.isfinite(region))
            |   (region < 0.1)
            |   (region > 1e8)
            )
            if np.any(is_bad_pixel):
                x = np.arange(region.size)
                region[is_bad_pixel] = np.interp(x[is_bad_pixel], x[~is_bad_pixel], region[~is_bad_pixel])
                
            convolved_flux[si:ei] = gaussian_filter1d(region, sigma, mode=mode)
            
        # now on a per-star basis    
        continuum = convolved_flux * convolved_inv_model_flux
        
        ivar = e_flux**(-2)
        ivar[(~np.isfinite(ivar)) | no_continuum] = 0
        
        chi2 = np.sum((model_flux * continuum - flux)**2 * ivar, axis=-1)
        
        shape = continuum.shape[:-1]

        index = np.argmin(chi2)
        #i, j, k = np.unravel_index(index, shape)
        point = [g[i] for g, i in zip(grid_points, np.unravel_index(index, shape))]

        grid_point = dict(zip(labels, point))
        opt_point = dict()
            
        for i, label in enumerate(labels):
            axis = tuple(sorted(set(range(len(grid_points))).difference((i,))))
            x = grid_points[i]
            y = np.min(chi2, axis=axis)
            y /= np.min(y)
            
            tck = interpolate.splrep(x, y)
            xi = np.linspace(np.min(x), np.max(x), N)
            yi = interpolate.splev(xi, tck)

            opt_point[label] = xi[np.argmin(yi)]
    
        if full_output:
            return (chi2, continuum, grid_point, opt_point)  
        return (opt_point, np.min(chi2))
    
    def _fast_analyze_spectrum(index, flux, e_flux):
        args = fast_analyze_spectrum(flux, e_flux)
        return (index, *args)
    
    print("about to start analysis")
    from time import time
    t_init = time()
    after_chi2, after_continuum, after_grid_point, after_opt_point = fast_analyze_spectrum(flux, e_flux, full_output=True)
    t_after = time() - t_init
        
    N = 1000
    
    fig, axes = plt.subplots(1, len(grid_points), figsize=(9, 3))
    for i, ax in enumerate(axes.flat):
        axis = tuple(sorted(set(range(len(grid_points))).difference((i,))))
        x = grid_points[i]
        y = np.min(after_chi2, axis=axis)
        y /= np.min(y)
        
        tck = interpolate.splrep(x, y)
        xi = np.linspace(np.min(x), np.max(x), N)
        yi = interpolate.splev(xi, tck)
        
        ax.plot(x, y)
        ax.plot(xi, yi)
        ax.axvline(xi[np.argmin(yi)], c="tab:red")
        ax.set_xlabel(labels[i])
        print(labels[i], xi[np.argmin(yi)])
        #ax.set_ylim(0.9, 1.5)
        

    fig.tight_layout()
    

    # Let's do the Tayar comparison
    import concurrent.futures
    
    from astropy.table import Table
    t = Table.read("/Users/andycasey/research/Grok.jl/sandbox/tayar_2015/apj514696t1_mrt_xm_aspcap.fits")    
    is_measurement = (t["f_vsini"] != "<")
    
    paths = []
    for twomass_id in t[is_measurement]["2MASS"]:
        apogee_id = twomass_id.lstrip("J")
        paths.append(f"/Users/andycasey/research/Grok.jl/sandbox/tayar_2015/spectra/apStar-dr17-2M{apogee_id}.fits")
    
    CHECKPOINT_FREQUENCY = 10
    names=("apogee_id", "grok_teff", "grok_logg", "grok_m_h", "grok_v_micro", "grok_v_sini", "chi2")
    output_path = "/Users/andycasey/research/Grok.jl/sandbox/tayar_2015/20240209_grok_results.fits"

    
    pool = concurrent.futures.ThreadPoolExecutor(8)
    
    futures = []
    for path in paths:
        wl, flux, e_flux, pixel_flags = read_apstar(path)
        futures.append(pool.submit(_fast_analyze_spectrum, path, flux, e_flux))
        
    results = []
    for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures)):
        path, opt_point, min_chi2 = future.result()
        apogee_id = os.path.basename(path).split("-dr17-")[1].split(".fits")[0]        
        results.append((apogee_id, opt_point["teff"], opt_point["logg"], opt_point["m_h"], opt_point.get("v_micro", np.nan), opt_point.get("v_sini", np.nan), min_chi2))            
        if (i % CHECKPOINT_FREQUENCY) == 0:
            Table(rows=results, names=names).write(output_path, overwrite=True)
                
    Table(rows=results, names=names).write(output_path, overwrite=True)
    raise a
        
    
    
    

            

    from astropy.table import Table
    
    
    from glob import glob

    paths = glob("/Users/andycasey/research/20240208_kepler_subset_spectra/apStar-*.fits")
    PARALLEL = True
    CHECKPOINT_FREQUENCY = 10
    names=("apogee_id", "grok_teff", "grok_logg", "grok_m_h", "grok_v_micro", "grok_v_sini", "chi2")
    output_path = expand_path("~/research/20240208_kepler_subset_spectra/20240209_grok_results.fits")

    
    
    if PARALLEL:
        
        pool = concurrent.futures.ThreadPoolExecutor(8)
        
        futures = []
        for path in paths:
            wl, flux, e_flux, pixel_flags = read_apstar(path)
            futures.append(pool.submit(_fast_analyze_spectrum, path, flux, e_flux))
            
        results = []
        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures), total=len(futures))):
            path, opt_point, min_chi2 = future.result()
            apogee_id = os.path.basename(path).split("-dr17-")[1].split(".fits")[0]        
            results.append((apogee_id, opt_point["teff"], opt_point["logg"], opt_point["m_h"], opt_point.get("v_micro", np.nan), opt_point.get("v_sini", np.nan), min_chi2))            
            if (i % CHECKPOINT_FREQUENCY) == 0:
                Table(rows=results, names=names).write(output_path, overwrite=True)
                

    else:
                    
        results = []
        for i, path in enumerate(tqdm(paths)):
            wl, flux, e_flux, pixel_flags = read_apstar(path)
            opt_point, min_chi2 = fast_analyze_spectrum(flux, e_flux)
            apogee_id = os.path.basename(path).split("-dr17-")[1].split(".fits")[0]        
            results.append((apogee_id, opt_point["teff"], opt_point["logg"], opt_point["m_h"], opt_point.get("v_micro", np.nan), opt_point.get("v_sini", np.nan), min_chi2))
            if (i % CHECKPOINT_FREQUENCY) == 0:
                Table(rows=results, names=names).write(output_path, overwrite=True)
    
    t = Table(rows=results, names=names)
    t.write(output_path, overwrite=True)
    
