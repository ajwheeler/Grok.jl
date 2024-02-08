import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy import interpolate
import os
from astropy.io import fits
from typing import Optional, Tuple, Union
from scipy.ndimage import median_filter


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))
    

def load_grid(path, fix_vmic=None, fill_non_finite=10.0, fix_off_by_one=False):
    
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
            labels = ("m_h", "logg", "teff")
            grid_points = (m_hs, loggs, teffs)
            
        else:
            labels = ("m_h", "v_micro", "logg", "teff")
            grid_points = (m_hs, v_mics, loggs, teffs)
            
        if fix_off_by_one:
            model_spectra.T[1:] = model_spectra.T[:-1]
        
        
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
    
    
    labels, grid_points, model_flux = load_grid(
        "~/Downloads/korg_grid_old.h5",
        fix_off_by_one=True,
        fill_non_finite=10.0,
        fix_vmic=2.0
    )    
    
    
    # load the shit
    wl, flux, e_flux, pixel_flags = read_apstar("~/Downloads/apStar-dr17-2M00000068+5710233.fits") 
    #wl, flux, e_flux, pixel_flags = read_apstar("/Users/andycasey/Downloads/apStar-dr17-2M05372530-0633125.fits")
    #wl, flux, e_flux, pixel_flags = read_apstar("/Users/andycasey/Downloads/apStar-dr17-2M09485609+1344395.fits")
    #wl, flux, e_flux, pixel_flags = read_apstar("/Users/andycasey/Downloads/apStar-dr17-2M18061464-2434528.fits")
    #wl, flux, e_flux, pixel_flags = read_apstar("/Users/andycasey/research/20240208_kepler_subset_spectra/apStar-dr17-2M19441934+4716319.fits")
                
    
    
    from tqdm import tqdm    
    import concurrent.futures
    import matplotlib.pyplot as plt


    def analyze_spectrum(wl, flux, e_flux, pixel_flags, n=100):
        # Do the continuum thing that FERRE does.
        continuum = np.zeros_like(model_flux)
        shape = continuum.shape[:-1]
        '''
        futures = []
        for index in range(np.prod(shape)):
            i, j, k = np.unravel_index(index, shape)
            futures.append(pool.submit(do_median_filter, index, wl, flux, model_flux[i, j, k]))
            
        for future in concurrent.futures.as_completed(futures):#, total=1 + index):
            index, s = future.result()
            i, j, k = np.unravel_index(index, shape)
            continuum[i, j, k] = s
        '''
        for index in tqdm(range(np.prod(shape)), total=np.prod(shape)):
            i, j, k = np.unravel_index(index, shape)
            _, s = do_median_filter(index, wl, flux, model_flux[i, j, k])
            continuum[i, j, k] = s
                    
        ivar = e_flux**(-2)
        no_continuum = np.all(continuum <= 0, axis=(0, 1, 2))
        ivar[(~np.isfinite(ivar)) | no_continuum] = 0
        
        chi2 = np.sum((model_flux * continuum - flux)**2 * ivar, axis=-1)

        index = np.argmin(chi2)
        i, j, k = np.unravel_index(index, shape)
        point = [g[i] for g, i in zip(grid_points, np.unravel_index(index, shape))]

        grid_point = dict(zip(labels, point))
        opt_point = dict()
            
        for i, label in enumerate(labels):
            axis = tuple(sorted(set(range(len(grid_points))).difference((i,))))
            x = grid_points[i]
            y = np.min(chi2, axis=axis)
            y /= np.min(y)
            
            tck = interpolate.splrep(x, y)
            xi = np.linspace(np.min(x), np.max(x), n)
            yi = interpolate.splev(xi, tck)

            opt_point[label] = xi[np.argmin(yi)]

        return (chi2, continuum, grid_point, opt_point)
    
    
    from time import time
    t_init = time()
    before_chi2, before_continuum, before_grid_point, before_opt_point = analyze_spectrum(wl, flux, e_flux, pixel_flags)
    t_before = time() - t_init


    '''
    # Do the ratio thing with a gaussian filter

    # let's apply the bad pixel stuff to the data first so we don't have to do it per RATIO
    flux_copy = np.copy(flux)
    for si, ei in slices:
        region = flux_copy[si:ei]
        
        is_bad_pixel = (
            (region < 0.01)
        |   (region > (np.nanmedian(region) + 3 * np.nanstd(region)))
        |   (~np.isfinite(region))
        |   (region < 0.1)
        |   (region > 1e8)
        )
        if np.any(is_bad_pixel):
            x = np.arange(region.size)
            flux_copy[si:ei][is_bad_pixel] = np.interp(x[is_bad_pixel], x[~is_bad_pixel], region[~is_bad_pixel])
    
    continuum = np.zeros_like(model_flux)
    for si, ei in slices:
        continuum[:, :, :, si:ei] = gaussian_filter1d((flux_copy[si:ei]/model_flux[:, :, :, si:ei]), 51, mode="mirror", axis=-1)
    
    
    ivar = e_flux**(-2)
    no_continuum = np.all(continuum <= 0, axis=(0, 1, 2))
    ivar[(~np.isfinite(ivar)) | no_continuum] = 0
    
    chi2 = np.sum((model_flux * continuum - flux)**2 * ivar, axis=-1)
    '''
    
    
    regions = [
        (15152, 15800),
        (15867, 16424),
        (16484, 16944),        
    ]
    slices = [wl.searchsorted(region) + [0, 1] for region in regions]
    
    sigma, mode = (51, "mirror")
            
    # Convolve the grid once and convolve data once
    convolved_inv_model_flux = np.zeros_like(model_flux)
    for si, ei in slices:
        convolved_inv_model_flux[:, :, :, si:ei] = gaussian_filter1d(1/model_flux[:, :, :, si:ei], sigma, mode=mode, axis=-1)            
    no_continuum = np.all(convolved_inv_model_flux == 0, axis=(0, 1, 2))        
        
    '''
        
    
    def analyze_spectrum(wl, flux, e_flux, pixel_flags, n=100):
        # Do the continuum thing that FERRE does.
        continuum = np.zeros_like(model_flux)
        shape = continuum.shape[:-1]
        futures = []
        for index in range(np.prod(shape)):
            i, j, k = np.unravel_index(index, shape)
            futures.append(pool.submit(do_median_filter, index, wl, flux, model_flux[i, j, k]))
            
        for future in concurrent.futures.as_completed(futures):#, total=1 + index):
            index, s = future.result()
            i, j, k = np.unravel_index(index, shape)
            continuum[i, j, k] = s
        
        ivar = e_flux**(-2)
        no_continuum = np.all(continuum <= 0, axis=(0, 1, 2))
        ivar[(~np.isfinite(ivar)) | no_continuum] = 0
        
        chi2 = np.sum((model_flux * continuum - flux)**2 * ivar, axis=-1)

        index = np.argmin(chi2)
        i, j, k = np.unravel_index(index, shape)
        point = [g[i] for g, i in zip(grid_points, np.unravel_index(index, shape))]

        grid_point = dict(zip(labels, point))
        opt_point = dict()
            
        for i, label in enumerate(labels):
            axis = tuple(sorted(set(range(len(grid_points))).difference((i,))))
            x = grid_points[i]
            y = np.min(chi2, axis=axis)
            y /= np.min(y)
            
            tck = interpolate.splrep(x, y)
            xi = np.linspace(np.min(x), np.max(x), n)
            yi = interpolate.splev(xi, tck)

            opt_point[label] = xi[np.argmin(yi)]

        return (chi2, continuum, grid_point, opt_point)   
    '''
    
    def fast_analyze_spectrum(flux, e_flux, pixel_flags, N=1000):
        
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
        i, j, k = np.unravel_index(index, shape)
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
    
        return (chi2, continuum, grid_point, opt_point)  
    
    from time import time
    t_init = time()
    after_chi2, after_continuum, after_grid_point, after_opt_point = fast_analyze_spectrum(flux, e_flux, pixel_flags)
    t_after = time() - t_init

        
    # options are:
    # 1. do the ratio thing with a gaussian filter
    # 2. convolve the grid once and convolve the data once
    
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
            

    from glob import glob

    paths = glob("/Users/andycasey/research/20240208_kepler_subset_spectra/apStar-*.fits")

    results = []
    for path in tqdm(paths):
        wl, flux, e_flux, pixel_flags = read_apstar(path)
        chi2, continuum, grid_point, opt_point = fast_analyze_spectrum(flux, e_flux, pixel_flags)
        apogee_id = os.path.basename(path).split("-dr17-")[1].split(".fits")[0]        
        results.append((apogee_id, opt_point["teff"], opt_point["logg"], opt_point["m_h"], np.min(chi2)))


    from astropy.table import Table
    t = Table(rows=results, names=("apogee_id", "grok_teff", "grok_logg", "grok_m_h", "chi2"))
    t.write("~/research/20240208_kepler_subset_spectra/fast_grok_results.fits")    
        
