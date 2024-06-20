"""
====================================================

This script processes FTIR spectra of plastic debris and can generate the following figures:
    1) plots of all spectra attributed to each sample
    2) plots of all spectra assigned to each material
    3) plots highlighting automatically detected peaks in each spectrum
    4) plots of automatically fitted peak shapes
    5) Principal Component Analysis results for imported spectra
    6) K-means clustering of spectral PCA coordinates
    
This script is designed to accept files with the following naming convention:
    MeasurementDate(YYYY-MM-DD)_SampleID_SpectrumNumber_OptionalNotes.CSV
This can be amended by changing lines 660-665

====================================================
"""

# ==================================================
# import necessary Python packages

import os
import math
import glob
import datetime
import numpy as np
import pandas as pd
import lmfit as lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage.color as Color

from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ==================================================
# user defined variables

# directory of folder containing spectra files to be imported and processed
Data_dir = './data/'

# directory of sample info database containing list of sample names and their material assignments
Database_dir = './data/Database.csv'

# directory of folder where figures will be saved
Fig_dir = './figures/'

# directory of folder where processed data will be saved
Out_dir = './output/'

# specify whether to process spectra in 'Absorbance' or 'Transmittance' mode
Mode = 'Absorbance'

# specify which figures to generate
Plot_sample_summary = True      # produce a figure for each unique sample
Plot_material_summary = True    # produce a figure for each unique material

# specify which processes to run
Detect_Peaks = True            # search for maxima in data
if Detect_Peaks == True:
    Fit_peaks = True           # fit maxima with mathemetical functions to get parameters
    if Fit_peaks == True:
        Fit_function = 'l'      # fitting function ('g' for gaussian, 'l' for lorentzian, 'pv' for pseudo-voigt)

Do_PCA = True                  # run Principal Component Analysis on spectra
if Do_PCA == True:
    Do_clustering = True       # run K-means clustering on PCA coordinates

# list expected materials and their plotting colours
Materials = ['not plastic', 'pumice', 'unassigned', 'PE', 'LDPE', 'MDPE', 'HDPE', 'PP', 'PPE', 'PS', 'PLA', 'ABS', 'EAA', 'PA']
Material_colors =  ['k', 'skyblue', 'chartreuse', 'r', 'g', 'gold', 'b', 'm', 'y', 'tab:gray', 'tab:orange', 'tab:brown', 'tab:pink', 'seagreen']

# generic lists of colours and markers for plotting
Color_list =  ['r', 'b', 'm', 'y', 'tab:gray', 'c', 'tab:orange', 'tab:brown', 'tab:pink', 'tan']
Marker_list = ['o', 's', 'v', '^', 'D', '*', 'p']

# expected peak positions by material (in cm-1)
Peak_positions = {
    'PE': [720, 730, 1460, 1470,  2845, 2880],
    'PP': [810, 840, 1380, 1460, 2845, 2870, 2920, 2950],
    'PPE': [720, 730, 810, 840, 1380, 1460, 2845, 2880, 2920, 2950],
    'PS': [700, 750, 2850, 2920, 3060]
}

"""
# ==================================================
# import sample info database
# ==================================================
"""

print()
print()
print("importing sample database...")

# import database and remove NaNs
sample_database = pd.read_csv(Database_dir, header=0, true_values="Y", false_values="N")
sample_database.fillna(value="", inplace=True)

print()
print(sample_database.info())

Sample_IDs = np.unique(sample_database['ID'])

"""
# ==================================================
# define functions for handling databases
# ==================================================
"""

def get_chemicalID(dataframe, sample_ID, debug=False):
    # get sample material assignment from dataframe
    yesno = dataframe['ID'] == sample_ID
    sliced_dataframe = dataframe[yesno]
    if debug == True:
        print("sliced rows:", len(sliced_dataframe))
    assignment = ""
    if len(sliced_dataframe) > 0:
        temp = sliced_dataframe['Assignment'].iloc[0]
        if debug == True:
            print("material assignment:", temp, type(temp))
        if temp != np.nan:
            assignment = temp
    return assignment

"""
# ==================================================
# define functions for handling FTIR spectra
# ==================================================
"""

# ==================================================
# functions for getting sample data from datadict

def get_sample_spectra(datadict, sample, mode=Mode, debug=False):
    if debug == True:
        print(datadict.keys())
    sort = np.ravel(np.where(datadict['sample'] == sample))
    if debug == True:
        print("sort:", len(sort), sort)
    x = datadict['frequency'][sort]
    if mode.lower() in ['a', 'absorb', 'absorbance']:
        y = datadict['absorbance'][sort]
    else:
        y = datadict['transmittance'][sort]
    return x, y

def get_sample_mean_spectrum(datadict, sample, mode=Mode, debug=False):
    if debug == True:
        print(datadict.keys())
    sort = np.ravel(np.where(datadict['sample'] == sample))
    if debug == True:
        print("sort:", len(sort), sort)
    x = datadict['frequency'][sort]
    if mode.lower() in ['a', 'absorb', 'absorbance']:
        y = datadict['absorbance'][sort]
    else:
        y = datadict['transmittance'][sort]
    if debug == True:
        print("sorted arrays:", np.shape(x), np.shape(y))
    if len(sort) > 1:
        # interpolate data to smallest common range, 0.5 cm-1 resolution
        x_min = 1000
        x_max = 3000
        for i in range(0, len(sort)):
            if np.amin(x[i]) < x_min:
                x_min = np.amin(x[i])
            if np.amax(x[i]) > x_max:
                x_max = np.amax(x[i])
        x_temp = np.linspace(x_min, x_max, 2*int(x_max-x_min))
        y_temp = np.zeros((len(sort), len(x_temp)))
        for i in range(0, len(sort)):
            y_temp[i] = np.interp(x_temp, x[i], y[i])
        if debug == True:
            print("mean of %s spectra:" % len(sort), np.shape(x_temp), np.shape(y_temp))
        return x_temp, np.mean(y_temp, axis=0)
    else:
        if debug == True:
            print("single spectrum:", np.shape(np.mean(x, axis=0)), np.shape(np.mean(y, axis=0)))
        return np.mean(x, axis=0), np.mean(y, axis=0)
    
def count_spectra(datadict, sample, mode=Mode):
    count = 0
    sort = np.where(datadict['ID'] == sample)
    return len(sort)

# ==================================================
# functions for converting Y axis values

def transmittance_to_absorbance(T):
    if np.any(T > 1.):
        A = 2 - np.log10(T)
    else:
        A = -np.log10(T)
    return A

def absorbance_to_transmittance(A, percentage=True):
    if percentage == True:
        T = 10**(2-A)
    else:
        T = 10**(-A)
    return T

def intensity2snr(intensity, noise):
    return intensity / noise

def snr2intensity(snr, noise):
    return snr * noise

# ==================================================
# functions for fitting and subtracting a linear/polynomial baseline from spectrum
    
def average_list(x, y, point_list, window, debug=False):
    # function for taking a set of user-defined points and creating arrays of their average x and y values
    if debug == True:
        print("        ", point_list)
    x_averages = np.zeros_like(point_list, dtype=float)
    y_averages = np.zeros_like(point_list, dtype=float)
    point_num = 0
    for i in range(np.size(point_list)):
        point_num += 1
        x_averages[i], y_averages[i] = local_average(x, y, point_list[i], window)
        if debug == True:
            print("        point", str(point_num), ": ", x_averages[i], y_averages[i])
    return x_averages, y_averages

def local_average(x, y, x_0, w):
    # function for finding the average position from a set of points, centered on 'x_0' with 'w' points either side
    center_ind = np.argmin(np.absolute(x - x_0))
    start_ind = center_ind - w
    end_ind = center_ind + w
    if start_ind < 0:
        start_ind = 0
    if end_ind > len(y)-1:
        end_ind = len(y)-1
    x_temp = x[start_ind:end_ind]
    y_temp = y[start_ind:end_ind]
    x_average = (np.average(x_temp))
    y_average = (np.average(y_temp))
    return x_average, y_average
    
def f_linbase(x, *params):
    # function for generating a linear baseline
    a, b = params
    y = a*x + b
    return y

def f_polybase(x, *params):
    # function for generating an exponential baseline
    y = params[0]
    for i in range(1, len(params)):
        y += params[i] * x**i
    return y
    
def linbase_fit(x_averages, y_averages, debug=False):
    # function for fitting selected average data-points using a linear function
    guess = [1., 0.]
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_linbase, x_averages, y_averages, p0=guess)
    if debug == True:
        print("        fitted parameters: ", fit_coeffs)
    return fit_coeffs, fit_covar

def polybase_fit(x_averages, y_averages, max_order=15, debug=False):
    # function for fitting selected average data-points using a polynominal function
    if len(x_averages) > int(max_order):
        guess = np.zeros((int(max_order)))
    else:
        guess = np.zeros_like(x_averages)
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_polybase, x_averages, y_averages, p0=guess)
    if debug == True:
        print("        fitted parameters:", fit_coeffs)
    return fit_coeffs, fit_covar

def baseline(x, y, point_list, base='poly', max_order=11, window=5, debug=False, plot=False):
    # function for fitting and subtracting a baseline from a spectrum
    x_averages, y_averages = average_list(x, y, sorted(point_list), window)
    if base in ['lin', 'linear']:
        fit_coeffs, fit_covar = linbase_fit(x_averages, y_averages, debug=debug)
        basefit = f_linbase(x, *fit_coeffs)
    else:
        if max_order > len(y_averages)-1:
            max_order = len(y_averages)-1
        fit_coeffs, fit_covar = polybase_fit(x_averages, y_averages, max_order=max_order, debug=debug)
        basefit = f_polybase(x, *fit_coeffs)
    y_sub = y - basefit
    if plot == True:
        plt.plot(x, y, 'k', label='orig.')
        plt.plot(x_averages, y_averages, 'ro', label='points')
        plt.plot(x, basefit, 'r', label='baseline')
        plt.plot(x, y_sub, 'b', label='sub')
        plt.xlim(np.amin(point_list)-50, np.amax(point_list)+50)
        plt.legend()
        plt.show()
    return y_sub

# ==================================================
# functions for finding minima/maxima in a spectrum

def find_min(x, y, x_start, x_end):
    x_slice = x[np.where((x_start <= x) & (x <= x_end))]
    y_slice = y[np.where((x_start <= x) & (x <= x_end))]
    i = np.argmin(y_slice)
    return x_slice[i], y_slice[i]
    
def find_max(x, y, x_start, x_end):
    x_slice = x[np.where((x_start <= x) & (x <= x_end))]
    y_slice = y[np.where((x_start <= x) & (x <= x_end))]
    i = np.argmax(y_slice)
    return x_slice[i], y_slice[i]

def find_peaks(x, y, window_length, threshold=0.1, mode=Mode, debug=False):
    # function for finding the peaks of input data. Each maximum will have the largest value within its window.
    if debug == True:
        print("input:", np.shape(x), np.shape(y))
    if mode.lower() in ['a', 'abs', 'absorb', 'absorbance']:
        index_list = argrelextrema(y, np.greater, order=int(window_length))     # determines indices of all maxima in absorbance data
        y_limit = threshold * np.amax(y)                                        # set the minimum threshold for absorbance 'peaks'
    else:
        index_list = argrelextrema(y, np.less, order=int(window_length))        # determines indices of all minima in transmittance data
        if np.amax(y) > 1.:
            y_limit = 100. - threshold * np.amin(y)                     # set the maximum threshold for transmittance 'peaks'
        else:
            y_limit = 1. - threshold * np.amin(y)                       # set the maximum threshold for transmittance 'peaks'
    all_peaks = np.asarray([x[index_list], y[index_list]])              # creates an array of x and y values for all maxima
    if debug == True:
        print(index_list)
        print(all_peaks)
    if debug == True:
        print(np.any(y == np.inf))
        print("y max:", np.amax(y))
        print("threshold: %0.2f" % threshold)
        print("y limit: %0.2f" % y_limit)
    if mode.lower() in ['a', 'abs', 'absorb', 'absorbance']:
        peaks_x = all_peaks[0, all_peaks[1] >= y_limit]     # records the x values for all valid peaks
        peaks_y = all_peaks[1, all_peaks[1] >= y_limit]     # records the y values for all valid peaks
    else:
        peaks_x = all_peaks[0, all_peaks[1] <= y_limit]     # records the x values for all valid peaks
        peaks_y = all_peaks[1, all_peaks[1] <= y_limit]     # records the y values for all valid peaks
    peaks = np.asarray([peaks_x, peaks_y])                  # creates an array for all valid peaks
    if debug == True:
        print(all_peaks[1, all_peaks[1] >= y_limit])
    return peaks

# ==================================================
# functions for fitting peak shapes from a list of known maxima

def G_curve(x, params):
    model = np.zeros_like(x)
    gradient = params['gradient']
    intercept = params['intercept']
    A = params['amplitude']
    mu = params['center']
    sigma = params['sigma']
    model += A * np.exp(-0.5*(x - mu)**2/(sigma**2)) + gradient*x + intercept
    return model

def multiPV_curve(x, params, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        eta = params['eta_%s' % i]
        model += A * (eta * (sigma**2)/((x - mu)**2 + sigma**2) + (1.-eta) * np.exp(-0.5*(x - mu)**2/(sigma**2)))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiPV_fit(params, x, y, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        eta = params['eta_%s' % i]
        model += A * (eta * (sigma**2)/((x - mu)**2 + sigma**2) + (1.-eta) * np.exp(-0.5*(x - mu)**2/(sigma**2)))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def multiL_curve(x, params, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['gamma_%s' % i]
        model += A * (sigma**2)/((x - mu)**2 + sigma**2)
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiL_fit(params, x, y, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['gamma_%s' % i]
        model += A * (sigma**2)/((x - mu)**2 + sigma**2)
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def multiG_curve(x, params, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiG_fit(params, x, y, maxima):
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def peak_fit_script(x, y, maxima, window=10., max_sigma=30., function='gaussian', vary_baseline=True, debug=False):
    # script for fitting a set of maxima with a pre-defined function (defaults to gaussian)
    params = lmfit.Parameters()
    params.add('gradient', value=0., vary=vary_baseline)
    params.add('intercept', value=np.amin(y), vary=vary_baseline)
    for i in range(0, len(maxima)):
        y_max = x[np.argmin(np.absolute(y - maxima[i]))]
        params.add('center_%s' % i, value=maxima[i], min=maxima[i]-window, max=maxima[i]+window)
        params.add('amplitude_%s' % i, value=y_max, min=0.)
        if function.lower() in ['pv', 'pseudo-voigt', 'psuedo-voigt']:
            params.add('sigma_%s' % i, value=10., min=2., max=2.*max_sigma)
            params.add('eta_%s' % i, value=0.5, min=0., max=1.)
        elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac']:
            params.add('width_%s' % i, value=10., min=2., max=2*max_sigma)
            params.add('rounding_%s' % i, value=5., min=0.)
        elif function.lower() in ['l', 'lorentz', 'lorentzian']:
            params.add('gamma_%s' % i, value=10., min=2., max=2*max_sigma)
        else:
            params.add('sigma_%s' % i, value=10., min=2., max=2.*max_sigma)
    if debug == True:
        print("        initial parameters:")
        print(params.pretty_print())
    if function.lower() in ['pv', 'pseudo-voigt', 'psuedo-voigt']:
        fit_output = lmfit.minimize(multiPV_fit, params, args=(x, y, maxima))
        fit_curve = multiPV_curve(x, fit_output.params, maxima)
    elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac']:
        fit_output = lmfit.minimize(multiFD_fit, params, args=(x, y, maxima))
        fit_curve = multiFD_curve(x, fit_output.params, maxima)
    elif function.lower() in ['l', 'lorentz', 'lorentzian']:
        fit_output = lmfit.minimize(multiL_fit, params, args=(x, y, maxima))
        fit_curve = multiL_curve(x, fit_output.params, maxima)
    else:
        fit_output = lmfit.minimize(multiG_fit, params, args=(x, y, maxima))
        fit_curve = multiG_curve(x, fit_output.params, maxima)
    if debug == True:
        print("        fit status: ", fit_output.message)
        print("        fitted parameters:")
        print(fit_output.params.pretty_print())
    return fit_output, fit_curve

def integrate_peak(x, y, x_start, x_end, mode=Mode, debug=False):
    # function for getting area under a band
    sliced = np.ravel(np.where((x_start <= x) & (x <= x_end)))
    if debug == True:
        print(np.shape(x), np.shape(y), np.shape(sliced))
    integr = np.trapz(y[sliced], x[sliced])
    if mode == 'Transmittance':
        integr = np.trapz(100.*np.ones_like(x[sliced]), x[sliced]) - integr
    return integr

def fit_spectral_range(x, y, maxima, x_start=400, x_end=4000, mode=Mode, Fit_function='gaussian', max_fwhm=100., window=20., vary_baseline=False, debug=False, plot=False):
    # function for fitting a spectral range for certain peaks
    x_slice = x[np.where((x_start <= x) & (x <= x_end))]
    y_slice = y[np.where((x_start <= x) & (x <= x_end))]
    # prepare dict for results
    fitted_peaks = {'function': [], 'centers': [], 'amplitudes': [], 'fwhm': [], 'centers_err': [], 'amplitudes_err': [], 'fwhm_err': [], 'integrated_intensities': []}
    if Fit_function.lower() in ['pv', 'pseudo-voigt', 'pseudo voigt']:
        function = 'pv'
        fitted_peaks['sigmas'] = []
        fitted_peaks['sigmas_err'] = []
        fitted_peaks['etas'] = []
        fitted_peaks['etas_err'] = []
    elif Fit_function.lower() in ['fd', 'fermi-dirac', 'fermi dirac']:
        function = 'fd'
        fitted_peaks['rounds'] = []
        fitted_peaks['rounds_err'] = []
        fitted_peaks['widths'] = []
        fitted_peaks['widths_err'] = []
    elif Fit_function.lower() in ['l', 'lorentz', 'lorentzian']:
        function = 'l'
        fitted_peaks['gammas'] = []
        fitted_peaks['gammas_err'] = []
    else:
        function = 'g'
        fitted_peaks['sigmas'] = []
        fitted_peaks['sigmas_err'] = []
    # do fit
    fit_output, fit_curve = peak_fit_script(x_slice, y_slice, maxima, window=window, max_fwhm=max_fwhm, function='gaussian', vary_baseline=vary_baseline, debug=debug)
    for i in range(0, len(maxima)):
        # add parameters to storage array
        fitted_peaks['function'].append(function)
        for prop in ['center', 'amplitude', 'sigma', 'gamma', 'eta', 'width', 'round']:
            key = prop+"_%s" % i
            if key in fit_output.params.keys():
                fitted_peaks["%ss" % prop].append(fit_output.params[key].value)
                if fit_output.params[key].stderr != None:
                    fitted_peaks[prop+"s_err"].append(fit_output.params[key].stderr)
                else:
                    fitted_peaks[prop+"s_err"].append(0.)
        if function == 'fd':
            # for Fermi-Dirac functions, FWHM is defined as twice the half-width
            fitted_peaks['fwhm'].append(2. * fit_output.params['width_%s' % i].value)
            if fit_output.params['width_%s' % i].stderr != None:
                fitted_peaks['fwhm_err'].append(2. * fit_output.params['width_%s' % i].stderr)
            else:
                fitted_peaks['fwhm_err'].append(0.)
        elif function == 'l':
            # for Lorentzian functions, FWHM is defined as twice gamma
            fitted_peaks['fwhm'].append(2. * fit_output.params['gamma_%s' % i].value)
            if fit_output.params['gamma_%s' % i].stderr != None:
                fitted_peaks['fwhm_err'].append(2. * fit_output.params['gamma_%s' % i].stderr)
            else:
                fitted_peaks['fwhm_err'].append(0.)
        else:
            # for pseudo-voigt and gaussian functions, FWHM is defined as 2*sqrt(2)*sigma
            fitted_peaks['fwhm'].append(2.355 * fit_output.params['sigma_%s' % i].value)
            if fit_output.params['sigma_%s' % i].stderr != None:
                fitted_peaks['fwhm_err'].append(2.355 * fit_output.params['sigma_%s' % i].stderr)
            else:
                fitted_peaks['fwhm_err'].append(0.)
        # generate curve and integrate
        params = lmfit.Parameters()
        params.add('gradient', value=0.)
        params.add('intercept', value=0.)
        params.add('amplitude_0', value=fit_output.params["amplitude_%s" % i])
        params.add('center_0', value=fit_output.params["center_%s" % i])
        if function == 'pv':
            params.add('sigma_0', value=fit_output.params["sigma_%s" % i])
            params.add('eta_0', value=fit_output.params["eta_%s" % i])
            peak_curve = multiPV_curve(x_slice, params, [fit_output.params["center_%s" % i]])
        elif function == 'fd':
            params.add('width_0', value=fit_output.params["width_%s" % i])
            params.add('round_0', value=fit_output.params["round_%s" % i])
            peak_curve = multiFD_curve(x_slice, params, [fit_output.params["center_%s" % i]])
        elif function == 'l':
            params.add('gamma_0', value=fit_output.params["gamma_%s" % i])
            peak_curve = multiL_curve(x_slice, params, [fit_output.params["center_%s" % i]])
        else:
            params.add('sigma_0', value=fit_output.params["sigma_%s" % i])
            peak_curve = multiG_curve(x_slice, params, [fit_output.params["center_%s" % i]])
        fitted_peaks['integrated_intensities'].append(np.trapz(peak_curve, x_slice))
    
    if plot == True:
        # plot results of fitting
        plt.figure(figsize=(8,6))
        # ax1: results of fit
        ax1 = plt.subplot2grid((4,5), (0,0), colspan=4, rowspan=3)
        ax1.set_title("%s\n%0.f-%0.f cm$^{-1}$ %s Peak Fitting" % (sample_data['ID'][i], x_start, x_end, function))
        ax1.set_ylabel("Average Intensity")
        # ax2: residuals
        ax2 = plt.subplot2grid((4,5), (3,0), colspan=4, sharex=ax1)
        ax2.set_xlabel("Frequency (cm$^{-1}$)")
        ax2.set_ylabel("Residual")
        # histogram of residuals
        ax3 = plt.subplot2grid((4,5), (3,4))
        ax3.set_yticks([])
        # determine y limits for residual, hist plots
        y_min = np.amin(y_slice-fit_curve)
        y_max = np.amax(y_slice-fit_curve)
        res_min = y_min - 0.1*(y_max-y_min)
        res_max = y_max + 0.1*(y_max-y_min)
        ax2.set_ylim(res_min, res_max)
        # plot input data and residuals
        ax1.plot(x_slice, y_slice, 'k')
        ax2.plot(x_slice, y_slice-fit_curve, 'k')
        ax3.hist(y_slice-fit_curve, range=(res_min, res_max), bins=20, orientation='horizontal', color='k')
        for i in range(len(maxima)):
            # plot and report peak positions
            plt.figtext(0.78, 0.93-0.08*i, "Center %s: %.1f" % (i+1, fit_output.params['center_%s' % i]))
            plt.figtext(0.78, 0.9-0.08*i, " FWHM %s: %.1f" % (i+1, 2.355*fit_output.params['sigma_%s' % i]))
            ax1.axvline(fit_output.params["center_%s" % i], color='k', linestyle=':')
            params = lmfit.Parameters()
            params.add('gradient', value=fit_output.params["gradient"])
            params.add('intercept', value=fit_output.params["intercept"])
            params.add('amplitude_0', value=fit_output.params["amplitude_%s" % i])
            params.add('center_0', value=fit_output.params["center_%s" % i])
            params.add('sigma_0', value=fit_output.params["sigma_%s" % i])
            peak_curve = multiG_curve(x_slice, params, [fit_output.params["center_%s" % i]])
            ax1.plot(x_slice, peak_curve, 'b:')
        total_curve = multiG_curve(x_slice, fit_output.params, maxima)
        ax1.plot(x_slice, total_curve, 'b--')
        y_max = np.amax([np.amax(y_slice), np.amax(total_curve)])
        ax1.set_xlim(x_start, x_end)
        ax1.set_ylim(np.amin([-0.2*y_max, np.amin(y_slice), np.amin(total_curve)]), 1.2*y_max)
        plt.show()
    return fitted_peaks

"""
# ==================================================
# import FTIR spectra from files
# ==================================================
"""

# ==================================================
# find FTIR spectra files in Data_dir folder

print()
print("searching for FTIR spectrum files...")

files = sorted(glob.glob('%s*.CSV' % Data_dir))
print()
print("files found:", len(files))
    
# ==================================================
# import FTIR spectra into ftir datadict

ftir = {
    'ID': [],
    'sample': [],
    'spec_num': [],
    'measurement_date': [],
    'note': [],
    'frequency': [],
    'wavelength': [],
    'x_start': [],
    'x_end': [],
    'transmittance': [],
    'absorbance': [],
    'material': [],
    'fig_dir': [],
    'out_dir': []
}

print()

count = 0
for file in files:
    if 'background' not in file:
        count += 1
        filename = file.split("/")[-1][:-4]
        print()
        print("%s: attempting to import data from %s" % (count, file.split("/")[-1]))
        while True:
            try:
                # get sample and spec info from filename
                sample = filename.split("_")[1]
                spec_num = filename.split("_")[2]
                date = datetime.datetime.strptime(filename.split("_")[0], "%Y-%m-%d")
                note = ""
                if len(filename.split("_")) > 3:
                    note = filename.split("_")[3]
                print("    sample name: ", sample)
                print("    spectrum num:", spec_num)
                print("    measured on: ", date.strftime("%Y-%m-%d"))
                print("    note:        ", note)
                    
                # import spectrum
                spec = np.genfromtxt(file, delimiter=',').transpose()
                absorb = transmittance_to_absorbance(spec[1])
                print("    spec file shape:", np.shape(spec))
                sliced = np.where(absorb != np.inf)
                print("        zeros/infinities trimmed:", np.count_nonzero(absorb == np.inf), "at", spec[0][absorb == np.inf])
                
                # add data to ftir dict
                print("    adding data to array")
                ftir['ID'].append(filename)
                ftir['sample'].append(sample)
                ftir['spec_num'].append(spec_num)
                ftir['measurement_date'].append(date)
                ftir['note'].append(note)
                ftir['frequency'].append(spec[0][sliced])
                ftir['wavelength'].append(10000./spec[0][sliced])
                ftir['x_start'].append(np.amin(spec[0]))
                ftir['x_end'].append(np.amax(spec[0]))
                ftir['transmittance'].append(spec[1][sliced])
                ftir['absorbance'].append(absorb[sliced])
                ftir['fig_dir'].append("%sby sample/%s/" % (Fig_dir, sample))
                ftir['out_dir'].append("%sby sample/%s/" % (Out_dir, sample))
                # check if this sample has material assignment in sample database:
                assignment = get_chemicalID(sample_database, sample, debug=False)
                if assignment != '':
                    ftir['material'].append(assignment)
                    print("    material assignment:", assignment)
                else:
                    ftir['material'].append("unassigned")
                    print("    sample does not have an assigned material")
                # create output/figure directories if required
                if not os.path.exists(ftir['fig_dir'][-1]):
                    os.makedirs(ftir['fig_dir'][-1])
                if not os.path.exists(ftir['out_dir'][-1]):
                    os.makedirs(ftir['out_dir'][-1])
                print("    success!")
                break
            except Exception as e:
                print("        something went wrong! Exception:", e)
                break
                
print()

# clean up arrays
for key in ftir.keys():
    ftir[key] = np.asarray(ftir[key])
    if len(ftir[key]) != len(ftir['ID']):
        print("WARNING: %s array length does not match ID array length!" % key)
        
# generate spectrum numbers based on measurement date
print(np.shape(ftir['ID']), np.shape(ftir['measurement_date']))
sort = np.lexsort((ftir['ID'], ftir['measurement_date']))
print(sort)
temp = ftir['measurement_date'][sort]
print(temp[0], temp[1])
ftir['spec_num'] = np.asarray([str(num+1).zfill(4) for num in sort])
print(ftir['spec_num'])

print()
print("files imported:", len(ftir['ID']))

for i in range(0, len(ftir['ID'])):
    print("%4i %-30s %4s %-5s" % (i, ftir['ID'][i], ftir['sample'][i], ftir['spec_num'][i]), ftir['material'][i])

# trim iterable lists down to those with FTIR data only
Sample_IDs = [sample for sample in Sample_IDs if sample in ftir['sample']]
materials = [material for material in Materials if np.any(sample_database['Assignment'] == material)]

print()
print("samples found in FTIR data:", len(Sample_IDs))
print("materials found in FTIR data:", len(materials))
        
# generate material-specific figure and output folders
spec_count = 0
print()
for material in materials:
    indices = sample_database['Assignment'].values == material
    samples = sample_database['ID'].iloc[indices].unique()
    spectra = [np.size(ftir['absorbance'][i], axis=0) for i in range(0, len(ftir['sample'])) if ftir['sample'][i] in samples]
    if len(spectra) > 0:
        print("%s: %d spectra" % (material, len(spectra)))
        spec_count += len(spectra)
        figdir = '%sby material/%s/' % (Fig_dir, material)
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        outdir = '%sby material/%s/' % (Out_dir, material)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
print("total spectra with material assignments:", spec_count)
                    
"""
# ==================================================
# process data and generate figures
# ==================================================
"""

print()
print("averaging and interpolating sample spectra...")

x_start, x_end = (400, 4000)         # start and end of x range for interpolating average spectra (in cm-1)
plot_start, plot_end = (600, 1800)  # start and end of x range for plotting average spectra (in cm-1)

# generate standardised x values for interpolation
standard_x = np.linspace(x_start, x_end, 2*int(x_end-x_start))
print()
print("x range of %i points between %0.f and %0.f cm-1" % (np.size(standard_x), np.amin(standard_x), np.amax(standard_x)))

# create second datadict for averaged data by sample
sample_data = {
    'sample': [],
    'material': [],
    'frequency': standard_x.copy(),
    'wavelength': 10000./standard_x.copy(),
    'transmittance': [],
    'absorbance': []
}

for sample in Sample_IDs:
    sample_data['sample'].append(sample)
    # get mean Transmittance spectrum
    x, y = get_sample_mean_spectrum(ftir, sample, mode='Transmittance', debug=False)
    # interpolate to match standardised x and add to array
    y_interp = np.interp(sample_data['frequency'], x, y)
    sample_data['transmittance'].append(y_interp)
    
    # get mean Absorbance spectrum
    x, y = get_sample_mean_spectrum(ftir, sample, mode='Absorbance', debug=False)
    # interpolate to match standardised x and add to array
    y_interp = np.interp(sample_data['frequency'], x, y)
    sample_data['absorbance'].append(y_interp)
    
    # get material assignment
    material = get_chemicalID(sample_database, sample)
    sample_data['material'].append(material)
    
    if Plot_sample_summary == True:
        # produce figure summarising data for this sample
        result = np.ravel(np.where(ftir['sample'] == sample))
        # plot results
        plt.figure(figsize=(8,4))
        plt.title("%s (%s) FTIR Spectra" % (sample, material))
        color = Material_colors[Materials.index(material)]
        count = 1
        y_max = 0.
        for i in result:
            plt.plot(ftir['frequency'][i], ftir[Mode.lower()][i], color, label='spec %s' % count)
            if np.amax(ftir[Mode.lower()][i]) > y_max:
                y_max = np.amax(ftir[Mode.lower()][i])
            count += 1
        if len(result) > 1:
            plt.plot(sample_data['frequency'], sample_data[Mode.lower()][-1], 'k', label="average")
        if Mode == 'Transmittance':
            plt.ylabel("Transmittance (%)")
            plt.ylim(0, 100)
        else:
            plt.ylabel("Absorbance")
            plt.ylim(-0.2*y_max, 1.2*y_max)
        plt.xlim(plot_start, plot_end)
        plt.xlabel("Frequency (cm$^{-1}$)")
        plt.legend()
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig("%sby sample/%s/%s_mean.png" % (Fig_dir, sample, sample), dpi=300)
        plt.savefig("%sby sample/%s/%s_mean.svg" % (Fig_dir, sample, sample), dpi=300)
        plt.show()
            
        # plot standardised and 1st derivative spectra
        y_snv = (sample_data[Mode.lower()][-1] - np.mean(sample_data[Mode.lower()][-1])) / np.std(sample_data[Mode.lower()][-1])
        print(np.shape(sample_data['frequency']), np.shape(y_snv))
        y_snv_1 = savgol_filter(y_snv, 25, polyorder = 5, deriv=1)
        plt.figure(figsize=(8,6))
        plt.subplot(311)
        plt.title("%s (%s)" % (sample, material))
        plt.plot(sample_data['frequency'], sample_data[Mode.lower()][-1], 'k')
        plt.xlim(plot_start, plot_end)
        plt.minorticks_on()
        plt.subplot(312)
        plt.title("Standardised")
        plt.plot(sample_data['frequency'], y_snv, 'r')
        plt.xlim(plot_start, plot_end)
        plt.minorticks_on()
        plt.subplot(313)
        plt.title("First Derivative")
        plt.plot(sample_data['frequency'], y_snv_1, 'b')
        plt.xlim(plot_start, plot_end)
        plt.xlabel("Frequency (cm^{-1})")
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig("%sby sample/%s/%s_standardised.png" % (Fig_dir, sample, sample), dpi=300)
        plt.show()
    
for key in sample_data.keys():
    sample_data[key] = np.asarray(sample_data[key])

"""
# ==================================================
# plot spectra by material
# ==================================================
"""

x_start, x_end = (400, 4000)

normalise = True
offset = 0.

if Plot_material_summary == True:
    for material in Materials:
        color = Material_colors[Materials.index(material) % len(Material_colors)]
        result = np.ravel(np.where(sample_data['material'] == material))
        if len(result) > 0:
            print()
            print(f"plotting {len(result)} spectra for {material} samples, color: {color}")
            plt.figure(figsize=(10,8))
            ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)
            ax2 = plt.subplot2grid((2,3), (1,0), colspan=2, sharey=ax1)
            ax3 = plt.subplot2grid((2,3), (1,2), sharey=ax1)
            ax1.set_title(material)
            ax1.set_xlabel("Frequency (cm$^{-1}$)")
            ax2.set_xlabel("Frequency (cm$^{-1}$)")
            ax3.set_xlabel("Frequency (cm$^{-1}$)")
            ax1.set_xlim(x_start, x_end)
            ax2.set_xlim(x_start, 1800)
            ax3.set_xlim(2800, 3100)
            if normalise == True:
                ax1.set_ylabel(f"Normalised {Mode}")
                ax2.set_ylabel(f"Normalised {Mode}")
                ax1.set_ylim(0, 1)
            elif Mode == 'Transmittance':
                ax1.set_ylabel("Transmittance (%)")
                ax2.set_ylabel("Transmittance (%)")
                ax1.set_ylim(0, 100)
            else:
                ax1.set_ylabel(Mode)
                ax2.set_ylabel(Mode)
            count = 0
            for i in result:
                # get info for this sample
                x = sample_data['frequency']
                y = sample_data[Mode.lower()][i]
                label = sample_data['sample'][i]
                if normalise == True:
                    y_min = find_min(x, y, x_start, x_end)[1]
                    y_max = find_max(x, y, x_start, x_end)[1]
                    ax1.plot(x, (y-y_min)/(y_max-y_min)+count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                    ax2.plot(x, (y-y_min)/(y_max-y_min)+count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                    ax3.plot(x, (y-y_min)/(y_max-y_min)+count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                else:
                    ax1.plot(x, y+count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                    ax2.plot(x, y+count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                    ax3.plot(x, y+count*offset, color, alpha=1./np.sqrt(len(result)), label=label)
                count += 1
            if offset > 0:
                ax1.set_yticks([])
                if normalise == True:
                    ax1.set_ylim(-0.2, (1.2-offset)+count*offset)
                elif Mode == 'Transmittance':
                    ax1.set_ylim(-40, 50+count*offset)
                else:
                    ax1.set_ylim(0.5, 0.5+count*offset)
            else:
                if normalise == True:
                    ax1.set_ylim(-0.2, 1.2)
                elif Mode == 'Transmittance':
                    ax1.set_ylim(0, 110)
                else:
                    ax1.set_ylim(0.5, np.ceil(y_max))
                    
            ### plt.legend()
            plt.tight_layout()
            plt.savefig("%sby material/%s_spectra.png" % (Fig_dir, material), dpi=300)
            plt.show()

"""
# ==================================================
# plot mean spectra for all materials
# ==================================================
"""

offset = 0.8
normalise = True

if Plot_material_summary == True:
    print()
    print("plotting mean spectra for each material")
    plt.figure(figsize=(10,8))
    ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)
    ax2 = plt.subplot2grid((2,3), (1,0), colspan=2, sharey=ax1)
    ax3 = plt.subplot2grid((2,3), (1,2), sharey=ax1)
    ax1.set_title("Mean Spectra by Material")
    ax1.set_xlabel("Frequency (cm$^{-1}$)")
    ax2.set_xlabel("Frequency (cm$^{-1}$)")
    ax3.set_xlabel("Frequency (cm$^{-1}$)")
    ax1.set_xlim(400, 4000)
    ax2.set_xlim(400, 1800)
    ax3.set_xlim(2800, 3000)
    if normalise == True:
        ax1.set_ylabel(f"Normalised {Mode}")
        ax2.set_ylabel(f"Normalised {Mode}")
        ax1.set_ylim(0, 1)
    elif Mode == 'Transmittance':
        ax1.set_ylabel("Transmittance (%)")
        ax2.set_ylabel("Transmittance (%)")
        ax1.set_ylim(0, 100)
    else:
        ax1.set_ylabel("Absorbance")
        ax2.set_ylabel("Absorbance")
    count = 0
    for material in ['not plastic', 'pumice', 'unassigned', 'PE', 'PP', 'PPE', 'PS']:
        print()
        result = np.ravel(np.where(sample_data['material'] == material))
        print(material, np.count_nonzero(sort))
        color = Material_colors[Materials.index(material) % len(Material_colors)]
        if len(result) > 0:
            x_av = sample_data['frequency']
            y_av = np.mean(sample_data[Mode.lower()][result], axis=0)
            print(np.shape(x_av), np.shape(y_av))
            if normalise == True:
                y_min = find_min(x_av, y_av, 400, 4000)[1]
                y_max = find_max(x_av, y_av, 400, 4000)[1]
                ax1.plot(x_av, (y_av-y_min)/(y_max-y_min)+count*offset, color, label=material)
                ax2.plot(x_av, (y_av-y_min)/(y_max-y_min)+count*offset, color, label=material)
                ax3.plot(x_av, (y_av-y_min)/(y_max-y_min)+count*offset, color, label=material)
            else:
                ax1.plot(x_av, y_av+count*offset, color, linestyle=style, label=material)
                ax2.plot(x_av, y_av+count*offset, color, linestyle=style, label=material)
                ax3.plot(x_av, y_av+count*offset, color, linestyle=style, label=material)
            count += 1
    if offset > 0:
        ax1.set_yticks([])
        if normalise == True:
            ax1.set_ylim(-0.2, (1.2-offset)+count*offset)
        elif Mode == 'Transmittance':
            ax1.set_ylim(-40, 50+count*offset)
        else:
            ax1.set_ylim(0.5, 0.5+count*offset)
    else:
        if normalise == True:
            ax1.set_ylim(-0.2, 1.2)
        elif Mode == 'Transmittance':
            ax1.set_ylim(0, 110)
        else:
            ax1.set_ylim(0.5, np.ceil(y_max))
    ax1.legend(loc='upper right')
    ax3.set_yticks([])
    plt.tight_layout()
    plt.savefig("%sby material/material_mean_spectra.png" % (Fig_dir), dpi=300)
    plt.show()
    
if Plot_material_summary == True:
    # plot overlay comparison of PE/PP/PPE polymers
    print()
    print("plotting overlaid spectra for PE, PP, PPE")
    
    offset = 0.
    plot_std = False
    
    plt.figure(figsize=(10,8))
    ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)
    ax2 = plt.subplot2grid((2,3), (1,0), colspan=2, sharey=ax1)
    ax3 = plt.subplot2grid((2,3), (1,2))
    ax1.set_title("Mean Spectra by Polymer")
    ax1.set_xlabel("Frequency (cm$^{-1}$)")
    ax2.set_xlabel("Frequency (cm$^{-1}$)")
    ax3.set_xlabel("Frequency (cm$^{-1}$)")
    ax1.set_xlim(400, 4000)
    ax2.set_xlim(400, 1800)
    ax3.set_xlim(2800, 3100)
    if normalise == True:
        ax1.set_ylabel(f"Normalised {Mode}")
        ax2.set_ylabel(f"Normalised {Mode}")
        ax1.set_ylim(0, 1)
    elif Mode == 'Transmittance':
        ax1.set_ylabel("Transmittance (%)")
        ax2.set_ylabel("Transmittance (%)")
        ax1.set_ylim(0, 100)
    else:
        ax1.set_ylabel("Absorbance")
        ax2.set_ylabel("Absorbance")
    count = 0
    for material in ['PE', 'PP', 'PPE', 'PS']:
        ### print()
        result = np.ravel(np.where(sample_data['material'] == material))
        ### print(material, np.count_nonzero(sort))
        color = Material_colors[Materials.index(material) % len(Material_colors)]
        if len(result) > 0:
            x_av = sample_data['frequency']
            y_av = np.mean(sample_data[Mode.lower()][result], axis=0)
            y_std = np.std(sample_data[Mode.lower()][result], axis=0)
            ### print(np.shape(x_av), np.shape(y_av))
            if normalise == True:
                y_min = find_min(x_av, y_av, 400, 4000)[1]
                y_max = find_max(x_av, y_av, 400, 4000)[1]
                if len(result) >= 5 and plot_std == True:
                    ax1.fill_between(x_av, (y_av+y_std-y_min)/(y_max-y_min)+count*offset, (y_av-y_std-y_min)/(y_max-y_min)+count*offset, color=color, linewidth=0., alpha=0.1)
                    ax2.fill_between(x_av, (y_av+y_std-y_min)/(y_max-y_min)+count*offset, (y_av-y_std-y_min)/(y_max-y_min)+count*offset, color=color, linewidth=0., alpha=0.1)
                    ax3.fill_between(x_av, (y_av+y_std-y_min)/(y_max-y_min)+count*offset, (y_av-y_std-y_min)/(y_max-y_min)+count*offset, color=color, linewidth=0., alpha=0.1)
                ax1.plot(x_av, (y_av-y_min)/(y_max-y_min)+count*offset, color, label=material)
                ax2.plot(x_av, (y_av-y_min)/(y_max-y_min)+count*offset, color, label=material)
                ax3.plot(x_av, (y_av-y_min)/(y_max-y_min)+count*offset, color, label=material)
            else:
                if len(result) >= 3 and plot_std == True:
                    ax1.fill_between(x_av, y_av+y_std+count*offset, y_av-y_std+count*offset, color=color, linewidth=0., alpha=0.1)
                    ax2.fill_between(x_av, y_av+y_std+count*offset, y_av-y_std+count*offset, color=color, linewidth=0., alpha=0.1)
                    ax3.fill_between(x_av, y_av+y_std+count*offset, y_av-y_std+count*offset, color=color, linewidth=0., alpha=0.1)
                ax1.plot(x_av, y_av+count*offset, color, linestyle=style, label=material)
                ax2.plot(x_av, y_av+count*offset, color, linestyle=style, label=material)
                ax3.plot(x_av, y_av+count*offset, color, linestyle=style, label=material)
            count += 1
    if offset > 0:
        ax1.set_yticks([])
        if normalise == True:
            ax1.set_ylim(-0.2, 1.2-offset+count*offset)
            ax3.set_ylim(-0.2, 1.2-offset+count*offset)
        elif Mode == 'Transmittance':
            ax1.set_ylim(0, 120-offset+count*offset)
            ax3.set_ylim(0, 120-offset+count*offset)
        else:
            ax1.set_ylim(0.5, 0.5+count*offset)
            ax3.set_ylim(0.5, 0.5+count*offset)
    else:
        if normalise == True:
            ax1.set_ylim(-0.1, 1.1)
            ax3.set_ylim(-0.1, 1.1)
        elif Mode == 'Transmittance':
            ax1.set_ylim(0, 110)
            ax3.set_ylim(0, 110)
        else:
            ax1.set_ylim(0.5, np.ceil(y_max))
            ax3.set_ylim(0.5, np.ceil(y_max))
    ax1.legend(loc='upper right')
    ax3.set_yticks([])
    plt.tight_layout()
    plt.savefig("%sby material/PE-PP-PPE_comparison.png" % (Fig_dir), dpi=300)
    plt.show()

            
"""
# ==================================================
# Peak Detection and Peak Fitting
# ==================================================
"""

subtract_baseline = True                    # subtract a baseline first?
baseline_type = 'polynomial'                # type of baseline ('linear' or 'polynomial')
baseline_order = 5                          # polynomial order
baseline_points = [450, 500, 600, 700, 750, 800, 1200, 1300, 1490, 1800, 1900, 2000, 2100, 2500, 2600, 2700, 3700, 3800, 3900, 4000]  # points on x to fit baseline to

rel_intensity_threshold = 0.3               # minimum intensity for peak detection vs spectrum max
SNR_threshold = 8.                          # minimum signal:noise ratio for peak detection

show_plots = False                          # show plots in viewer
plot_peak_summary = True                    # produce figure summarising peak results for each spectrum

if Fit_peaks == True:
    ftir['detected_peaks'] = []
    ftir['fitted_peaks'] = []
    
    print()
    for i in range(0, len(sample_data['sample'])):
        # for each sample average spectrum
        sample = sample_data['sample'][i]
        print()
        print(f"{i}/{len(ftir['sample'])} {ftir['ID'][i]}")
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(111)
        ax1.set_title("%s\nPeak Detection" % sample)
        
        x = sample_data['frequency']
        if subtract_baseline == True:
            y = baseline(x, sample_data[Mode.lower()][i], baseline_points, base=baseline_type, max_order=baseline_order, debug=False, plot=False)
        else:
            y = sample_data[Mode.lower()][i]
        
        # find maxima in spectrum that are above relative intensity threshold
        y_min = np.amin(y[np.where((600 <= x) & (x <= 3500))])
        y_max = np.amax(y[np.where((600 <= x) & (x <= 3500))])
        
        ax1.axhline(rel_intensity_threshold*y_max, color='b', linestyle=':', label='minimum intensity')
        
        window = 10
        sliced = np.ravel(np.where((700 <= x) & (x <= 3500)))
        maxima = find_peaks(x[sliced], y[sliced], window, rel_intensity_threshold, mode='absorbance')
        noise = [np.std(y[np.where((500 <= x) & (x <= 600))]), np.std(y[np.where((1800 <= x) & (x <= 1900))])]
        noise = np.amax(noise)
        plt.plot(x, y, 'k', label='data')
        print("    %s peak detection:" % ftir['sample'][i], maxima[0])
        print("        noise level: %0.1f counts" % noise)
        
        # remove maxima that are below SNR threshold
        ax1.axhline(float(SNR_threshold)*noise, color='r', linestyle=':', label='minimum SNR')
        maxima_pass = [[],[]]
        for i2 in range(0, len(maxima[0])):
            if maxima[1,i2] > float(SNR_threshold) * noise:
                maxima_pass[0].append(maxima[0,i2])
                maxima_pass[1].append(maxima[1,i2])
                ax1.text(maxima[0,i2], maxima[1,i2]+0.05*(y_max-y_min), "%0.f" % maxima[0,i2], rotation=90, va='bottom', ha='left')
        maxima_pass = np.asarray(maxima_pass)
        ax1.plot(maxima[0], maxima[1], 'ro', label='fail  (%0.0f)' % (len(maxima[0])-len(maxima_pass[0])))
        
        # plot remaining maxima
        ax1.plot(maxima_pass[0], maxima_pass[1], 'bo', label='pass (%0.0f)' % len(maxima_pass[0]))
        ax1.set_xlabel("Frequency (cm$^{-1}$)")
        if Mode == 'Transmittance':
            ax1.set_ylabel("Transmittance (%)")
        else:
            ax1.set_ylabel("Absorbance")
        # create second y axis for SNR values
        ax2 = ax1.twinx()
        ax2.set_ylim(intensity2snr(ax1.get_ylim(), noise))
        ax2.set_ylabel("SNR")
        ax1.legend()
        plt.tight_layout()
        plt.savefig("%s%s_detected-peaks.png" % (ftir['fig_dir'][i], ftir['ID'][i]), dpi=300)
        plt.show()
        # add maxima positions, intensities to array
        ftir["detected_peaks"].append(maxima_pass)
        
        # divide spectrum into distinct regions for individual fitting
        regions = []
        window = 200.
        fitted_regions_x = []
        fitted_regions_y = []
        if len(maxima_pass[0]) > 1:
            print()
            print("    generating fit regions for %s..." % ftir['ID'][i])
            # group peaks into fit regions based on window size
            # CAUTION: REQUIRES MAXIMA TO BE IN ASCENDING ORDER
            temp = [maxima_pass[0][0]]
            ### print("            ", temp)
            for i2 in range(1, len(maxima_pass[0])):
                local_max = maxima_pass[0][i2]
                if local_max - temp[-1] < 1.5*window:
                    # if gap between this peak and the last is <150% of the window size, add to temp group
                    temp.append(local_max)
                    ### print("            ", temp)
                else:
                    # otherwise create region for previous group and start new temp group
                    x_start = np.amin(temp) - window
                    x_end = np.amax(temp) + window
                    # round up/down to nearest 10
                    x_start = np.floor(x_start/10.)*10.
                    x_end = np.ceil(x_end/10.)*10.
                    # check region doesn't fall outside the data range
                    if np.amin(ftir['frequency'][i]) > x_start:
                        x_start = np.amin(ftir['freq'][i])
                    if np.amax(ftir['frequency'][i]) < x_end:
                        x_end = np.amax(ftir['frequency'][i])
                    regions.append([x_start, x_end])
                    ### print("        region %s: %0.f - %0.f cm-1, %s peaks" % (len(regions), x_start, x_end, len(temp)))
                    temp = [local_max]
                    ### print("            ", temp)
            # then resolve final region
            x_start = np.amin(temp) - window
            x_end = np.amax(temp) + window
            # round up/down to nearest 10
            x_start = np.floor(x_start/10.)*10.
            x_end = np.ceil(x_end/10.)*10.
            # check region doesn't fall outside the data range
            if np.amin(x) > x_start:
                x_start = np.amin(x)
            if np.amax(x) < x_end:
                x_end = np.amax(x)
            regions.append([x_start, x_end])
            print("        region %s: %0.f - %0.f cm-1, %s peaks" % (len(regions), x_start, x_end, len(temp)))
        elif len(maxima_pass[0]) == 1:
            print()
            print("    generating single fit region for %s..." % ftir['ID'][i])
            # create single region around only peak
            local_max = maxima_pass[0][0]
            x_start = local_max - window
            x_end = local_max + window
            # round up/down to nearest 10
            x_start = np.floor(x_start/10.)*10.
            x_end = np.ceil(x_end/10.)*10.
            # check region doesn't fall outside the data range
            if np.amin(x) > x_start:
                x_start = np.amin(x)
            if np.amax(x) < x_end:
                x_end = np.amax(x)
            regions.append([x_start, x_end])
            ### print("            ", [local_max])
            print("        region %s: %0.f - %0.f cm-1, 1 peak" % (len(regions), x_start, x_end))
        else:
            print("    cannot continue with fit, no peaks found!")
        
        fitted_peaks = {'function': [], 'centers': [], 'amplitudes': [], 'fwhm': [], 'centers_err': [], 'amplitudes_err': [], 'fwhm_err': []}
        if len(regions) > 0:
            # proceed with fitting each region separately based on detected peaks
            print()
            print("    at least one region found, proceeding with peak fit...")
            
            # set up arrays depending on specified function
            if Fit_function.lower() in ['pv', 'pseudo-voigt', 'pseudo voigt']:
                function = 'pv'
                fitted_peaks['sigmas'] = []
                fitted_peaks['sigmas_err'] = []
                fitted_peaks['etas'] = []
                fitted_peaks['etas_err'] = []
            elif Fit_function.lower() in ['l', 'lorentz', 'lorentzian']:
                function = 'l'
                fitted_peaks['gammas'] = []
                fitted_peaks['gammas_err'] = []
            else:
                function = 'g'
                fitted_peaks['sigmas'] = []
                fitted_peaks['sigmas_err'] = []
            print("        fitting function:", function.upper())
                
            # for each region to be fit
            for i2 in range(0, len(regions)):
                # slice data to region
                x_start = regions[i2][0]
                x_end = regions[i2][1]
                print()
                print("        region %s: %0.f - %0.f cm-1" % (i2+1, x_start, x_end))
                input_peaks = maxima_pass[0][np.where((x_start < maxima_pass[0]) & (maxima_pass[0] < x_end))]
                print("            %s peaks found:" % len(input_peaks), input_peaks)
                x_slice = x[np.where((x_start <= x) & (x <= x_end))]
                y_slice = y[np.where((x_start <= x) & (x <= x_end))]
                # proceed with fit
                fit_output, fit_curve = peak_fit_script(x_slice, y_slice, input_peaks, function=function, window=10., max_sigma=30.)
                plt.figure(figsize=(8,6))
                # ax1: results of fit
                ax1 = plt.subplot2grid((4,5), (0,0), colspan=4, rowspan=3)
                ax1.set_title("%s\n%0.f-%0.f cm$^{-1}$ %s Peak Fitting" % (ftir['ID'][i], x_start, x_end, Fit_function))
                ax1.set_ylabel("Average Intensity")
                # ax2: residuals
                ax2 = plt.subplot2grid((4,5), (3,0), colspan=4, sharex=ax1)
                ax2.set_xlabel("Frequency (cm$^{-1}$)")
                ax2.set_ylabel("Residual")
                # histogram of residuals
                ax3 = plt.subplot2grid((4,5), (3,4))
                ax3.set_yticks([])
                # determine y limits for residual, hist plots
                y_min = np.amin(y_slice-fit_curve)
                y_max = np.amax(y_slice-fit_curve)
                res_min = y_min - 0.1*(y_max-y_min)
                res_max = y_max + 0.1*(y_max-y_min)
                ax2.set_ylim(res_min, res_max)
                # plot input data and residuals
                ax1.plot(x_slice, y_slice, 'k')
                ax2.plot(x_slice, y_slice-fit_curve, 'k')
                ax3.hist(y_slice-fit_curve, range=(res_min, res_max), bins=20, orientation='horizontal', color='k')
                # plot individual peak fits
                x_temp = np.linspace(x_start, x_end, 10*len(x_slice))
                for i2 in range(0, len(input_peaks)):
                    # add parameters to storage array
                    fitted_peaks['function'].append(function)
                    for prop in ['center', 'amplitude', 'sigma', 'gamma', 'eta']:
                        key = prop + "_%s" % i2
                        if key in fit_output.params.keys():
                            fitted_peaks["%ss" % prop].append(fit_output.params[key].value)
                            if fit_output.params[key].stderr != None:
                                fitted_peaks[prop+"s_err"].append(fit_output.params[key].stderr)
                            else:
                                fitted_peaks[prop+"s_err"].append(0.)
                    
                    if function == 'l':
                        # for Fermi-Dirac functions, FWHM is defined as twice the half-width
                        fitted_peaks['fwhm'].append(2. * fit_output.params['gamma_%s' % i2].value)
                        if fit_output.params['gamma_%s' % i2].stderr != None:
                            fitted_peaks['fwhm_err'].append(2. * fit_output.params['gamma_%s' % i2].stderr)
                        else:
                            fitted_peaks['fwhm_err'].append(0.)
                    else:
                        # for pseudo-voigt and gaussian functions, FWHM is defined as 2*sqrt(2)*sigma
                        fitted_peaks['fwhm'].append(2.355 * fit_output.params['sigma_%s' % i2].value)
                        if fit_output.params['sigma_%s' % i2].stderr != None:
                            fitted_peaks['fwhm_err'].append(2.355 * fit_output.params['sigma_%s' % i2].stderr)
                        else:
                            fitted_peaks['fwhm_err'].append(0.)
                        
                    # plot and report peak positions
                    plt.figtext(0.78, 0.93-0.08*i2, "Center %s: %.1f" % (i2+1, fitted_peaks['centers'][-1]))
                    plt.figtext(0.78, 0.9-0.08*i2, " FWHM %s: %.1f" % (i2+1, fitted_peaks['fwhm'][-1]))
                    ax1.axvline(fit_output.params["center_%s" % i2], color='k', linestyle=':')
                    # create function curve for plotting
                    params = lmfit.Parameters()
                    params.add('gradient', value=fit_output.params["gradient"])
                    params.add('intercept', value=fit_output.params["intercept"])
                    params.add('amplitude_0', value=fit_output.params["amplitude_%s" % i2])
                    params.add('center_0', value=fit_output.params["center_%s" % i2])
                    if function == 'pv':
                        params.add('sigma_0', value=fit_output.params["sigma_%s" % i2])
                        params.add('eta_0', value=fit_output.params["eta_%s" % i2])
                        peak_curve = multiPV_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                    elif function == 'fd':
                        params.add('width_0', value=fit_output.params["width_%s" % i2])
                        params.add('round_0', value=fit_output.params["round_%s" % i2])
                        peak_curve = multiFD_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                    elif function == 'l':
                        params.add('gamma_0', value=fit_output.params["gamma_%s" % i2])
                        peak_curve = multiL_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                    else:
                        params.add('sigma_0', value=fit_output.params["sigma_%s" % i2])
                        peak_curve = multiG_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                    ax1.plot(x_temp, peak_curve, 'b:')
                # plot total peak fit
                if function == 'pv':
                    total_curve = multiPV_curve(x_temp, fit_output.params, input_peaks)
                elif function == 'fd':
                    total_curve = multiFD_curve(x_temp, fit_output.params, input_peaks)
                elif function == 'l':
                    total_curve = multiL_curve(x_temp, fit_output.params, input_peaks)
                else:
                    total_curve = multiG_curve(x_temp, fit_output.params, input_peaks)
                ax1.plot(x_temp, total_curve, 'b--')
                # finish fitting figure
                y_max = np.amax(y_slice)
                ax1.set_xlim(x_start, x_end)
                ax1.set_ylim(np.amin([-0.2*y_max, np.amin(y_slice), np.amin(total_curve)]), 1.2*y_max)
                plt.savefig("%s%s_%0.f-%0.fcm_fit.png" % (ftir['fig_dir'][i], ftir['ID'][i], x_start, x_end), dpi=300)
                plt.show()
                
        # convert results to numpy arrays and add to storage array
        ftir['fitted_peaks'].append({})
        for key in fitted_peaks.keys():
            ftir['fitted_peaks'][i][key] = np.asarray(fitted_peaks[key])
            print(key, np.shape(ftir['fitted_peaks'][i][key]))
        
        # save fit parameters to file
        if len(fitted_peaks['centers']) > 0:
            print()
            print("saving peak fit data to file")
            save_data = []
            header = []
            for prop in ['centers', 'amplitudes', 'fwhm', 'sigmas', 'gammas', 'etas']:
                name = prop
                if prop[-1] == 's':
                    name = prop[:-1]
                if prop in fitted_peaks.keys():
                    save_data.append(fitted_peaks[prop])
                    save_data.append(fitted_peaks[prop+"_err"])
                    header.append(name)
                    header.append(name+" standard error")
            save_data = np.vstack(save_data)
            save_name = "%s_%0.f-%0.fcm" % (ftir['ID'][i], ftir['x_start'][i], ftir['x_end'][i])
            # save data to output folder
            np.savetxt("%s%s_%s-fit-parameters.csv" % (ftir['out_dir'][i], save_name, function.upper()), save_data.transpose(), header=", ".join(header), delimiter=', ')

    print()
    print("sample array:", len(ftir['sample']))
    print("fitted peak array:" , len(ftir['fitted_peaks']))
    
"""
# ==================================================
# run PCA and K-means clustering on spectra
# ==================================================
"""

SNV = True
normalise = True
first_deriv = True

regions = [(400, 1800), (2800, 3200)]

debug = False

if Do_PCA == True:
    print()
    print("running PCA...")
    
    if normalise == True:
        text = '1st-deriv-of-norm'
    else:
        text = '1st-deriv'
        
    # get parameters for spectrum filtering
    for region in regions:
            x_start, x_end = region
            print()
            spec_count = len(ftir['ID'])
            sample_count = len(sample_data['sample'])
            print("    matching spectra:", spec_count)
            print("    matching samples:", sample_count)
            # set up figure
            plt.figure(figsize=(12, 6))
            print()
            print("Running PCA on FTIR region %0.f to %0.f cm-1" % (x_start, x_end))
            x_temp = np.linspace(x_start, x_end, 2*int(x_end - x_start)+1)
            y_temp = []
            mat_temp = []
            sample_temp = []
            id_temp = []
            
            # import and standardise spectra by sample
            for i, sample in enumerate(sample_data['sample']):
                assignment = get_chemicalID(sample_database, sample)
                # interpolate y data based on x_temp
                y = np.interp(x_temp, sample_data['frequency'], sample_data[Mode.lower()][i])
                if debug == True:
                    print("    %s Y arrays:" % sample_data['sample'][i], np.shape(y))
                # do normalisation if necessary
                if SNV == True:
                    y = (y - np.mean(y)) / np.std(y)
                elif normalise == True:
                    y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))
                if first_deriv == True:
                    # Calculate first derivative using a Savitzky-Golay filter
                    y_deriv = savgol_filter(y, 25, polyorder = 5, deriv=1)
                    y_temp.append(y_deriv)
                else:
                    y_temp.append(y)
                sample_temp.append(sample)
                # check material assignments and simply if necessary
                if assignment[0] == '':
                    mat_temp.append("unassigned")
                    print("%s is missing a material assignment!" % sample)
                mat_temp.append(assignment)
            print()
            print("    matching samples found:", len(sample_temp))
            if debug == True:
                print("        ", sample_temp)
                        
            sample_temp = np.asarray(sample_temp)
            mat_temp = np.asarray(mat_temp)
            x_temp = np.asarray(x_temp)
            y_temp = np.asarray(y_temp)
            
            print()
            print("    PCA x,y array shapes:", np.shape(x_temp), np.shape(y_temp))

            if len(sample_temp) > 6:
                # proceed with PCA
                temp = pd.DataFrame(y_temp, columns=x_temp, index=sample_temp)
                ### print(temp.info)

                pca = PCA(n_components=int(np.amin([6,len(sample_temp)])))
                principalComponents = pca.fit_transform(temp)
                principal_frame = pd.DataFrame(data=principalComponents, columns=['principal component '+str(i+1) for i in range(0, pca.n_components_)])

                print("    features:", pca.n_features_)
                print("    components:", pca.n_components_)
                print("    samples:", pca.n_samples_)
                print("    Explained variation per principal component:")
                for i in range(0, pca.n_components_):
                    print("        component %d: %0.3f" % (i+1, pca.explained_variance_ratio_[i]))

                final_frame = pd.concat([principal_frame, pd.DataFrame(mat_temp, columns=['material']), pd.DataFrame(sample_temp, columns=['sample']), pd.DataFrame(id_temp, columns=['ID'])], axis=1)

                # ax1: PCA coordinates plot
                ax1 = plt.subplot(121)
                ax1.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca.explained_variance_ratio_[0]))
                ax1.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca.explained_variance_ratio_[1]))
                ax1.set_title('FTIR, %s-%s cm$^{-1}$' % (x_start, x_end))
                labels = []
                for material, colour in zip(Materials, Material_colors):
                    indices = final_frame['material'] == material
                    if np.any(indices) == True:
                        labels.append(material+" (%s)" % (np.count_nonzero(mat_temp == material)))
                        ax1.scatter(final_frame.loc[indices, 'principal component 1'], final_frame.loc[indices, 'principal component 2'], edgecolors=colour, c=colour)
                ax1.grid()

                ax3 = plt.subplot(122)
                ax3.set_xlabel("Frequency (cm$^{-1}$)")
                ax3.set_ylabel("Variance")
                ax3.set_xlim(x_start, x_end)
                for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
                    comp = comp * var  # scale component by its variance explanation power
                    ax3.plot(x_temp, np.cumsum(comp), label="component %s" % (i+1))
                        
                # finish figure
                ax1.legend(labels)
                ax3.legend()
                plt.tight_layout()
                plt.savefig("%sPCA_%s_%0.f-%0.fcm.png" % (Fig_dir, text, x_start, x_end))
                plt.show()
                    
                if Do_clustering == True:
                    # run K-means
                    print()
                    print("running K-means clustering on FTIR, region %0.f-%0.f cm-1" % (x_start, x_end))

                    n_clusters = 5
                    print("    number of clusters:", n_clusters)

                    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
                    kmeans.fit(principalComponents)

                    for cluster in range(0, n_clusters):
                        print("        spectra in cluster %2d: %0.f, centered at: (%0.3f, %0.3f)" % (cluster, np.count_nonzero(kmeans.labels_ == cluster), kmeans.cluster_centers_[cluster,0], kmeans.cluster_centers_[cluster,1]))
                        for material in np.unique(mat_temp):
                            sort = np.logical_and(final_frame['material'] == material, kmeans.labels_ == cluster)
                            if np.count_nonzero(sort) > 0:
                                print("            %3.f%% %s (%0.f/%0.f %s spectra)" % (100*np.count_nonzero(sort)/np.count_nonzero(kmeans.labels_ == cluster), material, np.count_nonzero(sort), np.count_nonzero(final_frame['material'] == material), material))
                                if material == 'non plastic':
                                    print("                ", final_frame[sort].index.values)

                    plt.figure(figsize=(18,6))
                    ax1 = plt.subplot(131)
                    ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
                    ax3 = plt.subplot(133)
                    ax1.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca.explained_variance_ratio_[0]))
                    ax1.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca.explained_variance_ratio_[1]))
                    ax1.set_title('PCA of FTIR %s-%s cm$^{-1}$' % (x_start, x_end))
                    ax2.set_title('K-Means Clustering')
                    ax3.set_title("Cluster Mean Spectra")
                    ax3.set_xlim(x_start, x_end)
                    
                    # plot data points by material type
                    labels = []
                    for material, colour in zip(Materials, Material_colors):
                        indices = final_frame['material'] == material
                        if np.any(indices) == True:
                            labels.append(material+" (%s)" % (np.count_nonzero(mat_temp == material)))
                            mfc = colour
                            ax1.scatter(final_frame.loc[indices, 'principal component 1'], final_frame.loc[indices, 'principal component 2'], edgecolors=colour, c=mfc)
                    ax1.legend(labels)
                    ax1.grid(zorder=3)

                    # print clustering breakdown of each material group
                    print()
                    for material in np.unique(mat_temp):
                        indices = final_frame['material'] == material
                        bar_offset = 0
                        if np.any(indices) == True:
                            print()
                            print("%s: %0.f spectra in %0.f clusters:" % (material, np.count_nonzero(indices), len(np.unique(kmeans.labels_[indices]))))
                            for i in np.unique(kmeans.labels_[indices]):
                                sort = np.logical_and(final_frame['material'] == material, kmeans.labels_ == i)
                                if np.count_nonzero(sort) > 0:
                                    print("    cluster %2d: %4d spectra (%3.f%%)" % (i, np.count_nonzero(sort), 100.*np.count_nonzero(sort)/np.count_nonzero(indices)))
                    print()

                    # plot clusters as scatter
                    labels = []
                    for cluster in range(0, n_clusters):
                        # iterate over clusters
                        colour = Color_list[cluster]
                        indices = kmeans.labels_ == cluster
                        if np.any(indices) == True:
                            ax2.scatter(final_frame.loc[indices, 'principal component 1'], final_frame.loc[indices, 'principal component 2'], c=colour, label="cluster %0.f (%0.f)" % (cluster, np.count_nonzero(indices)), zorder=1)
                            labels.append("cluster" + str(cluster))
                            ### ax2.scatter(kmeans.cluster_centers_[cluster,0], kmeans.cluster_centers_[cluster,1], edgecolors=colour, c='w', marker='s', label="cluster %0.f (%0.f)" % (cluster, np.count_nonzero(indices)), zorder=2)
                            ax3.plot(x_temp, np.cumsum(np.mean(y_temp[indices], axis=0)), colour, label="cluster %0.f" % (cluster))
                    
                    # finish figure
                    ax2.legend()
                    ax2.grid(zorder=3)
                    ax3.legend()
                    plt.tight_layout()
                    plt.savefig("%sPCA_%s_%0.f-%0.fcm_clusters.png" % (Fig_dir, text, x_start, x_end))
                    plt.show()
                    
        

"""
# ==================================================
# save processed spectra
# ==================================================
"""

print()
print("saving data")

# save individual spectra
print()
for i in range(0, len(ftir['sample'])):
    sample = ftir['sample'][i]
    save_data = np.vstack((ftir['frequency'][i], ftir['wavelength'][i], ftir['transmittance'][i], ftir['absorbance'][i]))
    header = ["frequency (cm-1)", "wavelength (um)", "transmittance (%)", "absorbance (AU)"]
    np.savetxt("%s%s_spectra.csv" % (ftir['out_dir'][i], save_name), save_data.transpose(), header=", ".join(header), delimiter=', ')

# save spectra by sample
print()
for i, sample, in enumerate(sample_data['sample']):
    save_data = np.vstack((sample_data['frequency'], sample_data['wavelength'], sample_data['transmittance'][i], sample_data['absorbance'][i]))
    header = ["frequency (cm-1)", "wavelength (um)", "transmittance (%)", "absorbance (AU)"]
    np.savetxt("%sby sample/%s/%s_mean-spectrum.csv" % (Out_dir, sample, sample), save_data.transpose(), header=", ".join(header), delimiter=', ')
        
print()
print()
print("total spectra processed:", len(ftir['ID']))
print("total samples processed:", len(np.unique(ftir['sample'])))

print()
print()
print("DONE")
