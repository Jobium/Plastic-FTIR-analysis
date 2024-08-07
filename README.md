# Plastic-FTIR-analysis
Python scripts for processing and analysis of Fourier Transform Infrared (FTIR) spectra of plastic debris

These scripts process spectra of plastic debris, generate summary figures, and train classification algorithms to assign spectra to known materials.
  FTIR_processing: takes raw FTIR spectra and processes them, including generating summary figures
  FTIR_classification: trains machine learning algorithms to classify processed FTIR spectra by material

Written by Dr Joseph Razzell Hollis on 2024-06-20. Details and assessment of the underlying methodology were published by Razzell Hollis et al. in the Journal of Hazardous Materials in 2024 (DOI: TBC). Please cite the methods paper if you adapt this code for your own analysis.

Any updates to this script will be made available online at www.github.com/Jobium/Plastic-FTIR-analysis/

Python code requires Python 3.7 (or higher) and the following packages: os, math, glob, datetime, numpy, pandas, matplotlib, lmfit, scipy, skimage, sklearn.
