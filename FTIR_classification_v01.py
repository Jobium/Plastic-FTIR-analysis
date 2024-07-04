"""
====================================================

This script applies the following machine learning classification algorithms to FTIR spectra of plastic debris.
    1) Principal Component Analysis (PCA)
    2) Linear Discriminant Analysis (LDA)
    3) K-nearest Neighbours         (KNN)
    4) Support Vector Machine       (SVM)

This script is designed to accept files outputted by the FTIR_processing script.

Reference plastic spectra are from the FLOPP and FLOPP-e databases published by de Frond et al. (2021)

====================================================
"""

import os
import math
import glob
import datetime
import numpy as np
import pandas as pd
import lmfit as lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# ==================================================
# user defined variables

# directory of folder containing spectra files to be imported
Data_dir = './output/by sample/*/'

# directory of sample info database containing list of sample names and their material assignments
Database_dir = './data/Database.csv'

# directory of folder where figures will be saved
Fig_dir = './figures/'

# directory of folder where processed data will be saved
Out_dir = './output/'

# specify whether to process spectra in 'Absorbance' or 'Transmittance' mode
Mode = 'Absorbance'

# import reference spectra from the FLOPP/FLOPP-e databases?
Import_FLOPPe = True

# directory of reference spectra database
FLOPPe_dir = './data/FLOPP-FLOPPe_*_database.csv'

# specify which ML models to use
Test_Models = ['KNN', 'LDA', 'PCA-SVM', 'PCA-KNN', 'PCA-LDA', 'FLOPP-Euclid']

# define the spectral region for training models
x_start, x_end = (400, 1800)

# simplify material labels? (see lines XXX)
Simplify_labels = True

# only use these material labels for classification (empty list or list of strings)
Label_list = []

# list expected materials and their plotting colours
Materials = ['notplastic', 'pumice', 'unassigned', 'PE', 'LDPE', 'MDPE', 'HDPE', 'PP', 'PPE', 'PS', 'PLA', 'ABS', 'EAA', 'PA']
Material_colors =  ['k', 'skyblue', 'chartreuse', 'r', 'g', 'gold', 'b', 'm', 'y', 'tab:gray', 'tab:orange', 'tab:brown', 'tab:pink', 'seagreen']

# generic lists of colours and markers for plotting
Color_list =  ['r', 'b', 'm', 'y', 'tab:gray', 'c', 'tab:orange', 'tab:brown', 'tab:pink', 'tan']
Marker_list = ['o', 's', 'v', '^', 'D', '*', 'p']

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

# ==================================================
# function for running an ML model on predefined training and testing data

def run_model(model, training_data, training_classes, test_data, test_classes, pca_components=6, lda_components=3, knn_neighbours=3, debug=False):
    if debug == True:
        print("model:", model)
        print("    training dataset:", np.shape(training_data))
        print("    training classes:", np.unique(training_classes))
        print("    test dataset:", np.shape(test_data))
        print("    test classes:", np.unique(test_classes))
    labels = np.unique(np.concatenate((training_classes, test_classes)))
    # run appropriate machine learning model
    if 'SVM' in model:
        # support vector machine of PCA coordinates
        svm = SVC()
        # train SVM on training dataset
        trained_svm = svm.fit(training_data, training_classes)
        # predict classes for validation coords
        prediction = trained_svm.predict(test_data)
    elif 'LDA' in model:
        # linear discriminant analysis of spectra
        # determine how many LDA components to calculate
        if len(np.unique(training_classes))-1 < lda_components:
            lda_components = len(np.unique(training_classes))-1
        lda = LinearDiscriminantAnalysis(n_components=lda_components)
        # train lda fit on training dataset
        trained_lda = lda.fit(training_data, training_classes)
        # predict classes for validation dataset
        prediction = trained_lda.predict(test_data)
    elif 'KNN' in model:
        # K-nearest neighbours of spectra
        knn = KNeighborsClassifier(n_neighbors=knn_neighbours)
        # train KNN on training dataset
        trained_knn = knn.fit(training_data, training_classes)
        # predict classes for validation dataset
        prediction = trained_knn.predict(test_data)
    elif 'Euclid' in model:
        # nearest match in training data, by Euclidean distance (KNN, k=1)
        euclid = KNeighborsClassifier(n_neighbors=1)
        # train KNN on training dataset
        trained_euclid = euclid.fit(training_data, training_classes)
        # predict classes for validation dataset
        prediction = trained_euclid.predict(test_data)
    results = {
        'labels': labels,
        'matrix': confusion_matrix(test_classes, prediction, labels=labels, normalize='true'),
        'accuracy': accuracy_score(test_classes, prediction),
        'precision': precision_score(test_classes, prediction, average='weighted'),
        'recall': recall_score(test_classes, prediction, average='weighted'),
        'F1': f1_score(test_classes, prediction, average='weighted')
    }
    if debug == True:
        print("    confusion matrix shape:", np.shape(results['matrix']))
        print("    test results:")
        print("        accuracy =  %0.2f" % results['accuracy'])
        print("        precision = %0.2f" % results['precision'])
        print("        recall =    %0.2f" % results['recall'])
        print("        F1 score =  %0.2f" % results['F1'])
    return prediction, results

"""
# ==================================================
# import FTIR spectra from files
# ==================================================
"""

# ==================================================
# find FTIR spectra files in Data_dir folder

print()
print("searching for FTIR spectrum files...")

files = sorted(glob.glob('%s*_mean-spectrum.csv' % Data_dir))
print()
print('%s*_mean-spectrum.csv' % Data_dir)
print("files found:", len(files))
    
# ==================================================
# import sample-average FTIR spectra into sample_data datadict

x_start, x_end = (400, 1800)

# generate standardised x values for interpolation
standard_x = np.linspace(x_start, x_end, 2*int(x_end-x_start))

print()
print("x range of %i points between %0.f and %0.f cm-1" % (np.size(standard_x), np.amin(standard_x), np.amax(standard_x)))

sample_data = {
    'sample': [],
    'material': [],
    'frequency': standard_x.copy(),
    'wavelength': 10000./standard_x.copy(),
    'transmittance': [],
    'absorbance': []
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
                sample = filename.split("_")[0]
                print("    sample name: ", sample)
                    
                # import spectrum
                spec = np.genfromtxt(file, delimiter=',').transpose()
                absorb = transmittance_to_absorbance(spec[2])
                print("    spec file shape:", np.shape(spec))
                sliced = np.where(absorb != np.inf)
                print("        zeros/infinities trimmed:", np.count_nonzero(absorb == np.inf), "at", spec[0][absorb == np.inf])
                
                # interpolate based on standard_x
                y_interp = np.interp(standard_x, spec[0][sliced], spec[2][sliced])
                absorb_interp = transmittance_to_absorbance(y_interp)
                
                # add data to sample_data dict
                print("    adding data to array")
                sample_data['sample'].append(sample)
                sample_data['transmittance'].append(y_interp)
                sample_data['absorbance'].append(absorb_interp)
                
                # check if this sample has material assignment in sample database:
                assignment = get_chemicalID(sample_database, sample, debug=False)
                if assignment != '':
                    sample_data['material'].append(assignment)
                    print("    material assignment:", assignment)
                else:
                    sample_data['material'].append("unassigned")
                    print("    sample does not have an assigned material")
                print("    success!")
                break
            except Exception as e:
                print("        something went wrong! Exception:", e)
                break
                
print()

# clean up arrays
for key in sample_data.keys():
    sample_data[key] = np.asarray(sample_data[key])
    if len(sample_data[key]) != len(sample_data['sample']):
        print("WARNING: %s array length does not match ID array length!" % key)

print()
print("files imported:", len(sample_data['sample']))

for i in range(0, len(sample_data['sample'])):
    print("%4i %-30s %-5s" % (i, sample_data['sample'][i], sample_data['material'][i]))

# trim iterable lists down to those with sample_data only
Sample_IDs = [sample for sample in Sample_IDs if sample in sample_data['sample']]
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
    spectra = [np.size(sample_data['absorbance'][i], axis=0) for i in range(0, len(sample_data['sample'])) if sample_data['sample'][i] in samples]
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
# import FLOPP/FLOPP-e databases
# ==================================================
"""

# this relies on an integrated database containing all reference spectra.
#   only spectra with material labels that occur in the 'Materials' list will be imported

floppe = {
    'sample': [],
    'material': [],
    'frequency': [],
    'wavelength': [],
    'transmittance': [],
    'absorbance': []
}

floppe_dir = glob.glob(FLOPPe_dir)

if Import_FLOPPe == True and len(floppe_dir) > 0:
    print()
    print("importing FLOPP/FLOPP-e database...")
    print(floppe_dir)
    
    floppe_df = pd.read_csv(floppe_dir[-1])
    print("    FLOPP database rows:", len(floppe_df.index.values))
    print("    FLOPP database columns:",  floppe_df.columns)
    floppe_materials = np.asarray([mat.split(".")[0] for mat in floppe_df.columns.values[1:]])
    print("    materials in db:", np.unique(floppe_materials))
    
    print("    converting database to spectral array...")
    mode = floppe_dir[-1].split("_")[-2]
    print("        spectrum mode:", mode)
    freq = np.asarray(floppe_df['# Frequency'].values)
    print("        frequency range: %0.f - %0.f cm-1" % (np.amin(freq), np.amax(freq)))
    print(freq)
    temp = []
    for col_label in floppe_df.columns.values[1:]:
        temp.append(floppe_df[col_label])
    temp = np.asarray(temp)
    
    # standardise to absorbance for interpolation
    print(np.shape(temp))
    if mode.lower() == 'absorbance':
        absorbance = temp
    else:
        absorbance = transmittance_to_absorbance(temp)
        
    # take only reference spectra that are in Materials list
    result = [i for i in range(len(floppe_materials)) if floppe_materials[i] in Materials]
    print("        spectra in Materials list:", len(result))
        
    # interpolate y data based on standard_x
    absorbance_interp = []
    for i in result:
        absorbance_interp.append(np.interp(standard_x, freq, absorbance[i]))
    absorbance_interp = np.asarray(absorbance_interp)
    transmittance_interp = absorbance_to_transmittance(absorbance_interp)
    
    print("        spec arrays after interpolation:", np.shape(freq), np.shape(transmittance_interp), np.shape(absorbance_interp))
    floppe['sample'] = result
    floppe['material'] = floppe_materials[result]
    floppe['frequency'] = freq
    floppe['wavelength'] = 10000. / freq
    floppe['absorbance'] = absorbance_interp
    floppe['transmittance'] = transmittance_interp
    print("    successfully imported FLOPP/FLOPP-e database containing %s spectra!" % len(result))
    
print()
for key in floppe.keys():
    floppe[key] = np.asarray(floppe[key])
    if len(floppe[key]) != len(floppe['material']):
        print(f"{key} length does not match ID length!")

"""
# ==================================================
# train and test classification models
# ==================================================
"""

normalise = True        # normalise spectra before analysis
SNV = False             # take the standard normal variate of spectra before analysis
first_deriv = True      # take the first derivative of spectra before analysis

test_fraction = 0.2     # fraction of spectra to assign to testing group (always min 1 per material)

n_folds = 10            # number of folds for cross-validation
seed = 626              # set to 626 for repeated tests

print()
print("Training machine learning models on region %0.f to %0.f cm-1" % (x_start, x_end))

# ==================================================
# prepare arrays for model results

models = {}
    
for model in Test_Models:
    models[model] = {'accuracy': np.zeros(n_folds), 'precision': np.zeros(n_folds), 'recall': np.zeros(n_folds), 'F1': np.zeros(n_folds), 'matrix': [[] for i in range(n_folds)]}

print()
print("testing the following classification models:", models.keys())

# ==================================================
# prepare material label lists for classification

samples = np.asarray(sample_data['sample'])
materials = np.asarray(sample_data['material'])

if len(Label_list) > 0:
    # only use materials labels in user-defined Label_list
    mat_labels = np.asarray(Label_list)
elif Simplify_labels == True:
    # simplify material labels into groups
    simple_groups = {
        'notplastic': ['not plastic', 'pumice'],
        'PE': ['PE', 'LDPE', 'LLDPE', 'MDPE', 'HDPE']
    }
    for group, mats in simple_groups.items():
        result = [i for i in range(len(materials)) if materials[i] in mats]
        if len(result) > 0:
            materials[result] = group
    mat_labels = np.unique(materials)
else:
    mat_labels = np.unique(materials)
        
print()
print("classifying %s materials:" % len(mat_labels), mat_labels)
    
# filter out any spectra that do not match mat_labels
all_indices = np.asarray([i for i in range(len(materials)) if materials[i] in mat_labels])
print("%s spectra have valid labels" % len(all_indices))
print("%s material labels in dataset:" % len(np.unique(materials[all_indices])), np.unique(materials[all_indices]))
        
# set up training and test datasets
test_indices = []
np.random.seed(seed)
for material in mat_labels:
        result = all_indices[np.ravel(np.where(materials[all_indices] == material))]
        print(material, len(result), round(test_fraction*len(result)))
        print(np.count_nonzero(materials[all_indices] == material))
        ### print("    ", materials[result])
        if test_fraction*len(result) > 1:
            # return fraction of indices
            test_indices.append(np.random.choice(result, round(test_fraction*len(result)), replace=False))
        else:
            # return at least one index
            test_indices.append(np.random.choice(result, 1))
        print("    test indices:", len(test_indices[-1]))
        print("    test indices not in all_indices:", [i for i in test_indices[-1] if i not in all_indices])
        indx, counts = np.unique(test_indices[-1], return_counts=True)
        print("    repeated indices:", ", ".join(["%s (%s)" % (indx[i], counts[i]) for i in range(len(indx)) if counts[i] > 1]))
test_indices = np.concatenate(test_indices)
train_indices = np.setdiff1d(all_indices, test_indices)
print("%s spectra total" % len(all_indices))
print("%s spectra in test group" % len(test_indices), np.unique(materials[test_indices]))
print("%s spectra in train/validate group" % len(train_indices), np.unique(materials[train_indices]))
print("    %s spectra in both groups!" % len(np.intersect1d(test_indices, train_indices)))
print("%s total spectra" % (len(test_indices) + len(train_indices)), np.unique(materials[all_indices]))
other_indices = np.ravel(np.where(materials == 'unassigned'))
    
print()
print("index checks:")
print("indices in group that are not in all_indices:")
print("    testing:", len([i for i in test_indices if i not in all_indices]))
print("    training:", len([i for i in train_indices if i not in all_indices]))
print("indices in all_indices that are not in each group:")
print("    testing:", len([i for i in all_indices if i not in test_indices]))
print("    training:", len([i for i in all_indices if i not in train_indices]))
    
# ==================================================
# prepare spectral data

x_data = standard_x.copy()
y_data = []

for i in range(0, len(sample_data[Mode.lower()])):
    y = np.interp(x_data, sample_data['frequency'], sample_data[Mode.lower()][i])
    if SNV == True:
        y = (y - np.mean(y)) / np.std(y)
    elif normalise == True:
        y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))
    if first_deriv == True:
        y = savgol_filter(y, 25, polyorder = 5, deriv=1)
    y_data.append(y)
y_data = np.asarray(y_data)
print("y array shape:", np.shape(y_data))
     
# ==================================================
# do regular PCA for data visualisation
        
model_data = pd.DataFrame(y_data, columns=x_data, index=sample_data['sample'])
    
# generate PCA coordinates
pca = PCA(n_components=6).fit(model_data)
model_data_pca = pca.transform(model_data)
print("PCA coord array:", np.shape(model_data_pca))
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    print("    component %d: %0.3f" % (i+1, pca.explained_variance_ratio_[i]))
    
# ==================================================
# do K-means clustering of PCA coordinates

n_clusters = len(mat_labels)
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(model_data_pca)
    
print("clustering with K=%d clusters" % n_clusters)
for cluster in range(0, n_clusters):
    print("    spectra in cluster %2d: %0.f, centered at: (%0.3f, %0.3f)" % (cluster+1, np.count_nonzero(kmeans.labels_ == cluster), kmeans.cluster_centers_[cluster,0], kmeans.cluster_centers_[cluster,1]))
    for material in np.unique(materials):
        sort = np.logical_and(materials == material, kmeans.labels_ == cluster)
        if np.count_nonzero(sort) > 0:
            print("        %3.f%% %s (%0.f/%0.f %s spectra)" % (100*np.count_nonzero(sort)/np.count_nonzero(kmeans.labels_ == cluster), material, np.count_nonzero(sort), np.count_nonzero(materials == material), material))
    
n_components_plot = 3
    
# ==================================================
# generate PCA/K-means clustering figure for all data

plt.figure(figsize=(15,5))
ax1 = plt.subplot2grid((n_components_plot,3), (0,0), rowspan=n_components_plot)
ax2 = plt.subplot2grid((n_components_plot,3), (0,1), rowspan=n_components_plot, sharex=ax1, sharey=ax1)
ax1.set_title('PCA of %s-%s cm$^{-1}$' % (x_start, x_end))
ax1.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca.explained_variance_ratio_[0]))
ax1.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca.explained_variance_ratio_[1]))
ax2.set_title('K-Means Clusters')
# ax1: plot PCA coordinates by material
for material in np.unique(materials):
    color = Material_colors[Materials.index(material) % len(Material_colors)]
    result = np.ravel(np.where(materials == material))
    ax1.scatter(model_data_pca[result,0], model_data_pca[result,1], c=color, linewidth=0., alpha=1., label="%s (%d)" % (material, len(result)))
# ax2: plot K-means clusters in PCA space
for cluster in range(0, n_clusters):
    color = Material_colors[-(cluster+1)]
    result = np.ravel(np.where(kmeans.labels_ == cluster))
    ax2.scatter(model_data_pca[result,0], model_data_pca[result,1], c=color, linewidth=0., alpha=1., label="cl. %d (%d)" % (cluster+1, len(result)))
# ax3: plot loading spectra for each PC
for i in range(n_components_plot):
    ax = plt.subplot2grid((n_components_plot,3), (i,2))
    if i == 0:
        ax.set_title("PCA Loading Spectra")
    if i < n_components_plot-1:
        ax.set_xticks([])
    else:
        ax.set_xlabel("Frequency (cm$^{-1}$)")
    ax.set_ylabel("PC%d Coefficient" % (i+1))
    ax.set_xlim(x_start, x_end)
    comp = pca.components_[i] * pca.explained_variance_[i]  # scale component by its variance explanation power
    if first_deriv == True:
        ax.plot(x_data, np.cumsum(comp), label="PC %s" % (i+1))
    else:
        ax.plot(x_data, comp, label="PC %s" % (i+1))
ax1.legend()
ax1.grid(zorder=3)
ax2.legend()
ax2.grid(zorder=3)
plt.tight_layout()
plt.savefig("%sPCA.png" % (Fig_dir), dpi=300)
plt.savefig("%sPCA.svg" % (Fig_dir), dpi=300)
plt.show()
    
# ==================================================
# prepare FLOPP spectral data

floppe_temp = []
for i in range(0, len(floppe[Mode.lower()])):
    y = floppe[Mode.lower()][i]
    if SNV == True:
        y = (y - np.mean(y)) / np.std(y)
    elif normalise == True:
        y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))
    if first_deriv == True:
        y = savgol_filter(y, 25, polyorder = 5, deriv=1)
    floppe_temp.append(y)
floppe_temp = np.asarray(floppe_temp)
print(np.shape(x_data), np.shape(floppe_temp))
floppe_data = pd.DataFrame(floppe_temp, columns=x_data)
    
# ==================================================
# run K-Fold cross-validation

kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
for i, (train_index, validate_index) in enumerate(kf.split(model_data.iloc[train_indices])):
        print(f"Fold {i}: training on {len(train_index)} spectra, validating on {len(validate_index)}")
        #### print("    true classes:", truth)
        for model in sorted(models.keys()):
            print(f"    {model}:")
            # select training and validation datasets
            if 'FLOPP' in model:
                training_data = floppe_data
                training_classes = floppe['material']
                # trim down FTIR spectra if range exceeds FLOPP range
                check = np.where((np.amin(floppe_data.columns.values) <= model_data.columns.values) & (model_data.columns.values <= np.amax(floppe_data.columns.values)))
                validation_data = model_data[model_data.columns.values[check]].iloc[train_indices].iloc[validate_index]
                validation_classes = materials[train_indices][validate_index]
            elif 'PCA' in model:
                training_data = model_data_pca[train_indices][train_index]
                training_classes = materials[train_indices][train_index]
                validation_data = model_data_pca[train_indices][validate_index]
                validation_classes = materials[train_indices][validate_index]
            else:
                training_data = model_data.iloc[train_indices].iloc[train_index]
                training_classes = materials[train_indices][train_index]
                validation_data = model_data.iloc[train_indices].iloc[validate_index]
                validation_classes = materials[train_indices][validate_index]
            
            prediction, results = run_model(model, training_data, training_classes, validation_data, validation_classes, debug=False)
                
            models[model]['matrix'][i] = results['matrix']
            models[model]['accuracy'][i] = results['accuracy']
            models[model]['precision'][i] = results['precision']
            models[model]['recall'][i] = results['recall']
            models[model]['F1'][i] = results['F1']
    
perf_metrics = ['accuracy', 'precision', 'recall', 'F1']
    
# ==================================================
# report average performance metrics for each model

for model in sorted(models.keys()):
    print()
    print(f"{model} average performance:")
    for perf_metric in perf_metrics:
        print(f"    {perf_metric:11s}:  {np.mean(models[model][perf_metric]):0.2f} ({np.amin(models[model][perf_metric]):0.2f} - {np.amax(models[model][perf_metric]):0.2f})")
    
# ==================================================
# report best performing model, by metric

for perf_metric in perf_metrics:
    perf_results = []
    print()
    for model in sorted(models.keys()):
        print(f"{model:8s} {perf_metric}:  {np.mean(models[model][perf_metric]):0.2f} +/- {np.std(models[model][perf_metric]):0.2f}")
        perf_results.append(np.mean(models[model][perf_metric]))
    print(f"best model, by {perf_metric}: {sorted(models.keys())[np.argmax(perf_results)]}")
        
print()
print()

# ==================================================
# generate summary figure for all models

fig, axs = plt.subplots(len(models.keys()), 3, figsize=(15,5*len(models.keys())))
for i, model in enumerate(sorted(models.keys())):
        print()
        print(f"evaluating {model} on test dataset ({len(test_indices)} spectra)")
        ax0 = axs[i,0]
        ax1 = axs[i,1]
        ax2 = axs[i,2]
        ax0.set_title("PCA of %0.f-%0.f cm$^{-1}$" % (x_start, x_end))
        ax1.set_title(f"{model} Predictions")
        ax2.set_title(f"{model} Confusion Matrix")
        ax0.set_xlabel("PC1")
        ax0.set_ylabel("PC2")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        
        # plot PCA
        for material in mat_labels:
            result = np.ravel(np.where(materials == material))
            color = Material_colors[Materials.index(material) % len(Material_colors)]
            ax0.scatter(model_data_pca[result,0], model_data_pca[result,1], c=color, linewidth=0., alpha=1., label="%s (%s)" % (material, len(result)), zorder=3)
        for material in np.unique(materials[other_indices]):
            result = np.ravel(np.where(materials == material))
            color = Material_colors[Materials.index(material) % len(Material_colors)]
            ax0.scatter(model_data_pca[result,0], model_data_pca[result,1], c=color, linewidth=0., alpha=1., label="%s (%s)" % (material, len(result)), zorder=3)
        
        # select training and test datasets
        if 'FLOPP' in model:
            training_data = floppe_data
            training_classes = floppe['material']
            # trim down FTIR spectra if range exceeds FLOPP range
            check = np.where((np.amin(floppe_data.columns.values) <= model_data.columns.values) & (model_data.columns.values <= np.amax(floppe_data.columns.values)))
            test_data = model_data[model_data.columns.values[check]].iloc[test_indices]
            test_classes = materials[test_indices]
        elif 'PCA' in model:
            training_data = model_data_pca[train_indices]
            training_classes = materials[train_indices]
            test_data = model_data_pca[test_indices]
            test_classes = materials[test_indices]
        else:
            training_data = model_data.iloc[train_indices]
            training_classes = materials[train_indices]
            test_data = model_data.iloc[test_indices]
            test_classes = materials[test_indices]
        print(f"    materials in training set: {np.unique(training_classes)}")
        print(f"    materials in test set: {np.unique(test_classes)}")
            
        # train model and test on test dataset
        prediction, results = run_model(model, training_data, training_classes, test_data, test_classes, debug=True)
        
        ### print(f"        predictions:", prediction)
        print(f"{model} results:")
        print(f"    matrix shape:", np.shape(results['matrix']))
        print(f"    accuracy:  {results['accuracy']:0.2f}")
        print(f"    precision: {results['precision']:0.2f}")
        print(f"    recall:    {results['recall']:0.2f}")
        print(f"    F1:        {results['F1']:0.2f}")
        
        # plot confusion matrix and report counts
        ax2.set_xticks(range(len(results['labels'])))
        ax2.set_xticklabels(results['labels'])
        ax2.set_xlabel("Predicted")
        ax2.set_yticks(range(len(results['labels'])))
        ax2.set_yticklabels(results['labels'])
        ax2.set_ylabel("True")
        ax2.imshow(results['matrix'])
        for xi in range(len(results['labels'])):
            count  = np.count_nonzero(materials[test_indices] == results['labels'][xi])
            for yi in range(len(results['labels'])):
                ax2.text(yi, xi, "%0.f" % (results['matrix'][xi,yi] * count), color='w', ha='center', va='center')
        
        # generate PCA coordinates for plotting
        pca = PCA(n_components=6)
        trained_pca = pca.fit(model_data)
        trained_coords = trained_pca.transform(model_data.iloc[train_indices])
        test_coords = trained_pca.transform(model_data.iloc[test_indices])
        
        # plot datasets by material
        for material in mat_labels:
            color = Material_colors[Materials.index(material) % len(Material_colors)]
            # plot training data
            sort = materials[train_indices] == material
            ax1.scatter(trained_coords[sort,0], trained_coords[sort,1], c=color, linewidth=0., alpha=0.2)
            # plot test data
            sort = materials[test_indices] == material
            correct_prediction = prediction[sort] == materials[test_indices][sort]
            ax1.scatter(test_coords[sort,0][correct_prediction], test_coords[sort,1][correct_prediction], c=color)
            ax1.scatter(test_coords[sort,0][~correct_prediction], test_coords[sort,1][~correct_prediction], c='w', edgecolors=color)
            
        # add grid to PCA space
        ax0.grid(zorder=1)
        ax1.grid(zorder=1)
        # add legends
        ax0.legend()
# finish figure
fig.tight_layout()
plt.savefig("%sclassification_test.png" % (Fig_dir), dpi=300)
plt.savefig("%sclassification_test.svg" % (Fig_dir), dpi=300)
plt.show()

print()
print()
print("total samples processed:", len(sample_data['sample']))

# ==================================================
# save performance metrics from model training

for model in sorted(models.keys()):
    save_data = pd.DataFrame.from_dict(models[model])
    save_data.to_csv("%s%s_cross-validation.csv" % (Out_dir, model))
    
for metric in perf_metrics:
    save_data = []
    for model in sorted(models.keys()):
        save_data.append(models[model][metric])
    save_data = np.asarray(save_data)
    np.savetxt("%scross-validation_%s.csv" % (Out_dir, metric), np.asarray(save_data).transpose(), delimiter=", ", header=", ".join(sorted(models.keys())))

# ==================================================
# save assignments from model testing

datasets = np.asarray(['n/a' for i in range(len(sample_data['sample']))])
datasets[train_indices] = 'training'
datasets[test_indices] = 'testing'

save_data = {
    'sample': sample_data['sample'],
    'material': sample_data['material'],
    'dataset': datasets
}

save_data = pd.DataFrame.from_dict(save_data)
save_data.to_csv("%sFTIR_classification_results.csv" % Out_dir)

print()
print()
print("DONE")
