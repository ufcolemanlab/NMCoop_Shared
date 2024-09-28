#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:56:16 2023

@author: jcoleman
v09/25/2023

# conda update anaconda
# conda install spyder=5.4.3

Code to interface with Matlab mat file for Pizzi FWHM, perivessel data etc.
"""

import numpy as np
import scipy.io as sio
import mat73


def load_matfiledata(filename_mat):
#    dirname = '/#DATA/fam1/'
#    filename_vid = 'vid_mouse1_1_Day3_45s2.h264'
#    filename_mat = 'vid_mouse1_1_Day3_45s2.h264.mat'
    
    data_matfile = sio.loadmat(filename_mat, )
    # stimFrameArray = np.concatenate(data_matfile['stimFrames'])
    # stimFrameList = stimFrameArray.tolist()
    
    return data_matfile#, stimFrameList

#%%
#add TK here


dirname = '/Users/jcoleman/Documents/--LARGE DATA--/#Pizzi/TBI-sham leak analysis/results/process_v5-2ii/'
#dirname = '/Users/jcoleman/Documents/--LARGE DATA--/#Pizzi/TBI-sham leak analysis/process_v4_062723/process_v5-2ii/'
dataname ='MATdataForPython_12302023.mat'

#    filename_vid = 'vid_mouse1_1_Day3_45s2.h264'
#    datafile = 'vid_mouse1_1_Day3_45s2.h264.mat'

datafile = dirname + dataname
#dict_from_matfile = sio.loadmat(datafile)

data_dict = dict()
data_dict = mat73.loadmat(datafile)
#data_dict = sio.loadmat(datafile) # if not v7.3

"""
data_dict['AnalysisStruct']['Z1'].keys()
Out[17]: dict_keys(['m10_roi1_00001', 
                    'm11_roi1_00001', 
                    'm12_roi1_00001', 
                    'm13_roi1_00002', 
                    'm15_roi1_00001', 
                    'm16_roi1_00001', 
                    'm5_roi2_00003', 
                    'm6_roi1_00001', 
                    'm7_roi1_00001', 
                    'm9_roi1_00002'])

data_dict['AnalysisStruct']['Z1']['m16_roi1_00001'].keys()
Out[33]: dict_keys(['FeFv',
                    'XYZ3D',
                    'XYcoords',
                    'Zcoords',
                    'Zmicrons',
                    'foldFe',
                    'fwhm_Pv_vars',
                    'fwhm_flanking_intDen',
                    'fwhm_flanking_mean',
                    'maxFe',
                    'mean_FWHM_pix',
                    'mean_FWHM_ums',
                    'roiID',
                    'roiT_FWHM_pix',
                    'roiT_FWHM_ums',
                    'roiT_fwhm_flanking_tData', 
                    'roiT_profiles'])
"""
#dataZ1 =  data_dict['AnalysisStruct']['Z1']




#%% Iterate through ['XYZ3D'] in the the main data_dict

"""
Certainly! Here's a Python script that defines functions to plot the data and perform
cluster analysis. The script uses `matplotlib` for plotting and `sklearn` for clustering analysis. You can easily loop through multiple data sets using these functions.

Here are the steps the script will perform:
1. Parse the provided XYZ data.
2. Plot the data in a 3D scatter plot.
3. Perform K-Means clustering on the data.
4. Plot the clustered data.

```python
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

def parse_data(data):
    """
    Parse the given XYZ data into a numpy array.
    """
    data = np.array(data)
    return data

def plot_3d_scatter(data, title="3D Scatter Plot"):
    """
    Plot the data in a 3D scatter plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], -data[2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(title)
    plt.show()

def perform_kmeans_clustering(data, n_clusters=3):
    """
    Perform K-means clustering on the data and return the cluster labels.
    """
    data = data.T # Transpose for clustering (shape: [n_samples, n_features])
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_

def plot_clustered_data(data, labels, title="Clustered 3D Scatter Plot"):
    """
    Plot the clustered data in a 3D scatter plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[0], data[1], -data[2], c=labels, cmap='viridis', marker='o')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(title)
    plt.show()
    
def return_parsed_arrays(data, data_dict_key):
    """
    Main function to loop through data sets (eg data_dict), plot, and perform clustering.
    """
    data_arrays=dict()
    for key in data['AnalysisStruct']['Z1']:
        data_variable = data['AnalysisStruct']['Z1'][key][data_dict_key]
        parsed_data = parse_data(data_variable)
        #plot_3d_scatter(parsed_data, title=f"3D Scatter Plot for {key}")
        data_arrays[key] = parsed_data
    return data_arrays

def main(data_dict):
    """
    Main function to loop through data sets, plot, and perform clustering.
    """
    for key in data_dict['AnalysisStruct']['Z1']:
        data = data_dict['AnalysisStruct']['Z1'][key]['XYZ3D']
        parsed_data = parse_data(data)
        plot_3d_scatter(parsed_data, title=f"3D Scatter Plot for {key}")
        
        labels = perform_kmeans_clustering(parsed_data)
        plot_clustered_data(parsed_data, labels, title=f"Clustered 3D Plot for {key}")

# Dummy data dictionary for testing
data_dict_temp = {
    'AnalysisStruct': {
        'Z1': {
            'm16_roi1_00001': {
                'XYZ3D': [
                    [3.335000000000000000e+02, 4.905000000000000000e+02, 1.185000000000000000e+02, 2.115000000000000000e+02, 3.075000000000000000e+02, 1.060000000000000000e+02, 4.260000000000000000e+02, 4.770000000000000000e+02, 4.295000000000000000e+02, 4.140000000000000000e+02, 2.485000000000000000e+02, 3.580000000000000000e+02, 3.120000000000000000e+02, 2.735000000000000000e+02, 2.315000000000000000e+02, 4.260000000000000000e+02, 3.005000000000000000e+02, 4.500000000000000000e+02, 4.680000000000000000e+02, 3.160000000000000000e+02, 1.010000000000000000e+02, 3.445000000000000000e+02, 7.000000000000000000e+01, 2.330000000000000000e+02, 4.400000000000000000e+01, 5.200000000000000000e+01, 3.875000000000000000e+02, 2.075000000000000000e+02, 2.560000000000000000e+02, 2.505000000000000000e+02, 1.640000000000000000e+02, 3.480000000000000000e+02, 3.630000000000000000e+02, 3.360000000000000000e+02, 2.675000000000000000e+02, 4.280000000000000000e+02, 3.930000000000000000e+02, 1.595000000000000000e+02],
                    [3.450000000000000000e+01, 8.600000000000000000e+01, 2.735000000000000000e+02, 2.715000000000000000e+02, 1.110000000000000000e+02, 4.730000000000000000e+02, 4.185000000000000000e+02, 4.910000000000000000e+02, 3.510000000000000000e+02, 4.620000000000000000e+02, 3.965000000000000000e+02, 4.000000000000000000e+01, 2.945000000000000000e+02, 3.230000000000000000e+02, 3.515000000000000000e+02, 2.835000000000000000e+02, 1.980000000000000000e+02, 1.985000000000000000e+02, 3.160000000000000000e+02, 3.985000000000000000e+02, 8.900000000000000000e+01, 2.255000000000000000e+02, 3.995000000000000000e+02, 2.420000000000000000e+02, 2.810000000000000000e+02, 4.775000000000000000e+02, 2.690000000000000000e+02, 7.200000000000000000e+01, 2.705000000000000000e+02, 7.200000000000000000e+01, 3.530000000000000000e+02, 4.500000000000000000e+02, 2.525000000000000000e+02, 1.690000000000000000e+02, 1.800000000000000000e+02, 3.270000000000000000e+02, 2.570000000000000000e+02, 1.525000000000000000e+02],
                    [5.000000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00, 1.500000000000000000e+01, 1.500000000000000000e+01, 2.000000000000000000e+01, 2.000000000000000000e+01, 2.500000000000000000e+01, 2.500000000000000000e+01, 3.500000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.500000000000000000e+01, 5.000000000000000000e+01, 5.500000000000000000e+01, 6.500000000000000000e+01, 6.500000000000000000e+01, 7.500000000000000000e+01, 8.500000000000000000e+01, 9.500000000000000000e+01, 1.100000000000000000e+02, 1.100000000000000000e+02, 1.150000000000000000e+02, 1.150000000000000000e+02, 1.150000000000000000e+02, 1.200000000000000000e+02, 1.300000000000000000e+02, 1.350000000000000000e+02, 1.450000000000000000e+02, 1.500000000000000000e+02, 1.550000000000000000e+02, 1.600000000000000000e+02, 1.600000000000000000e+02, 1.650000000000000000e+02, 1.650000000000000000e+02, 1.850000000000000000e+02]
                ]
            }
        }
    }
}

# if __name__ == "__main__":
#     main(data_dict_temp)
#%%
    
"""
In this script:
- `parse_data(data)`: Parses the given XYZ data into a numpy array.
- `plot_3d_scatter(data, title)`: Plots the data in a 3D scatter plot.
- `perform_kmeans_clustering(data, n_clusters)`: Performs K-means clustering on the data and returns the cluster labels.
- `plot_clustered_data(data, labels, title)`: Plots the clustered data in a 3D scatter plot.
- `main(data_dict)`: Main function to loop through data sets, plot, and perform clustering.

You can test the script with the provided dummy data dictionary. This should plot the original scatter plot and the clustered scatter plot for each dataset available in the `data_dict`. Adjust the clustering parameters as needed for your analysis (e.g., `n_clusters`).

Make sure to install the necessary libraries if you haven't already:
```sh
pip install numpy matplotlib scikit-learn
"""

# #def return_parsed_arrays(data_dict)
# dict_variable = 'XYZ3D'
# #dict_variable = 'FeFv'
# data_3d_scatter=dict()
# for key in data_dict['AnalysisStruct']['Z1']:
#     data_array = data_dict['AnalysisStruct']['Z1'][key][dict_variable]
#     parsed_data = parse_data(data_array)
#     plot_3d_scatter(parsed_data, title=f"3D Scatter Plot for {key}")
    
#     data_3d_scatter[key] = parsed_data
    
#     #return data_3d_scatter
    
#     # labels = perform_kmeans_clustering(parsed_data)
#     # plot_clustered_data(parsed_data, labels, title=f"Clustered 3D Plot for {key}")
    
# # df = pd.DataFrame(data_array)
# # print("Original DataFrame:")
# # print(df)

dict_variable_names = ['FeFv',
                    'XYZ3D',
                    'XYcoords',
                    'Zcoords',
                    'Zmicrons',
                    'foldFe',
                    'fwhm_Pv_vars',
                    'fwhm_flanking_intDen',
                    'fwhm_flanking_mean',
                    'maxFe',
                    'mean_FWHM_pix',
                    'mean_FWHM_ums',
                    'roiID',
                    'roiT_FWHM_pix',
                    'roiT_FWHM_ums',
                    'roiT_fwhm_flanking_tData', 
                    'roiT_profiles']

data_xyz3d = return_parsed_arrays(data_dict, 'XYZ3D')
data_fefv = return_parsed_arrays(data_dict, 'FeFv')
data_fwhm_ums = return_parsed_arrays(data_dict, 'mean_FWHM_ums')

# def dict_to_csv(data_dict):
#import csv

# # Example dictionary
# data_dict_temp = {
#     'AnalysisStruct': {
#         'Z1': {
#             'm16_roi1_00001': {
#                 'XYZ3D': [
#                     [3.335000000000000000e+02, 4.905000000000000000e+02, 1.185000000000000000e+02, 2.115000000000000000e+02, 3.075000000000000000e+02, 1.060000000000000000e+02, 4.260000000000000000e+02, 4.770000000000000000e+02, 4.295000000000000000e+02, 4.140000000000000000e+02, 2.485000000000000000e+02, 3.580000000000000000e+02, 3.120000000000000000e+02, 2.735000000000000000e+02, 2.315000000000000000e+02, 4.260000000000000000e+02, 3.005000000000000000e+02, 4.500000000000000000e+02, 4.680000000000000000e+02, 3.160000000000000000e+02, 1.010000000000000000e+02, 3.445000000000000000e+02, 7.000000000000000000e+01, 2.330000000000000000e+02, 4.400000000000000000e+01, 5.200000000000000000e+01, 3.875000000000000000e+02, 2.075000000000000000e+02, 2.560000000000000000e+02, 2.505000000000000000e+02, 1.640000000000000000e+02, 3.480000000000000000e+02, 3.630000000000000000e+02, 3.360000000000000000e+02, 2.675000000000000000e+02, 4.280000000000000000e+02, 3.930000000000000000e+02, 1.595000000000000000e+02],
#                     [3.450000000000000000e+01, 8.600000000000000000e+01, 2.735000000000000000e+02, 2.715000000000000000e+02, 1.110000000000000000e+02, 4.730000000000000000e+02, 4.185000000000000000e+02, 4.910000000000000000e+02, 3.510000000000000000e+02, 4.620000000000000000e+02, 3.965000000000000000e+02, 4.000000000000000000e+01, 2.945000000000000000e+02, 3.230000000000000000e+02, 3.515000000000000000e+02, 2.835000000000000000e+02, 1.980000000000000000e+02, 1.985000000000000000e+02, 3.160000000000000000e+02, 3.985000000000000000e+02, 8.900000000000000000e+01, 2.255000000000000000e+02, 3.995000000000000000e+02, 2.420000000000000000e+02, 2.810000000000000000e+02, 4.775000000000000000e+02, 2.690000000000000000e+02, 7.200000000000000000e+01, 2.705000000000000000e+02, 7.200000000000000000e+01, 3.530000000000000000e+02, 4.500000000000000000e+02, 2.525000000000000000e+02, 1.690000000000000000e+02, 1.800000000000000000e+02, 3.270000000000000000e+02, 2.570000000000000000e+02, 1.525000000000000000e+02],
#                     [5.000000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00, 1.500000000000000000e+01, 1.500000000000000000e+01, 2.000000000000000000e+01, 2.000000000000000000e+01, 2.500000000000000000e+01, 2.500000000000000000e+01, 3.500000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.500000000000000000e+01, 5.000000000000000000e+01, 5.500000000000000000e+01, 6.500000000000000000e+01, 6.500000000000000000e+01, 7.500000000000000000e+01, 8.500000000000000000e+01, 9.500000000000000000e+01, 1.100000000000000000e+02, 1.100000000000000000e+02, 1.150000000000000000e+02, 1.150000000000000000e+02, 1.150000000000000000e+02, 1.200000000000000000e+02, 1.300000000000000000e+02, 1.350000000000000000e+02, 1.450000000000000000e+02, 1.500000000000000000e+02, 1.550000000000000000e+02, 1.600000000000000000e+02, 1.600000000000000000e+02, 1.650000000000000000e+02, 1.650000000000000000e+02, 1.850000000000000000e+02]
#                 ]
#             }
#         }
#     }
# }

#%% Flatten the dictionary to convert it to a CSV format
flat_data = {}
for z_key, z_value in data_dict_temp['AnalysisStruct'].items():
    for id_key, id_value in z_value.items():
        for param_key, param_value in id_value.items():
            if param_key not in flat_data:
                flat_data[param_key] = []
            for values in param_value:
                flat_data[param_key].extend(values)

# Get the maximum length of the values list to ensure all columns have the same length
max_len = max(len(v) for v in flat_data.values())

# Ensure all columns have the same length by filling shorter columns with None
for key, values in flat_data.items():
    if len(values) < max_len:
        flat_data[key].extend([None] * (max_len - len(values)))

# Write the data to a CSV file
csv_file = 'output.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(flat_data.keys())
    # Write the rows
    for row in zip(*flat_data.values()):
        writer.writerow(row)

print(f'Data has been written to {csv_file}')

# the ['XYZ3D'] concatenates into single vector of Xvals:Yvals:Zvals

#%% Attempt to pakcage the file into a CSV file - messy but may have all the data

import csv

def dict_to_csv(data_dict, csv_filename):
    """
    Convert a nested dictionary into a CSV file.
    
    Parameters:
    - data_dict: dict, the nested dictionary to be converted
    - csv_filename: str, the name of the output CSV file
    """
    # Flatten the dictionary to convert it to a CSV format
    flat_data = {}
    for z_key, z_value in data_dict['AnalysisStruct'].items():
        for id_key, id_value in z_value.items():
            for param_key, param_value in id_value.items():
                if param_key not in flat_data:
                    flat_data[param_key] = []
                for values in param_value:
                    flat_data[param_key].extend(values)

    # Get the maximum length of the values list to ensure all columns have the same length
    max_len = max(len(v) for v in flat_data.values())

    # Ensure all columns have the same length by filling shorter columns with None
    for key, values in flat_data.items():
        if len(values) < max_len:
            flat_data[key].extend([None] * (max_len - len(values)))

    # Write the data to a CSV file
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(flat_data.keys())
        # Write the rows
        for row in zip(*flat_data.values()):
            writer.writerow(row)

    print(f'Data has been written to {csv_filename}')

# Example usage
# data_dict_temp = {
#     'AnalysisStruct': {
#         'Z1': {
#             'm5_roi2_00003': {
#                 'XYZ3D': [
#                     [3.335000000000000000e+02, 4.905000000000000000e+02, 1.185000000000000000e+02, 2.115000000000000000e+02, 3.075000000000000000e+02, 1.060000000000000000e+02, 4.260000000000000000e+02, 4.770000000000000000e+02, 4.295000000000000000e+02, 4.140000000000000000e+02, 2.485000000000000000e+02, 3.580000000000000000e+02, 3.120000000000000000e+02, 2.735000000000000000e+02, 2.315000000000000000e+02, 4.260000000000000000e+02, 3.005000000000000000e+02, 4.500000000000000000e+02, 4.680000000000000000e+02, 3.160000000000000000e+02, 1.010000000000000000e+02, 3.445000000000000000e+02, 7.000000000000000000e+01, 2.330000000000000000e+02, 4.400000000000000000e+01, 5.200000000000000000e+01, 3.875000000000000000e+02, 2.075000000000000000e+02, 2.560000000000000000e+02, 2.505000000000000000e+02, 1.640000000000000000e+02, 3.480000000000000000e+02, 3.630000000000000000e+02, 3.360000000000000000e+02, 2.675000000000000000e+02, 4.280000000000000000e+02, 3.930000000000000000e+02, 1.595000000000000000e+02],
#                     [3.450000000000000000e+01, 8.600000000000000000e+01, 2.735000000000000000e+02, 2.715000000000000000e+02, 1.110000000000000000e+02, 4.730000000000000000e+02, 4.185000000000000000e+02, 4.910000000000000000e+02, 3.510000000000000000e+02, 4.620000000000000000e+02, 3.965000000000000000e+02, 4.000000000000000000e+01, 2.945000000000000000e+02, 3.230000000000000000e+02, 3.515000000000000000e+02, 2.835000000000000000e+02, 1.980000000000000000e+02, 1.985000000000000000e+02, 3.160000000000000000e+02, 3.985000000000000000e+02, 8.900000000000000000e+01, 2.255000000000000000e+02, 3.995000000000000000e+02, 2.420000000000000000e+02, 2.810000000000000000e+02, 4.775000000000000000e+02, 2.690000000000000000e+02, 7.200000000000000000e+01, 2.705000000000000000e+02, 7.200000000000000000e+01, 3.530000000000000000e+02, 4.500000000000000000e+02, 2.525000000000000000e+02, 1.690000000000000000e+02, 1.800000000000000000e+02, 3.270000000000000000e+02, 2.570000000000000000e+02, 1.525000000000000000e+02],
#                     [5.000000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00, 1.500000000000000000e+01, 1.500000000000000000e+01, 2.000000000000000000e+01, 2.000000000000000000e+01, 2.500000000000000000e+01, 2.500000000000000000e+01, 3.500000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.500000000000000000e+01, 5.000000000000000000e+01, 5.500000000000000000e+01, 6.500000000000000000e+01, 6.500000000000000000e+01, 7.500000000000000000e+01, 8.500000000000000000e+01, 9.500000000000000000e+01, 1.100000000000000000e+02, 1.100000000000000000e+02, 1.150000000000000000e+02, 1.150000000000000000e+02, 1.150000000000000000e+02, 1.200000000000000000e+02, 1.300000000000000000e+02, 1.350000000000000000e+02, 1.450000000000000000e+02, 1.500000000000000000e+02, 1.550000000000000000e+02, 1.600000000000000000e+02, 1.600000000000000000e+02, 1.650000000000000000e+02, 1.650000000000000000e+02, 1.850000000000000000e+02]
#                 ]
#             }
#             'm16_roi1_00001': {
#                 'XYZ3D': [
#                     [10.000000000000000e+02, 4.905000000000000000e+02, 1.185000000000000000e+02, 2.115000000000000000e+02, 3.075000000000000000e+02, 1.060000000000000000e+02, 4.260000000000000000e+02, 4.770000000000000000e+02, 4.295000000000000000e+02, 4.140000000000000000e+02, 2.485000000000000000e+02, 3.580000000000000000e+02, 3.120000000000000000e+02, 2.735000000000000000e+02, 2.315000000000000000e+02, 4.260000000000000000e+02, 3.005000000000000000e+02, 4.500000000000000000e+02, 4.680000000000000000e+02, 3.160000000000000000e+02, 1.010000000000000000e+02, 3.445000000000000000e+02, 7.000000000000000000e+01, 2.330000000000000000e+02, 4.400000000000000000e+01, 5.200000000000000000e+01, 3.875000000000000000e+02, 2.075000000000000000e+02, 2.560000000000000000e+02, 2.505000000000000000e+02, 1.640000000000000000e+02, 3.480000000000000000e+02, 3.630000000000000000e+02, 3.360000000000000000e+02, 2.675000000000000000e+02, 4.280000000000000000e+02, 3.930000000000000000e+02, 1.595000000000000000e+02],
#                     [10.0000000000000000e+01, 8.600000000000000000e+01, 2.735000000000000000e+02, 2.715000000000000000e+02, 1.110000000000000000e+02, 4.730000000000000000e+02, 4.185000000000000000e+02, 4.910000000000000000e+02, 3.510000000000000000e+02, 4.620000000000000000e+02, 3.965000000000000000e+02, 4.000000000000000000e+01, 2.945000000000000000e+02, 3.230000000000000000e+02, 3.515000000000000000e+02, 2.835000000000000000e+02, 1.980000000000000000e+02, 1.985000000000000000e+02, 3.160000000000000000e+02, 3.985000000000000000e+02, 8.900000000000000000e+01, 2.255000000000000000e+02, 3.995000000000000000e+02, 2.420000000000000000e+02, 2.810000000000000000e+02, 4.775000000000000000e+02, 2.690000000000000000e+02, 7.200000000000000000e+01, 2.705000000000000000e+02, 7.200000000000000000e+01, 3.530000000000000000e+02, 4.500000000000000000e+02, 2.525000000000000000e+02, 1.690000000000000000e+02, 1.800000000000000000e+02, 3.270000000000000000e+02, 2.570000000000000000e+02, 1.525000000000000000e+02],
#                     [10.00000000000000000e+00, 5.000000000000000000e+00, 5.000000000000000000e+00, 1.500000000000000000e+01, 1.500000000000000000e+01, 2.000000000000000000e+01, 2.000000000000000000e+01, 2.500000000000000000e+01, 2.500000000000000000e+01, 3.500000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.000000000000000000e+01, 4.500000000000000000e+01, 5.000000000000000000e+01, 5.500000000000000000e+01, 6.500000000000000000e+01, 6.500000000000000000e+01, 7.500000000000000000e+01, 8.500000000000000000e+01, 9.500000000000000000e+01, 1.100000000000000000e+02, 1.100000000000000000e+02, 1.150000000000000000e+02, 1.150000000000000000e+02, 1.150000000000000000e+02, 1.200000000000000000e+02, 1.300000000000000000e+02, 1.350000000000000000e+02, 1.450000000000000000e+02, 1.500000000000000000e+02, 1.550000000000000000e+02, 1.600000000000000000e+02, 1.600000000000000000e+02, 1.650000000000000000e+02, 1.650000000000000000e+02, 1.850000000000000000e+02]
#                 ]
#             }
#         }
#     }
# }

csv_filename = 'output2.csv'
dict_to_csv(data_dict, csv_filename)

#%% save dict

import pickle
#import numpy as np

# Example dictionary with a NumPy array
# data_dict = {
#     'array': np.array([1, 2, 3]),
#     'values': [10, 20, 30]
# }

# Save the dictionary to a file
with open('data_dict.pkl', 'wb') as file:
    pickle.dump(data_dict, file)
    
#%% OUTPUT FOR pca_code_chatgpt_1 (just need pickle file with full data_dict() [AnalysisStruct][Z1][...]])
import csv
import pickle
import numpy as np

def save_metrics_to_csv(data_dict, base_filename):
    # Extract metrics
    # 'XYcoords',
    # 'Zcoords',
    # 'Zmicrons',
    metrics = ['XYcoords', 'Zcoords', 'Zmicrons', 'XYZ3D', 'FeFv', 'mean_FWHM_ums']
    
    # Initialize dictionaries to hold the data for each metric
    metric_data = {metric: {} for metric in metrics}
    
    # Traverse the data dictionary to extract the relevant metrics
    for z_key, z_value in data_dict['AnalysisStruct'].items():
        for subject_id, metrics_dict in z_value.items():
            for metric in metrics:
                if metric in metrics_dict:
                    if subject_id not in metric_data[metric]:
                        metric_data[metric][subject_id] = []
                    metric_data[metric][subject_id].extend(metrics_dict[metric])
    
    # Save each metric to a separate CSV file
    for metric, data in metric_data.items():
        csv_filename = f"{base_filename}_{metric}.csv"
        
        # Find the maximum length of data to pad the columns if necessary
        max_len = max(len(values) for values in data.values())
        
        # Ensure all columns have the same length by padding with None
        for key, values in data.items():
            if len(values) < max_len:
                data[key].extend([None] * (max_len - len(values)))
        
        # Write to CSV
        with open(csv_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header
            writer.writerow(['Z'] + list(data.keys()))
            # Write the rows
            for i in range(max_len):
                row = [i] + [data[key][i] if i < len(data[key]) else None for key in data.keys()]
                writer.writerow(row)
        
        print(f"{metric} data has been written to {csv_filename}")

# Load your data dictionary
# with open('data_dict.pkl', 'rb') as file:
#     data_dict = pickle.load(file)

# Save metrics to CSV files
save_metrics_to_csv(data_dict, 'output')

#%%
import csv
import pickle
import numpy as np

def calculate_means_and_save_to_csv(data_dict, metrics, base_filename):
    """
    Calculate means for each subject and save to a CSV file.

    Parameters:
    - data_dict: dict, the nested dictionary with the data
    - metrics: list, the metrics to process
    - base_filename: str, the base filename for the output CSV
    
    Explanation:
Function Definition: The calculate_means_and_save_to_csv function takes the data dictionary, a list of metrics, and a base filename as input.
Metrics Initialization: It initializes a dictionary to hold the data for each metric.
Data Extraction: It traverses the input dictionary and extracts the relevant metrics for each subject ID.
Mean Calculation: It calculates the mean for each subject's values for the specified metrics.
Writing to CSV: It writes the calculated means to separate CSV files for each metric.
Usage:
Replace 'data_dict.pkl' with the path to your actual dictionary file.
Specify the metrics you want to process in the metrics list.
Run the script in your local Python environment to generate the CSV files with the calculated means.
This function will create CSV files named output_FeFv_means.csv and output_mean_FWHM_ums_means.csv, containing the mean values for each subject for the specified metrics.
    """
    # Initialize dictionaries to hold the data for each metric
    metric_data = {metric: {} for metric in metrics}
    
    # Traverse the data dictionary to extract the relevant metrics
    for z_key, z_value in data_dict['AnalysisStruct'].items():
        for subject_id, metrics_dict in z_value.items():
            for metric in metrics:
                if metric in metrics_dict:
                    if subject_id not in metric_data[metric]:
                        metric_data[metric][subject_id] = []
                    metric_data[metric][subject_id].extend(metrics_dict[metric])
    
    # Calculate means for each metric
    means_data = {metric: {} for metric in metrics}
    for metric, data in metric_data.items():
        for subject_id, values in data.items():
            if values:  # Ensure there are values to calculate the mean
                means_data[metric][subject_id] = np.nanmean(values)
    
    # Save the means to a CSV file
    for metric, data in means_data.items():
        csv_filename = f"{base_filename}_{metric}_means.csv"
        with open(csv_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header
            writer.writerow(['SubjectID', 'Mean'])
            # Write the rows
            for subject_id, mean_value in data.items():
                writer.writerow([subject_id, mean_value])
        
        print(f"Means for {metric} have been written to {csv_filename}")

# # Example usage
# with open('data_dict.pkl', 'rb') as file:
#     data_dict = pickle.load(file)

metrics = ['FeFv', 'mean_FWHM_ums']
calculate_means_and_save_to_csv(data_dict, metrics, 'output4')

           








