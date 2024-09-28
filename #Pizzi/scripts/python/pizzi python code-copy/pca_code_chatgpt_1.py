#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:30:58 2024

@author: jcoleman

Explanation:
Load and Convert Data:
The load_and_convert_data function reads data from the provided CSV files.
It converts string representations of lists to actual NumPy arrays using ast.literal_eval.
It then explodes these arrays into separate rows.
Perform PCA:
The perform_pca function performs PCA on the merged data and retains the specified number of principal components (default is 3).
Plotting:
The script uses Seaborn to create a scatter plot of the first two principal components.
It annotates each point with its label (index from the DataFrame).
The plot includes axis labels with the percentage of variance explained by each principal component.
Usage:
Place the script in the same directory as your CSV files.
Update the csv_filenames list with the paths to your CSV files if they are named differently or located in different directories.
Run the script in your Python environment to generate the PCA plot.
This script will create a scatter plot of the first two principal components of the data, with each point labeled by its identifier. The amount of variance explained by each principal component will be printed in the console.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import ast


def load_and_convert_data(csv_filenames):
    """
    Load data from CSV files and convert string representations of lists into actual lists.
    
    Parameters:
    - csv_filenames: list of str, the filenames of the CSV files to be loaded
    
    Returns:
    - merged_df: DataFrame, the merged data
    """
    data_frames = []
    for csv_filename in csv_filenames:
        df = pd.read_csv(csv_filename, index_col=0)
        
        # Convert string representations of lists to actual lists
        for column in df.columns:
            df[column] = df[column].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
        
        # Explode the lists into separate rows
        df = df.apply(pd.Series.explode).reset_index()
        data_frames.append(df)
    
    merged_df = pd.concat(data_frames, axis=1)
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    return merged_df

def parse_data(data):
    """
    Parse the given XYZ data into a numpy array.
    """
    data = np.array(data)
    return data

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

# def perform_pca(data, n_components=3):
#     """
#     Perform PCA on the data.
    
#     Parameters:
#     - data: DataFrame, the data on which PCA is to be performed
#     - n_components: int, the number of principal components to keep
    
#     Returns:
#     - pca_df: DataFrame, the PCA-transformed data
#     - explained_variance: list, the amount of variance explained by each of the selected components
#     """
#     from sklearn.preprocessing import StandardScaler

#     # Standardize the data
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(data)
    
#     pca = PCA(n_components=n_components)
#     principal_components = pca.fit_transform(scaled_data)
    
#     pca_df = pd.DataFrame(data=principal_components, index=scaled_data.index,
#                           columns=[f'PC{i+1}' for i in range(n_components)])
    
#     explained_variance = pca.explained_variance_ratio_
#     return pca_df, explained_variance

def perform_pca(data, n_components=3):
    """
    Perform PCA on the data.
    
    Parameters:
    - data: DataFrame, the data on which PCA is to be performed
    - n_components: int, the number of principal components to keep
    
    Returns:
    - pca_df: DataFrame, the PCA-transformed data
    - explained_variance: list, the amount of variance explained by each of the selected components
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(data=principal_components, index=data.index,
                          columns=[f'PC{i+1}' for i in range(n_components)])
    
    explained_variance = pca.explained_variance_ratio_
    return pca_df, explained_variance


#Variables in main dict()
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
#Dict-arrays of data
data_xyz3d = return_parsed_arrays(data_dict, 'XYZ3D')
data_xy_2d = return_parsed_arrays(data_dict, 'XYcoords')
data_z_ums = return_parsed_arrays(data_dict, 'Zmicrons')
data_fefv = return_parsed_arrays(data_dict, 'FeFv')
data_fwhm_ums = return_parsed_arrays(data_dict, 'mean_FWHM_ums')

# File paths of the CSV files
# filepath = 

# CAN I USE XYZ (or add XY? for NN or similar graph theory?)
csv_filenames = ['output_Zmicrons.csv', 'output_FeFv.csv', 'output_mean_FWHM_ums.csv']

# Load and convert data
merged_alldata=pd.DataFrame()
#merged_data = load_and_convert_data(csv_filenames)
#df_all.dropna() # drop NANs before kmeans etc
merged_alldata = df_pca_rundata.dropna() # drop NANs
merged_data = merged_alldata.reset_index(drop=True)
#merged_data_groups = merged_data[['groupID']].copy()
merged_data_groups = merged_data['groupID'].tolist()
merged_data = merged_data.drop('groupID', axis=1)
"""
Zmicrons         40.000000
mean_FWHM_ums     4.286358
FeFv             35.639807
groupID           2.000000
"""
#%%
# Perform PCA
# SKREE PLOT TO DETERMINET n_components?
pca_results, variance_explained = perform_pca(merged_data, n_components=3)
#pca_results, variance_explained = perform_pca(merged_data, n_components=4)


# Plot the first two principal components
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', data=pca_results, s=100, alpha=0.7)

# Add labels for each point
for i in range(pca_results.shape[0]):
    if merged_data_groups[i]==1: #'GroupA'
        plt.text(pca_results.PC1[i], pca_results.PC2[i], pca_results.index[i], 
                 horizontalalignment='center', verticalalignment='center', color='black')
    else:
        plt.text(pca_results.PC1[i], pca_results.PC2[i], pca_results.index[i], 
                 horizontalalignment='center', verticalalignment='center', color='red')

# Title and labels
plt.title('PCA of Vessel Data (1 region)')
plt.xlabel(f'PC1 ({variance_explained[0]:.2%} variance)')
plt.ylabel(f'PC2 ({variance_explained[1]:.2%} variance)')
plt.grid(True)

# Show plot
plt.show()

# Print the amount of variance explained by each component
for i, variance in enumerate(variance_explained):
    print(f'PC{i+1}: {variance:.2%} variance explained')
    
#%% &^&$(&*^*&*&)&**&^
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#import numpy as np

def plot_pc1_vs_original(original_data, pca_results, variable_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('PC1 vs Original Variables', fontsize=16)

    for i, var_name in enumerate(variable_names):
        x = original_data[var_name]
        y = pca_results['PC1']
        
        # Calculate Pearson correlation coefficient and p-value
        rho, p_value = stats.pearsonr(x, y)
        
        # Create scatter plot
        sns.scatterplot(x=x, y=y, ax=axes[i])
        
        # Set labels and title
        axes[i].set_xlabel(var_name)
        axes[i].set_ylabel('PC1')
        axes[i].set_title(f'PC1 vs {var_name}')
        
        # Add correlation coefficient and p-value to plot
        axes[i].text(0.05, 0.95, f'ρ = {rho:.3f}\np = {p_value:.3e}', 
                     transform=axes[i].transAxes, 
                     verticalalignment='top')
        
        # Add regression line
        sns.regplot(x=x, y=y, ax=axes[i], scatter=False, color='red')

    plt.tight_layout()
    plt.show()

# Assuming you have your original data in a DataFrame called 'merged_data'
# and your PCA results in a DataFrame called 'pca_results'
variable_names = ['Zmicrons', 'mean_FWHM_ums', 'FeFv']
# Zmicrons         40.000000
# mean_FWHM_ums     4.286358
# FeFv             35.639807

plot_pc1_vs_original(merged_data, pca_results, variable_names)
