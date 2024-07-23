#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:16:47 2024

@author: jcoleman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Step 1: Load and process the data
def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    df_melted = pd.melt(data, id_vars=[], var_name='Measurement', value_name='Value')
    df_melted['AreaID'] = df_melted['Measurement'].str.extract(r'(\d+)')
    df_melted['MeasurementType'] = df_melted['Measurement'].str.extract(r'([a-zA-Z]+)')
    df_pivot = df_melted.pivot_table(index=['AreaID'], columns='MeasurementType', values='Value', aggfunc='first').reset_index()
    numeric_columns = ['Area', 'Mean', 'X', 'Y', 'Circ', 'IntDen', 'RawIntDen', 'AR', 'Round', 'Solidity']
    df_pivot[numeric_columns] = df_pivot[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return df_pivot

# Step 2: Calculate ΔF/F for each ROI
def calculate_dff(df):
    baseline = df['Mean'].mean()  # Use the mean of Mean values as baseline for simplicity
    df['dF/F'] = (df['Mean'] - baseline) / baseline
    return df

# Step 3: Sort ROIs by proximity
def sort_rois_by_proximity(df):
    coords = df[['X', 'Y']].values
    dist_matrix = squareform(pdist(coords))
    sorted_indices = np.argsort(dist_matrix.sum(axis=0))
    sorted_df = df.iloc[sorted_indices]
    return sorted_df

# Step 4: Create the heatmap
def create_heatmap(df):
    dff_matrix = df[['dF/F']].values.T  # Transpose to have ROIs as rows
    plt.figure(figsize=(10, 8))
    plt.imshow(dff_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='ΔF/F')
    plt.title('Heatmap of ΔF/F Signals for ROIs')
    plt.xlabel('Time')
    plt.ylabel('ROI (sorted by proximity)')
    plt.show()

# File path
file_dir = '/Users/jcoleman/Documents/GitHub/NMCoop_Shared/#Febo/thy1gcamp6s_dataset1/'
file_name = 'mThy6s2_alldrift_D4_001Z1_ROIdata.csv'
file_path = file_dir + file_name
#file_path = '/mnt/data/mThy6s2_alldrift_D4_001Z1_ROIdata.csv'

# Process data
df = load_and_process_data(file_path)

# Calculate ΔF/F
df = calculate_dff(df)

# Sort ROIs by proximity
sorted_df = sort_rois_by_proximity(df)

# Create heatmap
create_heatmap(sorted_df)

