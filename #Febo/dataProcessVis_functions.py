#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:32:35 2024

@author: jcoleman
"""

import pandas as pd
import matplotlib.pyplot as plt

def load_and_process_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Reshape the dataframe to have Area, Mean, X, Y, Circ, IntDen, RawIntDen, AR, Round, Solidity in one column for easier manipulation
    df_melted = pd.melt(data, id_vars=[], var_name='Measurement', value_name='Value')

    # Separate into columns to understand which Area it belongs to and which measurement type it is
    df_melted['AreaID'] = df_melted['Measurement'].str.extract(r'(\d+)')
    df_melted['MeasurementType'] = df_melted['Measurement'].str.extract(r'([a-zA-Z]+)')

    # Create a pivot table to reshape back to a more useful dataframe
    df_pivot = df_melted.pivot_table(index=['AreaID'], columns='MeasurementType', values='Value', aggfunc='first').reset_index()

    # Convert necessary columns to numeric
    numeric_columns = ['Area', 'Mean', 'X', 'Y', 'Circ', 'IntDen', 'RawIntDen', 'AR', 'Round', 'Solidity']
    df_pivot[numeric_columns] = df_pivot[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return df_pivot

def identify_rois(df):
    # Identify the smallest and largest areas
    smallest_areas = df.nsmallest(2, 'Area')
    largest_areas = df.nlargest(2, 'Area')

    # Assume that the smallest areas are the line ROIs and the largest areas are the background ROIs
    line_rois = smallest_areas
    background_rois = largest_areas

    return line_rois, background_rois

def normalize_intden(line_rois, background_rois):
    # Normalize the integrated density of line ROIs to the closest background ROI integrated density
    def normalize_intden(row):
        distances = ((background_rois['X'] - row['X'])**2 + (background_rois['Y'] - row['Y'])**2)**0.5
        closest_background_intden = background_rois.loc[distances.idxmin(), 'IntDen']
        return row['IntDen'] / closest_background_intden

    line_rois['NormalizedIntDen'] = line_rois.apply(normalize_intden, axis=1)
    return line_rois

def create_scatter_plot(line_rois):
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(line_rois['X'], line_rois['Y'], s=line_rois['NormalizedIntDen']*100, alpha=0.6)
    plt.title('Scatter Plot of Line ROIs with Normalized Integrated Density')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.colorbar(scatter, label='Normalized Integrated Density')
    plt.show()

# File path
file_dir = '/Users/jcoleman/Documents/GitHub/NMCoop_Shared/#Febo/thy1gcamp6s_dataset1/'
file_name = 'mThy6s2_alldrift_D4_001Z1_ROIdata.csv'
file_path = file_dir + file_name
#file_path = '/mnt/data/mThy6s2_alldrift_D4_001Z1_ROIdata.csv'

# Process data
df = load_and_process_data(file_path)

# Identify ROIs
line_rois, background_rois = identify_rois(df)

# Normalize integrated density
line_rois = normalize_intden(line_rois, background_rois)

# Create scatter plot
create_scatter_plot(line_rois)
