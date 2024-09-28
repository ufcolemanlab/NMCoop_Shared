#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:33:25 2024

@author: jcoleman
"""

# librairies
import mat73
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import scipy.io as sio
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data from a Pickle file (*.pkl)
def data_load_picklefile(datafile):
    
    #print(datafile)
    
    data_dict = dict()
    with open(datafile, 'rb') as datafile:
        data_dict = pickle.load(datafile)
        
    return data_dict

# (OR) from a MATLAB *.mat >=v7.3 file file
def data_load_matfile(mat_version):
    """
    Example: $> newdata_dict = data_load_matfile(73)
    """
    dirname = '/Users/jcoleman/Documents/--LARGE DATA--/#Pizzi/TBI-sham leak analysis/process_v4_062723/process_v5-2ii/'
    dataname ='MATdataForPython_12302023.mat'
    datafile = dirname + dataname
    #dict_from_matfile = sio.loadmat(datafile) #<v7.3 file
    data_dict = dict()
    if mat_version >= 73:
        data_dict = mat73.loadmat(datafile)
    else:
        data_dict = sio.loadmat(datafile) # if <v7.3
        
    return data_dict

def getSubjectDataframe(dataIn,subjectID,groupID):
    
    #set up subject dataframe
    data1 = dataIn[subjectID]
    dataframe1 = pd.DataFrame()
    
    #setup tags and string IDs
    dataframe1['groupID'] = [groupID] * len(data1['FeFv'])
    """
    grpA =
    
      1×6 cell array
    
        {'m5'}    {'m6'}    {'m9'}    {'m10'}    {'m11'}    {'m12'}
    
    >> grpB
    
    grpB =
    
      1×4 cell array
    
        {'m7'}    {'m13'}    {'m15'}    {'m16'}
    """
    dataframe1['subjectID'] = [subjectID] * len(dataframe1)
    dataframe1['roiID'] = pd.DataFrame(data1['roiID']) #should this include tag for 'm10' etc?
    dataframe1['subjectID_roiID'] = dataframe1['subjectID'] + '_' + dataframe1['roiID']
    
    #setup data/variables/measures
    dataframe1['Zmicrons'] = pd.DataFrame(data1['Zmicrons'])
    dataframe1['mean_FWHM_ums'] = pd.DataFrame(data1['mean_FWHM_ums'])
    dataframe1['FeFv'] = pd.DataFrame(data1['FeFv'])
    #dataframe1['XYcoords'] = pd.DataFrame(data1['XYcoords'])
    dataframe1['Zcoords'] = pd.DataFrame(data1['Zcoords'])
    #data1 = dataZ1['m10_roi1_00001']
    
    #Need to create list of roi(x,y,z) in list() format
    data1_xyztmp = data1['XYZ3D']
    tempXYZ=list()
    for roi in range(np.size(data1_xyztmp,1)):
 
        c1=[data1_xyztmp[0][roi], data1_xyztmp[1][roi], data1_xyztmp[2][roi]]
        tempXYZ.append(c1)
        #{REPEAT c1 = ....['XYZ3D'][0][1], etc}
        tempXYZ.append
    
    #tempXYZ.reshape(rows=rois,3)
    
    dataframe1['XYZ3D'] = tempXYZ
    #time series?
    # basically loop all subjects/files and concatenate into one singular array

    #dataframe1.dropna()

    return dataframe1


def getSubjectDataframeForPCA(dataIn,subjectID,start_integer):
    
    # start_integer = starting number in sequence
        #* would be = 1 for the first subjectID
    
    #set up subject dataframe
    data1 = dataIn[subjectID]
    #start_integer_new = len(data1['FeFv'])
    
    dataframe1 = pd.DataFrame()
    
    #setup tags and string IDs        
    tempvar = list(range(len(data1['FeFv'])))
    dataframe1['numROI'] = [start_integer for x in tempvar]
    #add start_integer to create next sequence
        
    """
    grpA =
    
      1×6 cell array
    
        {'m5'}    {'m6'}    {'m9'}    {'m10'}    {'m11'}    {'m12'}
    
    >> grpB
    
    grpB =
    
      1×4 cell array
    
        {'m7'}    {'m13'}    {'m15'}    {'m16'}
    """
    #setup data/variables/measures
    dataframe1['Zmicrons'] = pd.DataFrame(data1['Zmicrons'])
    dataframe1['mean_FWHM_ums'] = pd.DataFrame(data1['mean_FWHM_ums'])
    dataframe1['FeFv'] = pd.DataFrame(data1['FeFv'])
    dataframe1['Zcoords'] = pd.DataFrame(data1['Zcoords'])


    return dataframe1

#%% LOAD DATA to dictionary var
    # Could combine the two functions into csingle onditional function (ie pickle OR mat file)
#Pickle location
dirname = "/Users/jcoleman/Documents/GitHub/NMCoop_Shared/#Pizzi/data1/"
dataname = "data_dict.pkl" # same data as the Matlab file - region1 from rTBI gg6x vessel data
#dataname = "data_dict.json" # if json file preferred
datafilepath = dirname + dataname

data_dict_raw = data_load_picklefile(datafilepath)
#Variables in main data dict()
# dict_variable_names = ['FeFv',
#                     'XYZ3D',
#                     'XYcoords',
#                     'Zcoords',
#                     'Zmicrons',
#                     'foldFe',
#                     'fwhm_Pv_vars',
#                     'fwhm_flanking_intDen',
#                     'fwhm_flanking_mean',
#                     'maxFe',
#                     'mean_FWHM_pix',
#                     'mean_FWHM_ums',
#                     'roiID',
#                     'roiT_FWHM_pix',
#                     'roiT_FWHM_ums',
#                     'roiT_fwhm_flanking_tData', 
#                     'roiT_profiles']

#%% Convert to PCA-friendly format (NEEDS WORK, CLEAN UP, but works on 7-23-24)

# # Dict-arrays of data of interest
# data_xyz3d = return_parsed_arrays(data_dict, 'XYZ3D')
# data_xy_2d = return_parsed_arrays(data_dict, 'XYcoords')
# data_z_ums = return_parsed_arrays(data_dict, 'Zmicrons')
# data_fefv = return_parsed_arrays(data_dict, 'FeFv')
# data_fwhm_ums = return_parsed_arrays(data_dict, 'mean_FWHM_ums')

#STEP1
dataZ1 =  data_dict_raw['AnalysisStruct']['Z1']
subjectIDlist = list(sorted(dataZ1.keys())) # this order needs to match groups ID list and id_integers
grpA = ['m5', 'm6', 'm9', 'm10', 'm11', 'm12']
grpB = ['m7', 'm13', 'm15', 'm16']
groupsIDlist = ['A','A','A','B','B','B','A','A','B','A']

# first get the length of each ['FeFv' column within dataZ1.keys()]
id_integers = list()
# use sorted() to keep list of lengths consistent with next stage
tempkeydata = list(dataZ1.keys())
for x in range(len(dataZ1.keys())):

    tempkey = tempkeydata[x]
    id_integers.append(len(dataZ1[tempkey]['FeFv']))
# create DataFrames
df_temp = pd.DataFrame()
df_forpca = pd.DataFrame()
for x in range(len(subjectIDlist)):
    
    grp_temp = groupsIDlist[x] #list of subjects and groups2
    df_temp = getSubjectDataframe(dataZ1,subjectIDlist[x],grp_temp)
    #need to figure out how to create id_integers first
    df_forpca = getSubjectDataframeForPCA(dataZ1,subjectIDlist[x],id_integers[x])
    
    if x == 0:
        df_all = df_temp
        df_allpca = df_forpca
    else:
        df_all = pd.concat([df_all, df_temp])
        df_allpca = pd.concat([df_allpca, df_forpca])
        
#add a column to both dataframes with integer IDs corresponding to df_all IDs
tempvar = list(range(len(df_all)))
df_all['rowID'] = [tempvar[x] for x in tempvar]
df_allpca['rowID'] = [tempvar[x] for x in tempvar]
#reassign 'rowID' to serve as the Index
df_all = df_all.set_index('rowID')
df_allpca = df_allpca.set_index('rowID')

#To display in print format for sanity check
rowtmp = 128
display(df_all.iloc[rowtmp])
display(df_allpca.iloc[rowtmp])

# ALSO add group ID number
df_allpca['groupID'] = df_all['groupID'] # or fix the function to include it
df_forPCA = df_allpca.replace(['A','B'],[1,2])
"""
The best way to do this [drop columns in a dataframe] in Pandas is to use drop:

df = df.drop('column_name', axis=1)
where 1 is the axis number (0 for rows and 1 for columns.)

Or, the drop() method accepts index/columns keywords as an alternative to specifying the axis. So we can now just do:

df = df.drop(columns=['column_nameA', 'column_nameB'])
"""

df_pca_rundata = df_forPCA.drop(columns=['numROI', 'Zcoords'])

#display(df_forPCA.iloc[rowtmp])
display(df_pca_rundata.iloc[rowtmp])

# STEP 2 - merging for PCA
# merge variables of interest (rowID, groupID, x, y, z, FeFv, FWHM - means)
merged_alldata=pd.DataFrame()
merged_alldata = df_pca_rundata.dropna() # drop NANs
merged_data = merged_alldata.reset_index(drop=True)
#merged_data_groups = merged_data[['groupID']].copy()
merged_data_groups = merged_data['groupID'].tolist()
merged_data = merged_data.drop('groupID', axis=1)
    # ?for time-series data, add (FeFv/FWHM mean, std, min, max) components to account for T

# Pickle the new PCA data variable(s)
# Save the dictionary to a file
with open('merged_datagroups_nonan_pca.pkl', 'wb') as file:
    pickle.dump(merged_alldata, file)
    
# need to add X, Y, Z, Tseries mean, std, min, max to (*_alldata)

#%% Preparing Pizzis 'XL data'

filepath = "/Users/jcoleman/Documents/GitHub/NMCoop_Shared/#Pizzi/data2/pizzi_perivessel_bytype_results.txt"
df_csvData = pd.read_csv(filepath, sep='\t')

# Print the first few rows of the DataFrame
print(df_csvData.head())

# need to convert 'vessel type' and 'location along AV axis' to numerical categories
# eg arteriole=1, capillary=2, venule=3
# eg Prior to bifurctation=1, post bifurcation=2, mid-capillary=3, mid vessel=4
# Define the mapping dictionaries
vessel_type_map = {
    'arteriole': 1,
    'capillary': 2,
    'venule': 3
}

location_map = {
    'Prior to bifurctation': 1,
    'pre bifurcation': 1, #calling the same as 'Prior...'
    'post bifurcation': 2,
    'mid-capillary': 3,
    'mid vessel': 4,
    ' mid vessel': 4 #calling the same as 'mid vessel' inclded for one 'typo' with space in front of 'mid'
}

# Apply the mapping to the columns
# (preferred)
df_csvData['vessel_type_num'] = df_csvData['vessel type'].map(vessel_type_map)
df_csvData['location_num'] = df_csvData['location along AV axis'].map(location_map)

# OR

# (deprecated)
# df_csvData['vessel_type_num'] = df_csvData['vessel type'].replace(vessel_type_map)
# df_csvData['location_num'] = df_csvData['location along AV axis'].replace(location_map)

print(df_csvData.head())
#%% merged csvData

merged_data_csv = df_csvData.drop('vessel type', axis=1)
merged_data_csv = merged_data_csv.drop('location along AV axis', axis=1)
merged_data_csv = merged_data_csv.drop('perivessel AU 1', axis=1)
merged_data_csv = merged_data_csv.drop('perivessel AU 2', axis=1)
merged_data_csv = merged_data_csv.drop('lumen AU', axis=1)

# Assigning grouping var
def assign_group(df, grpA_animals, grpB_animals):
    """
    Assigns a group value (1 or 2) to each row based on the Animal column.

    Parameters:
    df (DataFrame): Input DataFrame
    grpA_animals (list): List of animal numbers belonging to grpA
    grpB_animals (list): List of animal numbers belonging to grpB

    Returns:
    DataFrame: DataFrame with a new column "group"
    """
    df['group'] = df['Animal'].apply(lambda x: 1 if x in grpA_animals else 2 if x in grpB_animals else None)
    
    return df

grpA = (5, 6, 9, 10, 11, 12) #('5', '6', '9', '10', '11', '12') #% codes from Pizzi
grpB = (7,13,15,16) #('7', '13', '15', '16') #% codes from Pizzi

df2_csv = assign_group(merged_data_csv, grpA, grpB)
merged_data_csv = merged_data_csv.drop('group', axis=1)

print(df2_csv)



#%% ANALYSIS 1 - Run 'blind' PCA
# Run 'blind' PCA - scalar, stats, plot PC1 scores vs raw data (components=3)
    # Figure 1 = Skree plot?
    # Figure 2 = blind PCA with groupIDs plotted (PC1-PC2, 2D)
    # Figure 3 = blind PCA with groupIDs plotted (PC1-P2-PC3, 3D)
    # Figure 4 = PC1 vs raw w/ rho/pval plots

def perform_pca(data, n_components=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=principal_components, index=data.index,
                          columns=[f'PC{i+1}' for i in range(n_components)])
    explained_variance = pca.explained_variance_ratio_
    return pca_df, explained_variance


def check_normality(pca_df, group_labels):
    normality_results = {}
    unique_groups = list(set(group_labels))
    
    for column in ['PC1', 'PC2', 'PC3']:
        normality_results[column] = {'pvals': {}}
        
        for i, group in enumerate(unique_groups):
            group_data = pca_df[column][pd.Series(group_labels) == group]
            w_stat, p_value = shapiro(group_data)
            
            group_key = f'group{i+1}'
            normality_results[column]['pvals'][f'{group_key}_normal'] = p_value > 0.05
            normality_results[column]['pvals'][f'p_value_{group_key}'] = p_value
    
    return normality_results


def run_stat_tests(pca_df, group_labels, normality_results):
    results = {}
    unique_groups = list(set(group_labels))
    
    for col in ['PC1', 'PC2', 'PC3']:
        group1 = pca_df[col][pd.Series(group_labels) == unique_groups[0]]
        group2 = pca_df[col][pd.Series(group_labels) == unique_groups[1]]
        
        if (normality_results[col]['pvals']['group1_normal'] and 
            normality_results[col]['pvals']['group2_normal']):
            # Parametric test: Independent t-test
            t_stat, t_p_value = ttest_ind(group1, group2)
            test_name = 't-test'
            test_stat = t_stat
            p_value = t_p_value
        else:
            # Non-parametric test: Mann-Whitney U test
            u_stat, u_p_value = mannwhitneyu(group1, group2)
            test_name = 'Mann-Whitney U test'
            test_stat = u_stat
            p_value = u_p_value
        
        results[col] = {
            'test': test_name,
            'statistic': test_stat,
            'p_value': p_value
        }
        
    return results

# # Example data
# data = {
#     'Metric1': [0.1, 0.4, 0.7, 1.0],
#     'Metric2': [0.2, 0.5, 0.8, 1.1],
#     'Metric3': [0.3, 0.6, 0.9, 1.2],
#     'X': [1.0, 4.0, 7.0, 10.0],
#     'Y': [2.0, 5.0, 8.0, 11.0],
#     'Z': [3.0, 6.0, 9.0, 12.0]
# }
# merged_data = pd.DataFrame(data)
# merged_data_groups = np.array([1, 2, 1, 2])  # Group labels

# Perform PCA
# NEED TO ADD X, Y, Z DATA then Add Tseries DATA
pca_results, variance_explained = perform_pca(merged_data_csv, n_components=3)

# add the groups or split up from above if add a comprehensive dict() with >3 vars
group_indices = list()
#group_indices = list(merged_alldata['groupID'])
group_indices = list(df2_csv['group'])

# Check normality of PC scores
normality_results = check_normality(pca_results, group_indices)

# Run statistical tests based on normality results
# erged_alldata contains the group ID, in the future this could be the same var
stat_results = run_stat_tests(pca_results, group_indices, normality_results)

# Print results
for pc, results in stat_results.items():
    print(f"Results for {pc}:")
    print(f"  Test: {results['test']}")
    print(f"  Statistic: {results['statistic']}, p-value: {results['p_value']}")
    
# Pickle the 'blind' PCA results
# Save the dictionary to a file
with open('results2CSV_blind_pca.pkl', 'wb') as file:
    pickle.dump(df2_csv, file)

#%% ANALYSIS 1 - Plot data
# Plot the first two principal components
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='PC1', y='PC2', data=pca_results, s=100, alpha=0.7)

# # Add labels for each point
# for i in range(pca_results.shape[0]):
#     if merged_data_groups[i]==1: #'GroupA'
#         plt.text(pca_results.PC1[i], pca_results.PC2[i], pca_results.index[i], 
#                  horizontalalignment='center', verticalalignment='center', color='black')
#     else:
#         plt.text(pca_results.PC1[i], pca_results.PC2[i], pca_results.index[i], 
#                  horizontalalignment='center', verticalalignment='center', color='red')

# # Title and labels
# plt.title('PCA of Vessel Data (GalvoGalvo 6X Scans | 1/3 regions | All capillaries)')
# plt.xlabel(f'PC1 ({variance_explained[0]:.2%} variance)')
# plt.ylabel(f'PC2 ({variance_explained[1]:.2%} variance)')
# plt.grid(True)

# # Show plot
# plt.show()

plt.figure(figsize=(10, 8))

# Create a color map for the scatter plot
color_map = {1: 'black', 2: 'red'}
colors = [color_map[group] for group in group_indices]

# Create the scatter plot with the appropriate colors
sns.scatterplot(x='PC1', y='PC2', data=pca_results, s=100, alpha=0.7, hue=group_indices, palette=color_map, legend=False)

# Add labels for each point
for i in range(pca_results.shape[0]):
    if group_indices[i] == 1:  # 'GroupA'
        plt.text(pca_results.PC1[i], pca_results.PC2[i], pca_results.index[i],
                 horizontalalignment='center', verticalalignment='center', color='black')
    else:
        plt.text(pca_results.PC1[i], pca_results.PC2[i], pca_results.index[i],
                 horizontalalignment='center', verticalalignment='center', color='red')

# Title and labels
plt.title('PCA of Vessel Data (GalvoGalvo 6X Scans | 1/3 regions | Vessels by type - MP)')
plt.xlabel(f'PC1 ({variance_explained[0]:.2%} variance)')
plt.ylabel(f'PC2 ({variance_explained[1]:.2%} variance)')
plt.grid(True)

# Show plot
plt.show()

# Print the amount of variance explained by each component
for i, variance in enumerate(variance_explained):
    print(f'PC{i+1}: {variance:.2%} variance explained')
    
# Plot PC1 vs each raw var data
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
#variable_names = ['Zmicrons', 'mean_FWHM_ums', 'FeFv']
variable_names = ['Fe/Fv', 'vessel_type_num', 'location_num']
# Zmicrons         40.000000
# mean_FWHM_ums     4.286358
# FeFv             35.639807

plot_pc1_vs_original(merged_data_csv, pca_results, variable_names)

# plot group IDs over eqach 

#%% ANALYSIS 2 - Run 'groups' PCA    
# Run 'grouped' PCA - scalar, stats, +groupIDs to PCA, plot PC1 scores vs raw data (components=3)
    # Figure 1 = Skree plot?
    # Figure 2 = grouped PCA with groupIDs plotted (PC1-PC2, 2D)
    # Figure 3 = grouped PCA with groupIDs plotted (PC1-P2-PC3, 3D)
    # Figure 4 = PC1 vs raw w/ rho/pval plots



#%% save dict


#import numpy as np

# Example dictionary with a NumPy array
# data_dict = {
#     'array': np.array([1, 2, 3]),
#     'values': [10, 20, 30]
# }

