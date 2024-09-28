#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 20:29:42 2024

@author: jcoleman
"""

from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import mat73
import numpy as np

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

#%%

dataset = load_digits()
dataset.keys()

#%%
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

#%matplotlib inline

plt.plot(dataset.data[0])

plt.figure('Data rehsaped')

plt.plot(dataset.data[0].reshape(8,8))

plt.gray()
plt.matshow(dataset.data[0].reshape(8,8)) 


#%% kmeans

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
#%matplotlib inline
targetDir = '/Users/jcoleman/Dropbox (UFL)/PYTHONdb/LAbisambra/jupyter_notebooks_data examples/ML notebooks and info/kmeans tutorials/'

df = pd.read_csv(targetDir + "income.csv")
df.head()

plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted

df['cluster']=y_predicted
df.head()

#%% KMeans example with Vessel Data
#   with vessel data (each subject eg m10)
# grab vessel data (dict() with vectors - or generate string list etc)
dirname = '/Users/jcoleman/Documents/--LARGE DATA--/TBI-sham leak analysis/process_v4_062723/process_v5-2ii/'
dataname ='MATdataForPython_12302023.mat'
#    filename_vid = 'vid_mouse1_1_Day3_45s2.h264'
#    datafile = 'vid_mouse1_1_Day3_45s2.h264.mat'

datafile = dirname + dataname
#dict_from_matfile = sio.loadmat(datafile)

data_dict = mat73.loadmat(datafile)

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
"""

#%% USED 7-22-24 to prep data for PCA code
# input vars for function
dataZ1 =  data_dict['AnalysisStruct']['Z1']
# groupID = 'A'
# nsubject = 'm10_roi1_00001'

# #def getSubjectDataframe()
# #set up subject dataframe
# df1 = dataZ1[nsubject]
# df10 = pd.DataFrame()

# #setup tags and string IDs
# df10['groupID'] = [groupID] * len(df1['FeFv'])
# """
# grpA =

#   1×6 cell array

#     {'m5'}    {'m6'}    {'m9'}    {'m10'}    {'m11'}    {'m12'}

# >> grpB

# grpB =

#   1×4 cell array

#     {'m7'}    {'m13'}    {'m15'}    {'m16'}
# """
# df10['subjectID'] = [nsubject] * len(df10)
# df10['roiID'] = pd.DataFrame(df1['roiID']) #should this include tag for 'm10' etc?
# df10['subjectID_roiID'] = df10['subjectID'] + '_' + df10['roiID']

# #setup data/variables/measures
# df10['Zmicrons'] = pd.DataFrame(df1['Zmicrons'])
# df10['mean_FWHM_ums'] = pd.DataFrame(df1['mean_FWHM_ums'])
# df10['FeFv'] = pd.DataFrame(df1['FeFv'])
# #XY, XYZ3D, time series?
# # basically loop all subjects/files and concatenate into one singular array

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

#%% clean up data (eg drop NaN rows - check data?)
df_multidim = df_all.dropna() #df_all.dropna() # drop NANs before kmeans
df_multidimPCA = df_forPCA.dropna() #df_all.dropna() # drop NANs before kmeans etc

# df_grpA = df_multidim[df_multidim.groupID=='A']
# df_grpB = df_multidim[df_multidim.groupID=='B']
df_grpA = df_multidim[df_multidimPCA.groupID==1]
df_grpB = df_multidim[df_multidimPCA.groupID==2]

plt.scatter(df_grpA.FeFv,df_grpA.Zmicrons,color='blue')
plt.scatter(df_grpB.FeFv,df_grpB.Zmicrons,color='red')
plt.xlabel('Fe/Fv (mean over 10s)')
plt.ylabel('Z depth (microns)')

#%%compute kmeans and plot centroids

"""
There is no difference in methodology between 2 and 4 columns. If you have issues then they are probably due to the contents of your columns. K-Means wants numerical columns, with no null/infinite values and avoid categorical data. Here I do it with 4 numerical features:

import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=10, centers=3, n_features=4)

df = pd.DataFrame(X, columns=['Feat_1', 'Feat_2', 'Feat_3', 'Feat_4'])

kmeans = KMeans(n_clusters=3)

y = kmeans.fit_predict(df[['Feat_1', 'Feat_2', 'Feat_3', 'Feat_4']])

df['Cluster'] = y

print(df.head())
"""

df_grpA = X_scalerPCA[df_multidimPCA.groupID==1]
df_grpB = X_scalerPCA[df_multidimPCA.groupID==2]
#temp_data = df_grpB
# temp_data = df_multidim #all data
df_grps = X_scalerPCA

#def function()
km10 = KMeans(n_clusters=3)
#y_predicted = km10.fit_predict(temp_data[['FeFv','Zmicrons']])
# X_minMaxScaler below --> X_scalerPCA
df200 = pd.DataFrame()
df200 = pd.DataFrame(df_grps, columns=['numROI', 'Zmicrons', 'mean_FWHM_ums', 'FeFv'])
y_predicted = km10.fit_predict(df200[['FeFv','mean_FWHM_ums']])
y_predicted

df200['cluster'] = y_predicted
km10.cluster_centers_

temp_data = df200
#%% Plot clusters in scatter
#create temp arrays by clusters
colorsGrpA = ['red','black','green']
colorsGrpB = ['cyan','blue','magenta']
#colorsEdgeGrpB = #get indices for coloring grp B edges wehn Kmeans over all data vs just each group separtely

df101 = temp_data[temp_data.cluster==0] #assign all '0' cluster points, etc.
df102 = temp_data[temp_data.cluster==1] #assign all '1' cluster points, etc.
df103 = temp_data[temp_data.cluster==2] #assign all '2' cluster points, etc.
plt.scatter(df101.FeFv,df101['mean_FWHM_ums'],color=colorsGrpB[0], edgeColors='black')
#plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df102.FeFv,df102['mean_FWHM_ums'],color=colorsGrpB[1])
#plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(df103.FeFv,df103['mean_FWHM_ums'],color=colorsGrpB[2])
plt.scatter(km10.cluster_centers_[:,0],km10.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Fe/Fv (mean over 10s)')
#plt.ylabel('Z depth (microns)') 
plt.ylabel('mean_FWHM_ums')
plt.legend()
ax = plt.gca()
ax.set_xlim([-0.3, 0.5])
ax.set_ylim([-0.5, 1.0])
#%% apply scaler transformation

#temp_data = df_grpB
temp_data = df_all.dropna() #all data

scaler = MinMaxScaler()

#scaler.fit(df[['Income($)']])
scaler.fit(temp_data[['Zmicrons']])
temp_data['Zmicrons'] = scaler.transform(temp_data[['Zmicrons']])

scaler.fit(temp_data[['FeFv']])
temp_data['FeFv'] = scaler.transform(temp_data[['FeFv']])

# scaler.fit(temp_data[['mean_FWHM_ums']])
# temp_data['mean_FWHM_ums'] = scaler.transform(temp_data[['mean_FWHM_ums']])

temp_data.head()

#plt.scatter(temp_data.FeFv,temp_data.Zmicrons)
#plt.scatter(temp_data.mean_FWHM_ums,temp_data.Zmicrons)

#compute kmeans and plot centroids
km10 = KMeans(n_clusters=6) #, random_state=30)
y_predicted = km10.fit_predict(temp_data[['FeFv','Zmicrons']])
# y_predicted = km10.fit_predict(temp_data[['mean_FWHM_ums','Zmicrons']])
# y_predicted = km10.fit_predict(temp_data[['mean_FWHM_ums','FeFv']])
y_predicted

temp_data['cluster'] = y_predicted
km10.cluster_centers_

#create temp arrays by clusters
plt.title('ALL - FeFv vs Zmicrons')
df101 = temp_data[temp_data.cluster==0] #assign all '0' cluster points, etc.
df102 = temp_data[temp_data.cluster==1] #assign all '1' cluster points, etc.
df103 = temp_data[temp_data.cluster==2] #assign all '2' cluster points, etc.
df104 = temp_data[temp_data.cluster==3]
df105 = temp_data[temp_data.cluster==4]
df106 = temp_data[temp_data.cluster==5]

dfGrpA = temp_data[temp_data.groupID=='A']
dfGrpB = temp_data[temp_data.groupID=='B']

#grpA
plt.scatter(df101.FeFv,df101.Zmicrons,color='green')
plt.scatter(df102.FeFv,df102.Zmicrons,color='red')
plt.scatter(df103.FeFv,df103.Zmicrons,color='black')
plt.scatter(df104.FeFv,df104.Zmicrons,color='cyan')
plt.scatter(df105.FeFv,df105.Zmicrons,color='magenta')
plt.scatter(df106.FeFv,df106.Zmicrons,color='orange')

# plt.scatter(df101.mean_FWHM_ums,df101.Zmicrons,color='green')
# plt.scatter(df102.mean_FWHM_ums,df102.Zmicrons,color='red')
# plt.scatter(df103.mean_FWHM_ums,df103.Zmicrons,color='black')

# plt.scatter(df101.mean_FWHM_ums,df101.FeFv,color='green',edgecolors='none')
# plt.scatter(df102.mean_FWHM_ums,df102.FeFv,color='red',edgecolors='none')
# # plt.scatter(df103.mean_FWHM_ums,df103.FeFv,color='black',edgecolors='none')

#grpB
# plt.scatter(df101.FeFv,df101.Zmicrons,color='none',edgecolors='green')
# plt.scatter(df102.FeFv,df102.Zmicrons,color='none',edgecolors='red')
# plt.scatter(df103.FeFv,df103.Zmicrons,color='none',edgecolors='black')

# plt.scatter(df101.mean_FWHM_ums,df101.Zmicrons,color='cyan')
# plt.scatter(df102.mean_FWHM_ums,df102.Zmicrons,color='blue')
# plt.scatter(df103.mean_FWHM_ums,df103.Zmicrons,color='magenta')

# plt.scatter(df101.mean_FWHM_ums,df101.FeFv,color='none',edgecolors='green')
# plt.scatter(df102.mean_FWHM_ums,df102.FeFv,color='none',edgecolors='red')
# plt.scatter(df103.mean_FWHM_ums,df103.FeFv,color='none',edgecolors='black')

plt.scatter(km10.cluster_centers_[:,0],km10.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
#plt.xlabel('Fe/Fv (mean over 10s)')
#plt.ylabel('Z depth (microns)')

# plt.scatter(dfGrpA.mean_FWHM_ums,dfGrpA.FeFv,color='none',edgecolors='black')
# plt.scatter(dfGrpB.mean_FWHM_ums,dfGrpB.FeFv,color='none',edgecolors='cyan')

plt.legend()

#%% Elbow plot

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    #km.fit(df[['Age','Income($)']])
    km.fit(temp_data[['FeFv','Zmicrons']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.axhline(y = 2.0, color = 'b', linestyle = '--')
plt.axhline(y = 1.5, color = 'r', linestyle = '--') 
plt.axhline(y = 1.0, color = 'g', linestyle = '--') 
plt.show()
#plt.title('GrpA + GrpB')

#%% PCA exercise

#import pandas as pd

# https://www.kaggle.com/fedesoriano/heart-failure-prediction
#df_pca = df_multidim
df_multidimPCA.head()

#df_multidimPCA.shape() #?tuple error???

# X = df_multidimPCA.drop("rowID",axis='columns')
#X = df_multidimPCA.drop("numROI",axis='columns')
X1 = df_multidimPCA.drop("groupID",axis='columns')
X2 = X1.drop("numROI",axis='columns')
X = X2.drop("Zcoords",axis='columns')


#? Add column of numbers corresponding to rowID?
#y = df5.HeartDisease
# maybe 'A' 'B' shou dbe 1 or 2?
y = df_multidimPCA.groupID

X.head()
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)
#%%
# X_train.shape()
# X_test.shape()

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
model_rf.score(X_test, y_test)

#%% re-train with PCA
from sklearn.decomposition import PCA

#pca = PCA(0.95)
pca = PCA()
X_pca = pca.fit_transform(X)
X_pca

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)
#X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, random_state=30)
#%%
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train_pca, y_train)
model_rf.score(X_test_pca, y_test)

#%% Scale df for PCA
#temp_data = df_grpB
#X_minMaxScaler = X #all data

X_minMaxScaler = df_forPCA.dropna() #all data #.drop("groupID",axis='columns')
X_minMaxScaler = X_minMaxScaler.drop("groupID",axis='columns')
#X_minMaxScaler = X_minMaxScaler.drop("numROI",axis='columns')
X_minMaxScaler = X_minMaxScaler.drop("Zcoords",axis='columns')

scaler = MinMaxScaler()

#scaler.fit(df[['Income($)']])
scaler.fit(X_minMaxScaler[['Zmicrons']])
X_minMaxScaler['Zmicrons'] = scaler.transform(X_minMaxScaler[['Zmicrons']])

scaler.fit(X_minMaxScaler[['mean_FWHM_ums']])
X_minMaxScaler['mean_FWHM_ums'] = scaler.transform(X_minMaxScaler[['mean_FWHM_ums']])

scaler.fit(X_minMaxScaler[['FeFv']])
X_minMaxScaler['FeFv'] = scaler.transform(X_minMaxScaler[['FeFv']])

#pca = PCA(0.95)
pca = PCA()
X_scalerPCA = pca.fit_transform(X_minMaxScaler)
X_scalerPCA

#%%
colors = y-1
#c=colors,marker="o", cmap="bwr_r" #red is 0, blue is 1
#plt.scatter(X_pca[:,1],X_pca[:,2],c=colors,marker="o", cmap="bwr_r")
plt.scatter(X_pca[:,1],X_pca[:,2],c=y,cmap='viridis',edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


#%% Alt PCA
# Initialise
nc = 10
pca1 = PCA() #pca(n_components=nc)
pca2 = PCA() #pca(n_components=nc)
 
# Scale the features to have zero mean and standard devisation of 1
# This is important when correlating data with very different variances
X_scaled = StandardScaler().fit_transform(X)
#nfeat2 = StandardScaler().fit_transform(dfeat)

# X1 = pca1.fit_transform(X_scaled)
X1 = pca1.fit_transform(X_scaled)
#X1 = pca1.fit(X_scaled)
#expl_var_1 = X1.explained_variance_ratio_

#%%
#colors = (y-1)
#labplot = ['GroupA', 'GroupB']
#c=colors,marker="o", cmap="bwr_r" #red is 0, blue is 1
#plt.scatter(X_pca[:,1],X_pca[:,2],c=colors,marker="o", cmap="bwr_r")
#plt.scatter(X_scaled[:,1],X_scaled[:,2],c=y,cmap='viridis',edgecolor='k')
#plt.scatter(X_scaled[:,1],X_scaled[:,2],c=y,cmap='bwr_r')#,edgecolor='k')
plt.scatter(X_scalerPCA[:,2],X_scalerPCA[:,3],c=y,cmap='bwr_r')#,edgecolor='k')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Z/FWHM/FeFv - minMaxScaler')
# plt.legend(['GroupA', 'GroupB'],loc='upper right')
plt.show()

