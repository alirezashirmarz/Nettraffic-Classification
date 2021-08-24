# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:02:30 2021

@author: alireza Shirmarz & Negin Mohammadi
"""

"""
Import Required Packages
"""
import pandas as pd  # DataFram Import
import numpy as np   # Numpy Import

""" ----------------------------Import & Merge Data Sets--------------------------"""
"""
Import All Datasets 
"""
## Set File Address in Local PC ## 
File_Addr1="E:\\Dataset\\140004_Dataset_NargesMohammadi\\Scenario A1\\DS-15s-VPN.arff"
File_Addr2="E:\\Dataset\\140004_Dataset_NargesMohammadi\\Scenario A1\\DSt-30s-VPN.arff"
File_Addr3="E:\\Dataset\\140004_Dataset_NargesMohammadi\\Scenario A1\\DS-60s-VPN.arff"
File_Addr4="E:\\Dataset\\140004_Dataset_NargesMohammadi\\Scenario A1\\DS-120s-VPN.arff"
File_Addr_To_Save="E:\\Dataset\\140004_Dataset_NargesMohammadi\\Scenario A1\\ALLData.arff"


## DS1 Import ##
DS1=pd.read_csv(File_Addr1)
DS1.columns=['duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat','max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond','flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat','std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active','min_idle', 'mean_idle', 'max_idle', 'std_idle', 'class1']
shape1=DS1.shape
print(shape1)
print("Columns:",DS1.columns,"\n","Rows:",DS1.index)
DS1.head(10)


## DS2 Import ##
DS2=pd.read_csv(File_Addr2)
DS2.columns=['duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat','max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond','flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat','std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active','min_idle', 'mean_idle', 'max_idle', 'std_idle', 'class1']
shape2=DS2.shape
print(shape2)
print("Columns:",DS2.columns,"\n","Rows:",DS2.index)
DS2.head(10)


## DS3 Import ##
DS3=pd.read_csv(File_Addr3)
DS3.columns=['duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat','max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond','flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat','std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active','min_idle', 'mean_idle', 'max_idle', 'std_idle', 'class1']
shape3=DS3.shape
print(shape3)
print("Columns:",DS3.columns,"\n","Rows:",DS3.index)
DS3.head(10)


## DS4 Import ##
DS4=pd.read_csv(File_Addr4)
DS4.columns=['duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat','max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond','flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat','std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active','min_idle', 'mean_idle', 'max_idle', 'std_idle', 'class1']
shape4=DS4.shape
print(shape4)
print("Columns:",DS4.columns,"\n","Rows:",DS4.index)
DS4.head(10)

""" Merge 4 Dataset in DS """

temp = [DS1,DS2,DS3,DS4]
DS=pd.concat(temp)
shape=DS.shape
print(shape)
print("Columns:",DS.columns,"\n","Rows:",DS.index)
DS.head(10)


""" Save in a New Dataset """
DS.to_csv(File_Addr_To_Save)


""" Load DS """


""" ----------------------------  Dataset Preprocessing --------------------------"""

""" Check the Null Value in DS """
tmp=DS.isna().sum() # if All Columns are 0 (Zero), This dataset is okey!!

for i in range(len(tmp)):
    if tmp[i]!=0:
        print("Num:", tmp[i], "index:", i)
        
 ## ]In this section Index of Columns which have null value
 
 """ َ--------------------------- Autoencoder Implementation -------------------- """
 
 
 
"""
Read Dataset
ds.loc['RowName']['Colname']
ds.iloc[Rowid][Colid]
"""

