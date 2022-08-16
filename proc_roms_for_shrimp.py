#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:17:31 2021

@author: N
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Bottom Layer Temperature data
df1 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/BLT_per_area_shrimp_0_to_706.csv') 
df2 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/BLT_per_area_shrimp_706_to_1412.csv') 
df3 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/BLT_per_area_shrimp_1412_to_2118.csv') 
df4 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/BLT_per_area_shrimp_2118_to_2294.csv') 
df5 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/BLT_per_area_shrimp_2294_to_2470.csv') 
df6 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/BLT_per_area_shrimp_2470_to_2646.csv') 
df7 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/BLT_per_area_shrimp_2646_to_2827.csv') 

df_raw_BLT = df1.append([df2, df3, df4, df5, df6, df7])
df_raw_BLT = df_raw_BLT.reset_index()
df_raw_BLT = df_raw_BLT.drop(columns = ['Unnamed: 0'])
df_raw_BLT = df_raw_BLT.drop(columns = ['index'])
df_raw_BLT['date'] = pd.to_datetime(df_raw_BLT['date'], format = '%Y%m%d')

# Load Top Layer Temperature
df8 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TLT_per_area_shrimp_0_to_706.csv')
df9 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TLT_per_area_shrimp_706_to_1412.csv')
df10 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TLT_per_area_shrimp_1412_to_2118.csv')
df11 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TLT_per_area_shrimp_2118_to_2827.csv')

df_raw_TLT = df8.append([df9, df10, df11])
df_raw_TLT = df_raw_TLT.reset_index()
df_raw_TLT = df_raw_TLT.drop(columns = ['Unnamed: 0'])
df_raw_TLT = df_raw_TLT.drop(columns = ['index'])
df_raw_TLT['date'] = pd.to_datetime(df_raw_TLT['date'], format = '%Y%m%d')

# Load top layer LS, CS, and TLT on 150 km offshore
df12 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_1502118_to_2827.csv')
df13 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_0_to_177.csv')
df14 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_177_to_353.csv')
df15 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_353_to_530.csv')
df16 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_530_to_706.csv')
df17 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_706_to_883.csv')
df18 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_883_to_1059.csv')
df19 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_1059_to_1236.csv') 
df20 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_1236_to_1412.csv')
df21 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_1412_to_1589.csv')                                     
df22 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_1589_to_1765.csv')
df23 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_1765_to_1942.csv')
df24 = pd.read_csv('/Users/Documents/PostdocUW/data_shrimp/TL_per_area_shrimp_150_1942_to_2118.csv')                    
                   
                   
df_raw_150 = df12.append([df13, df14, df15,df16, df17, df18,df19, df20, df21,df22, df23, df24])
df_raw_150 = df_raw_150.drop(columns = ['Unnamed: 0'])
df_raw_150['date'] = pd.to_datetime(df_raw_150['date'], format = '%Y%m%d')

year_overlap = list(range(1981,2011))
yearly = list(range(1980,2011))

area_number = [32,30,29,28,26,24,22,21,20,19,18,12]

feml_precond =["-10-01","-02-28"]
Spwn = ["-03-01", "-04-30"]
lrvl_phase =["-05-01","-09-30"]
may       = ["-05-01","-05-31"]
june      = ["-06-01","-06-30"]
july      = ["-07-01","-07-31"]
august    = ["-08-01","-08-31"]
september = ["-09-01","-09-30"]

meanBLT_fml_precond_area = pd.DataFrame()
meanBLT_spwn_area = pd.DataFrame()
maxTLT_spwn_area = pd.DataFrame()
maxTLT_150_area = pd.DataFrame()
meanTLT_150_area = pd.DataFrame()
LS_150_area = pd.DataFrame()
CS_150_area = pd.DataFrame()
maxTLT_150_area_may = pd.DataFrame()
meanTLT_150_area_may = pd.DataFrame()
LS_150_area_may = pd.DataFrame()
CS_150_area_may = pd.DataFrame()
maxTLT_150_area_june = pd.DataFrame()
meanTLT_150_area_june = pd.DataFrame()
LS_150_area_june = pd.DataFrame()
CS_150_area_june = pd.DataFrame()
maxTLT_150_area_july = pd.DataFrame()
meanTLT_150_area_july = pd.DataFrame()
LS_150_area_july = pd.DataFrame()
CS_150_area_july = pd.DataFrame()
maxTLT_150_area_aug= pd.DataFrame()
meanTLT_150_area_aug = pd.DataFrame()
LS_150_area_aug = pd.DataFrame()
CS_150_area_aug = pd.DataFrame()
maxTLT_150_area_sep= pd.DataFrame()
meanTLT_150_area_sep = pd.DataFrame()
LS_150_area_sep = pd.DataFrame()
CS_150_area_sep = pd.DataFrame()
for i in area_number:
    meanBLT_fml_precond_area[str(i)] = [np.nan]+[df_raw_BLT[(df_raw_BLT['area'] == i) & (df_raw_BLT['date'] > pd.to_datetime(str(u-1)+feml_precond[0])) & (df_raw_BLT['date'] < pd.to_datetime(str(u)+feml_precond[1]))]['BLT'].mean() 
     for u in year_overlap] 
    meanBLT_spwn_area[str(i)] = [df_raw_BLT[(df_raw_BLT['area'] == i) & (df_raw_BLT['date'] > pd.to_datetime(str(u)+Spwn[0])) & (df_raw_BLT['date'] < pd.to_datetime(str(u)+Spwn[1]))]['BLT'].mean() 
     for u in yearly ] 
    maxTLT_spwn_area[str(i)] = [df_raw_TLT[(df_raw_TLT['area'] == i) & (df_raw_TLT['date'] > pd.to_datetime(str(u)+Spwn[0])) & (df_raw_TLT['date'] < pd.to_datetime(str(u)+Spwn[1]))]['BLT'].max() 
     for u in yearly ]
    
    # average values over the whole larval phase
    maxTLT_150_area[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+lrvl_phase[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+lrvl_phase[1]))]['TLT'].max() 
     for u in yearly ]
    meanTLT_150_area[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+lrvl_phase[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+lrvl_phase[1]))]['TLT'].mean() 
     for u in yearly ]
    LS_150_area[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+lrvl_phase[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+lrvl_phase[1]))]['TLS'].mean() 
     for u in yearly ]
    CS_150_area[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+lrvl_phase[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+lrvl_phase[1]))]['TCS'].mean() 
     for u in yearly ]
    
    # value during may first month of larval phase
    maxTLT_150_area_may[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+may[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+may[1]))]['TLT'].max() 
     for u in yearly ]
    meanTLT_150_area_may[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+may[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+may[1]))]['TLT'].mean() 
     for u in yearly ]
    LS_150_area_may[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+may[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+may[1]))]['TLS'].mean() 
     for u in yearly ]
    CS_150_area_may[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+may[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+may[1]))]['TCS'].mean() 
     for u in yearly ]   
    
    # value during june of larval phase
    maxTLT_150_area_june[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+june[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+june[1]))]['TLT'].max() 
     for u in yearly ]
    meanTLT_150_area_june[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+june[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+june[1]))]['TLT'].mean() 
     for u in yearly ]
    LS_150_area_june[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+june[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+june[1]))]['TLS'].mean() 
     for u in yearly ]
    CS_150_area_june[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+june[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+june[1]))]['TCS'].mean() 
     for u in yearly ]  
    
    # value during july of larval phase
    maxTLT_150_area_july[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+july[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+july[1]))]['TLT'].max() 
     for u in yearly ]
    meanTLT_150_area_july[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+july[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+july[1]))]['TLT'].mean() 
     for u in yearly ]
    LS_150_area_july[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+july[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+july[1]))]['TLS'].mean() 
     for u in yearly ]
    CS_150_area_july[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+july[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+july[1]))]['TCS'].mean() 
     for u in yearly ] 
    
    # value during august of larval phase
    maxTLT_150_area_aug[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+august[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+august[1]))]['TLT'].max() 
     for u in yearly ]
    meanTLT_150_area_aug[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+august[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+august[1]))]['TLT'].mean() 
     for u in yearly ]
    LS_150_area_aug[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+august[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+august[1]))]['TLS'].mean() 
     for u in yearly ]
    CS_150_area_aug[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+august[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+august[1]))]['TCS'].mean() 
     for u in yearly ] 
    
    # value during september of larval phase
    maxTLT_150_area_sep[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+september[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+september[1]))]['TLT'].max() 
     for u in yearly ]
    meanTLT_150_area_sep[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+september[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+september[1]))]['TLT'].mean() 
     for u in yearly ]
    LS_150_area_sep[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+september[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+september[1]))]['TLS'].mean() 
     for u in yearly ]
    CS_150_area_sep[str(i)] = [df_raw_150[(df_raw_150['area'] == i) & (df_raw_150['date'] > pd.to_datetime(str(u)+september[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+september[1]))]['TCS'].mean() 
     for u in yearly ] 
    
meanBLT_fml_precond_area['year'] = list(range(1980,2011))
meanBLT_spwn_area['year'] = list(range(1980,2011))
maxTLT_spwn_area['year'] = list(range(1980,2011))
maxTLT_150_area['year'] = list(range(1980,2011))
meanTLT_150_area['year'] = list(range(1980,2011))
LS_150_area['year'] = list(range(1980,2011))
CS_150_area['year'] = list(range(1980,2011))
maxTLT_150_area_may['year'] = list(range(1980,2011))
meanTLT_150_area_may['year'] = list(range(1980,2011))
LS_150_area_may['year'] = list(range(1980,2011))
CS_150_area_may['year'] = list(range(1980,2011))
maxTLT_150_area_june['year'] = list(range(1980,2011))
meanTLT_150_area_june['year'] = list(range(1980,2011))
LS_150_area_june['year'] = list(range(1980,2011))
CS_150_area_june['year'] = list(range(1980,2011))
maxTLT_150_area_july['year'] = list(range(1980,2011))
meanTLT_150_area_july['year'] = list(range(1980,2011))
LS_150_area_july['year'] = list(range(1980,2011))
CS_150_area_july['year'] = list(range(1980,2011))
maxTLT_150_area_aug['year'] = list(range(1980,2011))
meanTLT_150_area_aug['year'] = list(range(1980,2011))
LS_150_area_aug['year'] = list(range(1980,2011))
CS_150_area_aug['year'] = list(range(1980,2011))
maxTLT_150_area_sep['year'] = list(range(1980,2011))
meanTLT_150_area_sep['year'] = list(range(1980,2011))
LS_150_area_sep['year'] = list(range(1980,2011))
CS_150_area_sep['year'] = list(range(1980,2011))

meanBLT_fml_precond_OR = [np.nan]+[df_raw_BLT[(df_raw_BLT['area'] >= 19) & (df_raw_BLT['area'] <= 28) & (df_raw_BLT['date'] > pd.to_datetime(str(u-1)+feml_precond[0])) & (df_raw_BLT['date'] < pd.to_datetime(str(u)+feml_precond[1]))]['BLT'].mean() 
     for u in year_overlap] 
meanBLT_spwn_OR = [df_raw_BLT[(df_raw_BLT['area'] >= 19) & (df_raw_BLT['area'] <= 28) & (df_raw_BLT['date'] > pd.to_datetime(str(u)+Spwn[0])) & (df_raw_BLT['date'] < pd.to_datetime(str(u)+Spwn[1]))]['BLT'].mean() 
     for u in yearly ] 
maxBLT_fml_precond_OR = [np.nan]+[df_raw_BLT[(df_raw_BLT['area'] >= 19) & (df_raw_BLT['area'] <= 28) & (df_raw_BLT['date'] > pd.to_datetime(str(u-1)+feml_precond[0])) & (df_raw_BLT['date'] < pd.to_datetime(str(u)+feml_precond[1]))]['BLT'].max() 
     for u in year_overlap]
maxBLT_spwn_OR = [df_raw_BLT[(df_raw_BLT['area'] >= 19) & (df_raw_BLT['area'] <= 28) & (df_raw_BLT['date'] > pd.to_datetime(str(u)+Spwn[0])) & (df_raw_BLT['date'] < pd.to_datetime(str(u)+Spwn[1]))]['BLT'].max() 
     for u in yearly ] 
maxTLT_spwn_OR = [df_raw_TLT[(df_raw_TLT['area'] >= 19) & (df_raw_TLT['area'] <= 28) & (df_raw_TLT['date'] > pd.to_datetime(str(u)+Spwn[0])) & (df_raw_TLT['date'] < pd.to_datetime(str(u)+Spwn[1]))]['BLT'].max() 
     for u in yearly ]
maxTLT_150_OR = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+lrvl_phase[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+lrvl_phase[1]))]['TLT'].max() 
     for u in yearly ]
meanTLT_150_OR = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+lrvl_phase[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+lrvl_phase[1]))]['TLT'].mean() 
     for u in yearly ]
LS_150_OR = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+lrvl_phase[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+lrvl_phase[1]))]['TLS'].mean() 
     for u in yearly ]
CS_150_OR = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+lrvl_phase[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+lrvl_phase[1]))]['TCS'].mean() 
     for u in yearly ]
maxTLT_150_OR_may = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+may[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+may[1]))]['TLT'].max() 
     for u in yearly ]
meanTLT_150_OR_may = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+may[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+may[1]))]['TLT'].mean() 
     for u in yearly ]
LS_150_OR_may = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+may[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+may[1]))]['TLS'].mean() 
     for u in yearly ]
CS_150_OR_may = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+may[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+may[1]))]['TCS'].mean() 
     for u in yearly ]
maxTLT_150_OR_june = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+june[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+june[1]))]['TLT'].max() 
     for u in yearly ]
meanTLT_150_OR_june = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+june[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+june[1]))]['TLT'].mean() 
     for u in yearly ]
LS_150_OR_june = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+june[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+june[1]))]['TLS'].mean() 
     for u in yearly ]
CS_150_OR_june = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+june[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+june[1]))]['TCS'].mean() 
     for u in yearly ]
maxTLT_150_OR_july = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+july[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+july[1]))]['TLT'].max() 
     for u in yearly ]
meanTLT_150_OR_july = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+july[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+july[1]))]['TLT'].mean() 
     for u in yearly ]
LS_150_OR_july = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+july[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+july[1]))]['TLS'].mean() 
     for u in yearly ]
CS_150_OR_july = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+july[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+july[1]))]['TCS'].mean() 
     for u in yearly ]
maxTLT_150_OR_aug = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+august[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+august[1]))]['TLT'].max() 
     for u in yearly ]
meanTLT_150_OR_aug = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+august[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+august[1]))]['TLT'].mean() 
     for u in yearly ]
LS_150_OR_aug = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+august[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+august[1]))]['TLS'].mean() 
     for u in yearly ]
CS_150_OR_aug = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+august[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+august[1]))]['TCS'].mean() 
     for u in yearly ]
maxTLT_150_OR_sep = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+september[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+september[1]))]['TLT'].max() 
     for u in yearly ]
meanTLT_150_OR_sep = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+september[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+september[1]))]['TLT'].mean() 
     for u in yearly ]
LS_150_OR_sep = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+september[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+september[1]))]['TLS'].mean() 
     for u in yearly ]
CS_150_OR_sep = [df_raw_150[(df_raw_150['area'] >= 19) & (df_raw_150['area'] <= 28) & (df_raw_150['date'] > pd.to_datetime(str(u)+september[0])) & (df_raw_150['date'] < pd.to_datetime(str(u)+september[1]))]['TCS'].mean() 
     for u in yearly ]
##############################################################################################
# gather independent and dependent variables into one giant matrix
VPE =  pd.read_excel('/Users/Documents/PostdocUW/shrimp_data/OR-VPE.xlsx', index_col=0)

X_shrimp =pd.DataFrame({'year': list(range(1980,2011)),
                       'meanBLT_fml_precond': meanBLT_fml_precond_OR,
                       'meanBLT_spwn': meanBLT_spwn_OR,
                       'maxBLT_fml_precond' : maxBLT_fml_precond_OR,
                       'maxTLT_spwn': maxTLT_spwn_OR,
                       'maxTLT_150': maxTLT_150_OR,
                       'meanTLT_150': meanTLT_150_OR,
                       'LS_150': LS_150_OR,
                       'CS_150': CS_150_OR,
                       'maxTLT_150_may': maxTLT_150_OR_may,
                       'meanTLT_150_may': meanTLT_150_OR_may,
                       'LS_150_may': LS_150_OR_may,
                       'CS_150_may': CS_150_OR_may,
                       'maxTLT_150_june': maxTLT_150_OR_june,
                       'meanTLT_150_june': meanTLT_150_OR_june,
                       'LS_150_june': LS_150_OR_june,
                       'CS_150_june': CS_150_OR_june,
                       'maxTLT_150_july': maxTLT_150_OR_july,
                       'meanTLT_150_july': meanTLT_150_OR_july,
                       'LS_150_july': LS_150_OR_july,
                       'CS_150_july': CS_150_OR_july,
                       'maxTLT_150_aug': maxTLT_150_OR_aug,
                       'meanTLT_150_aug': meanTLT_150_OR_aug,
                       'LS_150_aug': LS_150_OR_aug,
                       'CS_150_aug': CS_150_OR_aug,
                       'maxTLT_150_sep': maxTLT_150_OR_sep,
                       'meanTLT_150_sep': meanTLT_150_OR_sep,
                       'LS_150_sep': LS_150_OR_sep,
                       'CS_150_sep': CS_150_OR_sep,
                       'Y': VPE[(VPE['Larval release year']>=1980) & (VPE['Larval release year']<=2010)]['OR VPE calculated']
                       })
                       
X_shrimp.to_csv("/Users/Dropbox/UM_PD/R/data/X_shrimp.csv",)

##############################################################################################

meanBLT_fml_precond_area.to_csv("/Users/Dropbox/UM_PD/R/data/meanBLT_fml_precond_area.csv",)
meanBLT_spwn_area.to_csv("/Users/Dropbox/UM_PD/R/data/meanBLT_spwn_area.csv",)
maxTLT_spwn_area.to_csv("/Users/Dropbox/UM_PD/R/data/maxTLT_spwn_area.csv",)
maxTLT_150_area.to_csv("/Users/Dropbox/UM_PD/R/data/meanBLT_fml_precond_area.csv",)
meanTLT_150_area.to_csv("/Users/Dropbox/UM_PD/R/data/maxTLT_150_area.csv",)
LS_150_area.to_csv("/Users/Dropbox/UM_PD/R/data/meanBLT_fml_precond_area.csv",)
CS_150_area.to_csv("/Users/Dropbox/UM_PD/R/data/LS_150_area.csv",)
maxTLT_150_area_may.to_csv("/Users/Dropbox/UM_PD/R/data/maxTLT_150_area_may.csv",)
meanTLT_150_area_may.to_csv("/Users/Dropbox/UM_PD/R/data/meanTLT_150_area_may.csv",)
LS_150_area_may.to_csv("/Users/Dropbox/UM_PD/R/data/LS_150_area_may.csv",)
CS_150_area_may.to_csv("/Users/Dropbox/UM_PD/R/data/CS_150_area_may.csv",)
maxTLT_150_area_june.to_csv("/Users/Dropbox/UM_PD/R/data/maxTLT_150_area_june.csv",)
meanTLT_150_area_june.to_csv("/Users/Dropbox/UM_PD/R/data/meanTLT_150_area_june.csv",)
LS_150_area_june.to_csv("/Users/Dropbox/UM_PD/R/data/LS_150_area_june.csv",)
CS_150_area_june.to_csv("/Users/Dropbox/UM_PD/R/data/CS_150_area_june.csv",)
maxTLT_150_area_july.to_csv("/Users/Dropbox/UM_PD/R/data/maxTLT_150_area_july.csv",)
meanTLT_150_area_july.to_csv("/Users/Dropbox/UM_PD/R/data/meanTLT_150_area_july.csv",)
LS_150_area_july.to_csv("/Users/Dropbox/UM_PD/R/data/LS_150_area_july.csv",)
CS_150_area_july.to_csv("/Users/Dropbox/UM_PD/R/data/CS_150_area_july.csv",)
maxTLT_150_area_aug.to_csv("/Users/Dropbox/UM_PD/R/data/maxTLT_150_area_aug.csv",)
meanTLT_150_area_aug.to_csv("/Users/Dropbox/UM_PD/R/data/meanTLT_150_area_aug.csv",)
LS_150_area_aug.to_csv("/Users/Dropbox/UM_PD/R/data/LS_150_area_aug.csv",)
CS_150_area_aug.to_csv("/Users/Dropbox/UM_PD/R/data/CS_150_area_aug.csv",)
maxTLT_150_area_sep.to_csv("/Users/Dropbox/UM_PD/R/data/maxTLT_150_area_sep.csv",)
meanTLT_150_area_sep.to_csv("/Users/Dropbox/UM_PD/R/data/meanTLT_150_area_sep.csv",)
LS_150_area_sep.to_csv("/Users/Dropbox/UM_PD/R/data/LS_150_area_sep.csv",)
CS_150_area_sep.to_csv("/Users/Dropbox/UM_PD/R/data/CS_150_area_sep.csv",)



