2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 11:11:41 2021

@author: N
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np


############################################################################## 
df_CS_1 = pd.read_csv('/Users/Documents/PostdocUW/data/CS_0_to_942.csv') 
df_CS_2 = pd.read_csv('/Users/Documents/PostdocUW/data/CS_942_to_1884.csv') 
df_CS_3 = pd.read_csv('/Users/Documents/PostdocUW/data/CS_1884_to_2827.csv') 

df_CS1 = pd.DataFrame()
df_CS1 = df_CS1.append(df_CS_1);df_CS1 = df_CS1.append(df_CS_2);df_CS1 = df_CS1.append(df_CS_3)
df_CS1 = df_CS1.drop(columns = ['count'])
df_CS1 = df_CS1.drop(columns = ['Unnamed: 0'])

df_CS_new = pd.read_csv('/Users/Documents/PostdocUW/datanew/Data_CS.csv') 
df_CS_new = pd.DataFrame({'date':df_CS_new['date'],'WA_TLT':df_CS_new['TLT_WA'],
                          'OR_TLT': df_CS_new['TLT_OR'],'NCA_TLT':df_CS_new['TLT_NCA'],
                          'SCA_TLT':df_CS_new['TLT_SCA'],'WA_LS':df_CS_new['WA_LS'],
                          'OR_LS':df_CS_new['WA_OR'],'NCA_LS':df_CS_new['WA_NCA'],
                          'SCA_LS':df_CS_new['WA_SCA'],'WA_CS':df_CS_new['CS_WA'],
                          'OR_CS':df_CS_new['CS_OR'],'NCA_CS':df_CS_new['CS_NCA'],
                          'SCA_CS':df_CS_new['CS_SCA']})
df_CS1['date'] = pd.to_datetime(df_CS1['date'], format = '%Y%m%d')
df_CS_new['date'] = pd.to_datetime(df_CS_new['date'], format = '%Y-%m-%d')
df_CS = df_CS_new.append(df_CS1)
########################################################################################
# Bottom layer temperature for spawning condition 10km off-shore
df_BLT_LSold = pd.read_csv('/Users/Documents/PostdocUW/data/BLT.csv')  
df_BLT_LSnew = pd.read_csv('/Users/Documents/PostdocUW/datanew/Data_BLT_LS.csv') 
df_BLT_LSold = df_BLT_LSold.drop(columns = ['Unnamed: 0'])
df_BLT_LSnew = df_BLT_LSnew.drop(columns = ['Unnamed: 0'])
df_CS1 = df_CS1.reset_index()
df_BLT_LSold = pd.DataFrame({'date':df_CS1['date'],'BLT_WA': df_BLT_LSold['WA'],'BLT_OR': df_BLT_LSold['OR'],'BLT_NCA': df_BLT_LSold['NCA'],'BLT_SCA': df_BLT_LSold['SCA']})
df_BLT_spwn = pd.DataFrame()
df_BLT_spwn = df_BLT_spwn.append(df_BLT_LSnew); df_BLT_spwn = df_BLT_spwn.append(df_BLT_LSold)
df_BLT_spwn['date'] = pd.to_datetime(df_BLT_spwn['date'], format = '%Y-%m-%d')
df_BLT_spwn = df_BLT_spwn.sort_values(by=['date'], ascending=False)
df_BLT_spwn = df_BLT_spwn.reset_index()
df_BLT_spwn = df_BLT_spwn.drop(columns = ['index'])
########################################################################################
# Bottom layer temperature for female conditioning on the continental shielf
df_BLT_CS_1 = pd.read_csv('/Users/Documents/PostdocUW/data/BLT_CS_0_to_706.csv') 
df_BLT_CS_2 = pd.read_csv('/Users/Documents/PostdocUW/data/BLT_CS_706_to_1412.csv') 
df_BLT_CS_3 = pd.read_csv('/Users/Documents/PostdocUW/data/BLT_CS_1412_to_2118.csv') 
df_BLT_CS_4 = pd.read_csv('/Users/Documents/PostdocUW/data/BLT_CS_2118_to_2827.csv') 

df_BLT_CS_1 = df_BLT_CS_1.drop(columns = ['Unnamed: 0', 'count', 'date'])
df_BLT_CS_2 = df_BLT_CS_2.drop(columns = ['Unnamed: 0', 'count', 'date'])
df_BLT_CS_3 = df_BLT_CS_3.drop(columns = ['Unnamed: 0', 'count', 'date'])
df_BLT_CS_4 = df_BLT_CS_4.drop(columns = ['Unnamed: 0', 'count', 'date'])

df_BLT_femlcond1 = pd.DataFrame()
df_BLT_femlcond1 = df_BLT_femlcond1.append(df_BLT_CS_1); df_BLT_femlcond1 = df_BLT_femlcond1.append(df_BLT_CS_2);df_BLT_femlcond1 = df_BLT_femlcond1.append(df_BLT_CS_3);df_BLT_femlcond1 = df_BLT_femlcond1.append(df_BLT_CS_4)
df_BLT_femlcond1 = df_BLT_femlcond1.reset_index()
df_BLT_femlcond1 = pd.DataFrame({'date':df_BLT_LSold['date'],'BLT_WA': df_BLT_femlcond1['WA'],'BLT_OR': df_BLT_femlcond1['OR'],'BLT_NCA': df_BLT_femlcond1['NCA'],'BLT_SCA': df_BLT_femlcond1['SCA']})


df_BLT_CSnew = pd.read_csv('/Users/Documents/PostdocUW/datanew/Data_BLT_CS.csv') 
df_BLT_CSnew = df_BLT_CSnew.drop(columns = ['Unnamed: 0'])
df_BLT_CSnew = df_BLT_CSnew.sort_values(by=['date'], ascending=False)

df_BLT_femlcond = pd.DataFrame()
df_BLT_femlcond = df_BLT_femlcond.append(df_BLT_CSnew); df_BLT_femlcond = df_BLT_femlcond.append(df_BLT_femlcond1);
df_BLT_femlcond['date'] = pd.to_datetime(df_BLT_femlcond['date'], format = '%Y-%m-%d')
df_BLT_femlcond = df_BLT_femlcond.reset_index()
df_BLT_femlcond = df_BLT_femlcond.drop(columns = ['index'])
#########################################################################################

df_150_1 = pd.read_csv('/Users/Documents/PostdocUW/data/Data_150_0_to_942.csv') 
df_150_2 = pd.read_csv('/Users/Documents/PostdocUW/data/Data_150_942_to_1884.csv') 
df_150_3 = pd.read_csv('/Users/Documents/PostdocUW/data/Data_150_1884_to_2827.csv') 

df_1501 = pd.DataFrame()
df_1501 = df_1501.append(df_150_1); df_1501 = df_1501.append(df_150_2); df_1501 = df_1501.append(df_150_3);
df_1501 = df_1501.drop(columns = ['count'])
df_1501 = df_1501.drop(columns = ['Unnamed: 0'])

df_150_new = pd.read_csv('/Users/Documents/PostdocUW/datanew/Data_150.csv') 
df_150_new = pd.DataFrame({'date':df_150_new['date'],'WA_TLT':df_150_new['TLT_WA'],
                          'OR_TLT': df_150_new['TLT_OR'],'NCA_TLT':df_150_new['TLT_NCA'],
                          'SCA_TLT':df_150_new['TLT_SCA'],'WA_LS':df_150_new['WA_LS'],
                          'OR_LS':df_150_new['WA_OR'],'NCA_LS':df_150_new['WA_NCA'],
                          'SCA_LS':df_150_new['WA_SCA'],'WA_CS':df_150_new['CS_WA'],
                          'OR_CS':df_150_new['CS_OR'],'NCA_CS':df_150_new['CS_NCA'],
                          'SCA_CS':df_150_new['CS_SCA']})
df_1501['date'] = pd.to_datetime(df_1501['date'], format = '%Y%m%d')
df_150_new['date'] = pd.to_datetime(df_150_new['date'], format = '%Y-%m-%d')
df_150 = df_150_new.append(df_1501)


df_250_1 = pd.read_csv('/Users/Documents/PostdocUW/data/Data_250_0_to_942.csv') 
df_250_2 = pd.read_csv('/Users/Documents/PostdocUW/data/Data_250_942_to_1884.csv') 
df_250_3 = pd.read_csv('/Users/Documents/PostdocUW/data/Data_250_1884_to_2827.csv') 

df_2501 = pd.DataFrame()
df_2501 = df_2501.append(df_250_1); df_2501 = df_2501.append(df_250_2); df_2501 = df_2501.append(df_250_3);
df_2501 = df_2501.drop(columns = ['count'])
df_2501 = df_2501.drop(columns = ['Unnamed: 0'])

df_250_new = pd.read_csv('/Users/Documents/PostdocUW/datanew/Data_250.csv') 
df_250_new = pd.DataFrame({'date':df_250_new['date'],'WA_TLT':df_250_new['TLT_WA'],
                          'OR_TLT': df_250_new['TLT_OR'],'NCA_TLT':df_250_new['TLT_NCA'],
                          'SCA_TLT':df_250_new['TLT_SCA'],'WA_LS':df_250_new['WA_LS'],
                          'OR_LS':df_250_new['WA_OR'],'NCA_LS':df_250_new['WA_NCA'],
                          'SCA_LS':df_250_new['WA_SCA'],'WA_CS':df_250_new['CS_WA'],
                          'OR_CS':df_250_new['CS_OR'],'NCA_CS':df_250_new['CS_NCA'],
                          'SCA_CS':df_250_new['CS_SCA']})
df_2501['date'] = pd.to_datetime(df_2501['date'], format = '%Y%m%d')
df_250_new['date'] = pd.to_datetime(df_250_new['date'], format = '%Y-%m-%d')
df_250 = df_250_new.append(df_2501)

years_ORWA = list(range(1980,2020))
years_CA = list(range(1981,2020))
years_femcond = list(range(1980,2019))

##############################################################################################
##############################################################################################
# Variables per state or region
WA_var_name = ['date', 'meanBLT_precond','meanBLT_spwn',
               'maxTLT_z1','meanLST_z1','meanCST_z1',
               'maxTLT_z2','meanLST_z2','meanCST_z2',
               'maxTLT_z3','meanLST_z3','meanCST_z3',
               'maxTLT_z4','meanLST_z4','meanCST_z4',
               'maxTLT_z5','meanLST_z5','meanCST_z5',
               'maxTLT_mg','meanLST_mg','meanCST_mg','meanBLT_jvnl0']

# Set time range for each stage
# Washington
femlcond_WA =["-04-01","-12-31"]
Spwn_WA = ["-01-01", "-03-31"]
z1_w = ["-01-01","-04-14"]
z2_w = ["-01-14","-04-23"]
z3_w = ["-01-23","-05-06"]
z4_w = ["-02-06","-05-23"]
z5_w = ["-02-23","-06-12"]
mg_w = ["-03-12","-07-09"]
jvl_wa = ["-04-08","-12-31"]

# varibles for the WA

# female conditioning: bottom 
meanBLT_precond_WA = [np.nan]+[df_BLT_femlcond[(df_BLT_femlcond['date'] > pd.to_datetime(str(u)+femlcond_WA[0])) & (df_BLT_femlcond['date'] < pd.to_datetime(str(u)+femlcond_WA[1]))]['BLT_WA'].mean() 
     for u in years_femcond]

# Spawning bottom layer temperature
meanBLT_spwn_WA = [df_BLT_spwn[(df_BLT_spwn['date'] > pd.to_datetime(str(u)+Spwn_WA[0])) & (df_BLT_spwn['date'] < pd.to_datetime(str(u)+Spwn_WA[1]))]['BLT_WA'].mean() 
      for u in years_ORWA]
# 
meanBLT_jvnl0_WA = [df_BLT_spwn[(df_BLT_spwn['date'] > pd.to_datetime(str(u)+jvl_wa[0])) & (df_BLT_spwn['date'] < pd.to_datetime(str(u)+jvl_wa[1]))]['BLT_WA'].mean() 
      for u in years_ORWA]
# other variables
maxTLT_z1_WA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z1_w[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_w[1]))]['WA_TLT'].max() 
     for u in years_ORWA]
maxTLT_z2_WA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z2_w[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_w[1]))]['WA_TLT'].max() 
     for u in years_ORWA]
maxTLT_z3_WA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z3_w[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_w[1]))]['WA_TLT'].max() 
     for u in years_ORWA]
maxTLT_z4_WA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_w[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_w[1]))]['WA_TLT'].max() 
     for u in years_ORWA]
maxTLT_z5_WA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_w[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_w[1]))]['WA_TLT'].max() 
     for u in years_ORWA]
maxTLT_mg_WA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_w[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_w[1]))]['WA_TLT'].max() 
     for u in years_ORWA]
meanLST_z1_WA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z1_w[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_w[1]))]['WA_LS'].mean() 
     for u in years_ORWA]
meanLST_z2_WA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z2_w[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_w[1]))]['WA_LS'].mean() 
     for u in years_ORWA]
meanLST_z3_WA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z3_w[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_w[1]))]['WA_LS'].mean() 
     for u in years_ORWA]
meanLST_z4_WA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_w[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_w[1]))]['WA_LS'].mean() 
     for u in years_ORWA]
meanLST_z5_WA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_w[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_w[1]))]['WA_LS'].mean() 
     for u in years_ORWA]
meanLST_mg_WA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_w[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_w[1]))]['WA_LS'].mean() 
     for u in years_ORWA]
meanCST_z1_WA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z1_w[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_w[1]))]['WA_CS'].mean() 
     for u in years_ORWA]
meanCST_z2_WA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z2_w[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_w[1]))]['WA_CS'].mean() 
     for u in years_ORWA]
meanCST_z3_WA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z3_w[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_w[1]))]['WA_CS'].mean() 
     for u in years_ORWA]
meanCST_z4_WA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_w[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_w[1]))]['WA_CS'].mean() 
     for u in years_ORWA]
meanCST_z5_WA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_w[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_w[1]))]['WA_CS'].mean() 
     for u in years_ORWA]
meanCST_mg_WA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_w[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_w[1]))]['WA_CS'].mean() 
     for u in years_ORWA]

# Create a new DataFrame container for all the variables per state of or region

Data_newpd_WA = {'date': years_ORWA, 'meanBLT_precond':meanBLT_precond_WA, 'meanBLT_spwn':meanBLT_spwn_WA,
              'maxTLT_z1':maxTLT_z1_WA,'maxTLT_z2':maxTLT_z2_WA,'maxTLT_z3':maxTLT_z3_WA,
              'maxTLT_z4':maxTLT_z4_WA,'maxTLT_z5':maxTLT_z5_WA,'maxTLT_mg':maxTLT_mg_WA,
              'meanLST_z1':meanLST_z1_WA,'meanLST_z2':meanLST_z2_WA,'meanLST_z3':meanLST_z3_WA,
              'meanLST_z4':meanLST_z4_WA,'meanLST_z5':meanLST_z5_WA,'meanLST_mg':meanLST_mg_WA,
              'meanCST_z1':meanCST_z1_WA,'meanCST_z2':meanCST_z2_WA,'meanCST_z3':meanCST_z3_WA,
               'meanCST_z4':meanCST_z4_WA,'meanCST_z5':meanCST_z5_WA,'meanCST_mg':meanCST_mg_WA,
               'meanBLT_jvnl0':meanBLT_jvnl0_WA}

df_var_WA = pd.DataFrame(Data_newpd_WA)

##############################################################################################
##############################################################################################
# Oregon

OR_var_name = ['date', 'meanBLT_precond','meanBLT_spwn',
               'maxTLT_z1','meanLST_z1','meanCST_z1',
               'maxTLT_z2','meanLST_z2','meanCST_z2',
               'maxTLT_z3','meanLST_z3','meanCST_z3',
               'maxTLT_z4','meanLST_z4','meanCST_z4',
               'maxTLT_z5','meanLST_z5','meanCST_z5',
               'maxTLT_mg','meanLST_mg','meanCST_mg','meanBLT_jvnl0']


femlcond_OR =["-04-01","-12-31"]
Spwn_OR = ["-01-01", "-03-31"]
z1_o = ["-01-01","-04-14"]
z2_o = ["-01-14","-04-23"]
z3_o = ["-01-23","-05-06"]
z4_o = ["-02-06","-05-23"]
z5_o = ["-02-23","-06-12"]
mg_o = ["-03-12","-07-09"]
jvl_o = ["-04-08","-12-31"]

# varibles for the OR
# female conditioning 
meanBLT_precond_OR = [np.nan] + [df_BLT_femlcond[(df_BLT_femlcond['date'] > pd.to_datetime(str(u)+femlcond_OR[0])) & (df_BLT_femlcond['date'] < pd.to_datetime(str(u)+femlcond_OR[1]))]['BLT_OR'].mean() 
     for u in years_femcond]

# Spawning bottom layer temperature
meanBLT_spwn_OR = [df_BLT_spwn[(df_BLT_spwn['date'] > pd.to_datetime(str(u)+Spwn_OR[0])) & (df_BLT_spwn['date'] < pd.to_datetime(str(u)+Spwn_OR[1]))]['BLT_OR'].mean() 
      for u in years_ORWA]
# 
meanBLT_jvnl0_OR = [df_BLT_spwn[(df_BLT_spwn['date'] > pd.to_datetime(str(u)+jvl_o[0])) & (df_BLT_spwn['date'] < pd.to_datetime(str(u)+jvl_o[1]))]['BLT_OR'].mean() 
      for u in years_ORWA]

# other variables
maxTLT_z1_OR = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z1_o[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_o[1]))]['OR_TLT'].max() 
     for u in years_ORWA]
maxTLT_z2_OR = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z2_o[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_o[1]))]['OR_TLT'].max() 
     for u in years_ORWA]
maxTLT_z3_OR = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z3_o[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_o[1]))]['OR_TLT'].max() 
     for u in years_ORWA]
maxTLT_z4_OR = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_o[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_o[1]))]['OR_TLT'].max() 
     for u in years_ORWA]
maxTLT_z5_OR = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_o[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_o[1]))]['OR_TLT'].max() 
     for u in years_ORWA]
maxTLT_mg_OR = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_o[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_o[1]))]['OR_TLT'].max() 
     for u in years_ORWA]
meanLST_z1_OR = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z1_o[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_o[1]))]['OR_LS'].mean() 
     for u in years_ORWA]
meanLST_z2_OR = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z2_o[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_o[1]))]['OR_LS'].mean() 
     for u in years_ORWA]
meanLST_z3_OR = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z3_o[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_o[1]))]['OR_LS'].mean() 
     for u in years_ORWA]
meanLST_z4_OR = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_o[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_o[1]))]['OR_LS'].mean() 
     for u in years_ORWA]
meanLST_z5_OR = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_o[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_o[1]))]['OR_LS'].mean() 
     for u in years_ORWA]
meanLST_mg_OR = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_o[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_o[1]))]['OR_LS'].mean() 
     for u in years_ORWA]
meanCST_z1_OR = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z1_o[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_o[1]))]['OR_CS'].mean() 
     for u in years_ORWA]
meanCST_z2_OR = [df_CS[(df_CS['date'] > pd.to_datetime(str(u)+z2_o[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_o[1]))]['OR_CS'].mean() 
     for u in years_ORWA]
meanCST_z3_OR = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z3_o[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_o[1]))]['OR_CS'].mean() 
     for u in years_ORWA]
meanCST_z4_OR = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_o[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_o[1]))]['OR_CS'].mean() 
     for u in years_ORWA]
meanCST_z5_OR = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_o[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_o[1]))]['OR_CS'].mean() 
     for u in years_ORWA]
meanCST_mg_OR = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_o[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_o[1]))]['OR_CS'].mean() 
     for u in years_ORWA]

Data_newpd_OR = {'date': years_ORWA, 'meanBLT_precond':meanBLT_precond_OR,'meanBLT_spwn':meanBLT_spwn_OR,
              'maxTLT_z1':maxTLT_z1_OR,'maxTLT_z2':maxTLT_z2_OR,'maxTLT_z3':maxTLT_z3_OR,
              'maxTLT_z4':maxTLT_z4_OR,'maxTLT_z5':maxTLT_z5_OR,'maxTLT_mg':maxTLT_mg_OR,
              'meanLST_z1':meanLST_z1_OR,'meanLST_z2':meanLST_z2_OR,'meanLST_z3':meanLST_z3_OR,
              'meanLST_z4':meanLST_z4_OR,'meanLST_z5':meanLST_z5_OR,'meanLST_mg':meanLST_mg_OR,
              'meanCST_z1':meanCST_z1_OR,'meanCST_z2':meanCST_z2_OR,'meanCST_z3':meanCST_z3_OR,
               'meanCST_z4':meanCST_z4_OR,'meanCST_z5':meanCST_z5_OR,'meanCST_mg':meanCST_mg_OR,
               'meanBLT_jvnl0':meanBLT_jvnl0_OR}

df_var_OR = pd.DataFrame(Data_newpd_OR)

##############################################################################################
##############################################################################################
# North CA

NCA_var_name = ['date', 'meanBLT_precond','meanBLT_spwn',
               'maxTLT_z1','meanLST_z1','meanCST_z1',
               'maxTLT_z2','meanLST_z2','meanCST_z2',
               'maxTLT_z3','meanLST_z3','meanCST_z3',
               'maxTLT_z4','meanLST_z4','meanCST_z4',
               'maxTLT_z5','meanLST_z5','meanCST_z5',
               'maxTLT_mg','meanLST_mg','meanCST_mg','meanBLT_jvnl0']

femlcond_NCA =["-04-01","-12-31"]
Spwn_NCA = ["-01-01", "-03-31"]
z1_n = ["-01-01","-04-14"]
z2_n = ["-01-14","-04-23"]
z3_n = ["-01-23","-05-06"]
z4_n = ["-02-06","-05-23"]
z5_n = ["-02-23","-06-12"]
mg_n = ["-03-12","-07-09"]
jvl_n = ["-04-08","-12-31"]
# varibles for the NCA

# female conditioning 
meanBLT_precond_NCA = [np.nan] + [df_BLT_femlcond[(df_BLT_femlcond['date'] > pd.to_datetime(str(u)+femlcond_NCA[0])) & (df_BLT_femlcond['date'] < pd.to_datetime(str(u)+femlcond_NCA[1]))]['BLT_NCA'].mean() 
     for u in years_femcond]

# Spawning bottom layer temperature
meanBLT_spwn_NCA = [df_BLT_spwn[(df_BLT_spwn['date'] > pd.to_datetime(str(u-1)+Spwn_NCA[0])) & (df_BLT_spwn['date'] < pd.to_datetime(str(u)+Spwn_NCA[1]))]['BLT_NCA'].mean() 
      for u in years_ORWA]
# 
meanBLT_jvnl0_NCA = [df_BLT_spwn[(df_BLT_spwn['date'] > pd.to_datetime(str(u)+jvl_n[0])) & (df_BLT_spwn['date'] < pd.to_datetime(str(u)+jvl_n[1]))]['BLT_NCA'].mean() 
      for u in years_ORWA]

# other variables
maxTLT_z1_NCA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z1_n[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_n[1]))]['NCA_TLT'].max() 
     for u in years_ORWA]
maxTLT_z2_NCA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z2_n[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_n[1]))]['NCA_TLT'].max() 
     for u in years_ORWA]
maxTLT_z3_NCA = [df_150[(df_150['date'] > pd.to_datetime(str(u-1)+z3_n[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_n[1]))]['NCA_TLT'].max() 
     for u in years_ORWA]
maxTLT_z4_NCA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_n[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_n[1]))]['NCA_TLT'].max() 
     for u in years_ORWA]
maxTLT_z5_NCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_n[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_n[1]))]['NCA_TLT'].max() 
     for u in years_ORWA]
maxTLT_mg_NCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_n[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_n[1]))]['NCA_TLT'].max() 
     for u in years_ORWA]
meanLST_z1_NCA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z1_n[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_n[1]))]['NCA_LS'].mean() 
     for u in years_ORWA]
meanLST_z2_NCA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z2_n[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_n[1]))]['NCA_LS'].mean() 
     for u in years_ORWA]
meanLST_z3_NCA = [df_150[(df_150['date'] > pd.to_datetime(str(u-1)+z3_n[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_n[1]))]['NCA_LS'].mean() 
     for u in years_ORWA]
meanLST_z4_NCA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_n[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_n[1]))]['NCA_LS'].mean() 
     for u in years_ORWA]
meanLST_z5_NCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_n[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_n[1]))]['NCA_LS'].mean() 
     for u in years_ORWA]
meanLST_mg_NCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_n[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_n[1]))]['NCA_LS'].mean() 
     for u in years_ORWA]
meanCST_z1_NCA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z1_n[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_n[1]))]['NCA_CS'].mean() 
     for u in years_ORWA]
meanCST_z2_NCA = [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z2_n[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_n[1]))]['NCA_CS'].mean() 
     for u in years_ORWA]
meanCST_z3_NCA = [df_150[(df_150['date'] > pd.to_datetime(str(u-1)+z3_n[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_n[1]))]['NCA_CS'].mean() 
     for u in years_ORWA]
meanCST_z4_NCA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_n[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_n[1]))]['NCA_CS'].mean() 
     for u in years_ORWA]
meanCST_z5_NCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_n[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_n[1]))]['NCA_CS'].mean() 
     for u in years_ORWA]
meanCST_mg_NCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_n[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_n[1]))]['NCA_CS'].mean() 
     for u in years_ORWA]

Data_newpd_NCA = {'date': years_ORWA, 'meanBLT_precond':meanBLT_precond_NCA,'meanBLT_spwn':meanBLT_spwn_NCA,
              'maxTLT_z1':maxTLT_z1_NCA,'maxTLT_z2':maxTLT_z2_NCA,'maxTLT_z3':maxTLT_z3_NCA,
              'maxTLT_z4':maxTLT_z4_NCA,'maxTLT_z5':maxTLT_z5_NCA,'maxTLT_mg':maxTLT_mg_NCA,
              'meanLST_z1':meanLST_z1_NCA,'meanLST_z2':meanLST_z2_NCA,'meanLST_z3':meanLST_z3_NCA,
              'meanLST_z4':meanLST_z4_NCA,'meanLST_z5':meanLST_z5_NCA,'meanLST_mg':meanLST_mg_NCA,
              'meanCST_z1':meanCST_z1_NCA,'meanCST_z2':meanCST_z2_NCA,'meanCST_z3':meanCST_z3_NCA,
               'meanCST_z4':meanCST_z4_NCA,'meanCST_z5':meanCST_z5_NCA,'meanCST_mg':meanCST_mg_NCA,
               'meanBLT_jvnl0':meanBLT_jvnl0_NCA}

df_var_NCA = pd.DataFrame(Data_newpd_NCA)

##############################################################################################
##############################################################################################
# Central CA

SCA_var_name = ['date','meanBLT_precond','meanBLT_spwn',
               'maxTLT_z1','meanLST_z1','meanCST_z1',
               'maxTLT_z2','meanLST_z2','meanCST_z2',
               'maxTLT_z3','meanLST_z3','meanCST_z3',
               'maxTLT_z4','meanLST_z4','meanCST_z4',
               'maxTLT_z5','meanLST_z5','meanCST_z5',
               'maxTLT_mg','meanLST_mg','meanCST_mg','meanBLT_jvnl0']

femlcond_SCA =["-03-01","-11-30"]
Spwn_SCA = ["-12-01", "-02-28"]
z1_s = ["-12-01","-03-14"]
z2_s = ["-12-14","-03-23"]
z3_s = ["-12-23","-04-06"]
z4_s = ["-01-06","-04-23"]
z5_s = ["-01-23","-05-12"]
mg_s = ["-02-12","-06-09"]
jvl_s = ["-03-11","-12-31"]
# varibles for the SCA

# female conditioning 
meanBLT_precond_SCA = [np.nan] + [df_BLT_femlcond[(df_BLT_femlcond['date'] > pd.to_datetime(str(u)+femlcond_SCA[0])) & (df_BLT_femlcond['date'] < pd.to_datetime(str(u)+femlcond_SCA[1]))]['BLT_SCA'].mean() 
     for u in years_femcond]

# Spawning bottom layer temperature
meanBLT_spwn_SCA = [np.nan] + [df_BLT_spwn[(df_BLT_spwn['date'] > pd.to_datetime(str(u-1)+Spwn_SCA[0])) & (df_BLT_spwn['date'] < pd.to_datetime(str(u)+Spwn_SCA[1]))]['BLT_SCA'].mean() 
      for u in years_CA]
# 
meanBLT_jvnl0_SCA = [df_BLT_spwn[(df_BLT_spwn['date'] > pd.to_datetime(str(u)+jvl_s[0])) & (df_BLT_spwn['date'] < pd.to_datetime(str(u)+jvl_s[1]))]['BLT_SCA'].mean() 
      for u in years_ORWA]
# other variables
maxTLT_z1_SCA = [np.nan] + [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z1_s[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_s[1]))]['SCA_TLT'].max() 
     for u in years_CA]
maxTLT_z2_SCA = [np.nan] + [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z2_s[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_s[1]))]['SCA_TLT'].max() 
     for u in years_CA]
maxTLT_z3_SCA = [np.nan] + [df_150[(df_150['date'] > pd.to_datetime(str(u-1)+z3_s[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_s[1]))]['SCA_TLT'].max() 
     for u in years_CA]
maxTLT_z4_SCA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_s[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_s[1]))]['SCA_TLT'].max() 
     for u in years_ORWA]
maxTLT_z5_SCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_s[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_s[1]))]['SCA_TLT'].max() 
     for u in years_ORWA]
maxTLT_mg_SCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_s[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_s[1]))]['SCA_TLT'].max() 
     for u in years_ORWA]
meanLST_z1_SCA = [np.nan] + [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z1_s[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_s[1]))]['SCA_LS'].mean() 
     for u in years_CA]
meanLST_z2_SCA = [np.nan] + [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z2_s[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_s[1]))]['SCA_LS'].mean() 
     for u in years_CA]
meanLST_z3_SCA = [np.nan] + [df_150[(df_150['date'] > pd.to_datetime(str(u-1)+z3_s[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_s[1]))]['SCA_LS'].mean() 
     for u in years_CA]
meanLST_z4_SCA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_s[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_s[1]))]['SCA_LS'].mean() 
     for u in years_ORWA]
meanLST_z5_SCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_s[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_s[1]))]['SCA_LS'].mean() 
     for u in years_ORWA]
meanLST_mg_SCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_s[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_s[1]))]['SCA_LS'].mean() 
     for u in years_ORWA]
meanCST_z1_SCA = [np.nan] + [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z1_s[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z1_s[1]))]['SCA_CS'].mean() 
     for u in years_CA]
meanCST_z2_SCA = [np.nan] + [df_CS[(df_CS['date'] > pd.to_datetime(str(u-1)+z2_s[0])) & (df_CS['date'] < pd.to_datetime(str(u)+z2_s[1]))]['SCA_CS'].mean() 
     for u in years_CA]
meanCST_z3_SCA = [np.nan] + [df_150[(df_150['date'] > pd.to_datetime(str(u-1)+z3_s[0])) & (df_150['date'] < pd.to_datetime(str(u)+z3_s[1]))]['SCA_CS'].mean() 
     for u in years_CA]
meanCST_z4_SCA = [df_150[(df_150['date'] > pd.to_datetime(str(u)+z4_s[0])) & (df_150['date'] < pd.to_datetime(str(u)+z4_s[1]))]['SCA_CS'].mean() 
     for u in years_ORWA]
meanCST_z5_SCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+z5_s[0])) & (df_250['date'] < pd.to_datetime(str(u)+z5_s[1]))]['SCA_CS'].mean() 
     for u in years_ORWA]
meanCST_mg_SCA = [df_250[(df_250['date'] > pd.to_datetime(str(u)+mg_s[0])) & (df_250['date'] < pd.to_datetime(str(u)+mg_s[1]))]['SCA_CS'].mean() 
     for u in years_ORWA]


Data_newpd_SCA = {'date': years_ORWA, 'meanBLT_precond':meanBLT_precond_SCA,'meanBLT_spwn':meanBLT_spwn_SCA,
              'maxTLT_z1':maxTLT_z1_SCA,'maxTLT_z2':maxTLT_z2_SCA,'maxTLT_z3':maxTLT_z3_SCA,
              'maxTLT_z4':maxTLT_z4_SCA,'maxTLT_z5':maxTLT_z5_SCA,'maxTLT_mg':maxTLT_mg_SCA,
              'meanLST_z1':meanLST_z1_SCA,'meanLST_z2':meanLST_z2_SCA,'meanLST_z3':meanLST_z3_SCA,
              'meanLST_z4':meanLST_z4_SCA,'meanLST_z5':meanLST_z5_SCA,'meanLST_mg':meanLST_mg_SCA,
              'meanCST_z1':meanCST_z1_SCA,'meanCST_z2':meanCST_z2_SCA,'meanCST_z3':meanCST_z3_SCA,
               'meanCST_z4':meanCST_z4_SCA,'meanCST_z5':meanCST_z5_SCA,'meanCST_mg':meanCST_mg_SCA,
               'meanBLT_jvnl0':meanBLT_jvnl0_SCA}

df_var_SCA = pd.DataFrame(Data_newpd_SCA)


# X_WA = df_var_WA[(df_var_WA['date']<=2012) & (df_var_WA['date']>=1980)][WA_var_name[0:]]
# X_WA['meanBLT_precond'][31] = np.nan; X_WA['meanBLT_precond'][32] = np.nan
# X_OR = df_var_OR[(df_var_OR['date']<=2012) & (df_var_OR['date']>=1980)][OR_var_name[0:]]
# X_OR['meanBLT_precond'][31] = np.nan;X_OR['meanBLT_precond'][32] = np.nan
# X_NCA = df_var_NCA[(df_var_NCA['date']<=2012) & (df_var_NCA['date']>=1980)][NCA_var_name[0:]]
# X_NCA['meanBLT_precond'][31] = np.nan;X_NCA['meanBLT_precond'][32] = np.nan
# X_SCA = df_var_SCA[(df_var_SCA['date']<=2012) & (df_var_SCA['date']>=1980)][SCA_var_name[0:]]
# X_SCA['meanBLT_precond'][31] = np.nan;X_SCA['meanBLT_precond'][32] = np.nan

X_WA = df_var_WA[WA_var_name[0:]]
#X_WA['meanBLT_precond'][31] = np.nan; X_WA['meanBLT_precond'][32] = np.nan
X_OR = df_var_OR[OR_var_name[0:]]
#X_OR['meanBLT_precond'][31] = np.nan;X_OR['meanBLT_precond'][32] = np.nan
X_NCA = df_var_NCA[NCA_var_name[0:]]
#X_NCA['meanBLT_precond'][31] = np.nan;X_NCA['meanBLT_precond'][32] = np.nan
X_SCA = df_var_SCA[SCA_var_name[0:]]
#X_SCA['meanBLT_precond'][31] = np.nan;X_SCA['meanBLT_precond'][32] = np.nan

##################################################################################
Y = pd.read_csv('/Users/Documents/PostdocUW/data_roms/crab_model_results_2020422.csv') 

Y_WA = pd.DataFrame({'date':Y[(Y['area']=='WA') & (Y['season']<=2016) & (Y['season']>=1984)]['season'],'Y':Y[(Y['area']=='WA') & (Y['season']<=2016) & (Y['season']>=1984)]['mean_est_thousands_mt'],'Y_SD':Y[(Y['area']=='WA') & (Y['season']<=2016) & (Y['season']>=1984)]['est_sd']})
Y_WA = Y_WA.reset_index()
Y_WA = Y_WA.drop(columns = ['index'])
Y_OR = pd.DataFrame({'date':Y[(Y['area']=='OR') & (Y['season']<=2016) & (Y['season']>=1984)]['season'],'Y':Y[(Y['area']=='OR') & (Y['season']<=2016) & (Y['season']>=1984)]['mean_est_thousands_mt'],'Y_SD':Y[(Y['area']=='OR') & (Y['season']<=2016) & (Y['season']>=1984)]['est_sd']})
Y_OR = Y_OR.reset_index()
Y_OR = Y_OR.drop(columns = ['index'])
Y_NCA = pd.DataFrame({'date':Y[(Y['area']=='North CA') & (Y['season']<=2016) & (Y['season']>=1984)]['season'],'Y':Y[(Y['area']=='North CA') & (Y['season']<=2016) & (Y['season']>=1984)]['mean_est_thousands_mt'],'Y_SD':Y[(Y['area']=='North CA') & (Y['season']<=2016) & (Y['season']>=1984)]['est_sd']})
Y_NCA = Y_NCA.reset_index()
Y_NCA = Y_NCA.drop(columns = ['index'])
Y_SCA = pd.DataFrame({'date':Y[(Y['area']=='Central CA') & (Y['season']<=2016) & (Y['season']>=1984)]['season'],'Y':Y[(Y['area']=='Central CA') & (Y['season']<=2016) & (Y['season']>=1984)]['mean_est_thousands_mt'],'Y_SD':Y[(Y['area']=='Central CA') & (Y['season']<=2016) & (Y['season']>=1984)]['est_sd']})
Y_SCA = Y_SCA.reset_index()
Y_SCA = Y_SCA.drop(columns = ['index'])

##########################
# to joint between X and Y I need to index both using time "date"
X_WA  = X_WA.set_index('date')
X_OR  = X_OR.set_index('date')
X_NCA = X_NCA.set_index('date')
X_SCA = X_SCA.set_index('date')
Y_WA  = Y_WA.set_index('date')
Y_OR  = Y_OR.set_index('date')
Y_NCA  = Y_NCA.set_index('date')
Y_SCA  = Y_SCA.set_index('date')
###############################
# now Join both

X_WA = X_WA.join(Y_WA, how='left')
X_OR = X_OR.join(Y_OR, how='left')
X_NCA = X_NCA.join(Y_NCA, how='left')
X_SCA = X_SCA.join(Y_SCA, how='left')

#################################################################################

X_WA.to_csv("/Users/Documents/PostdocUW/proc_data/X_WA.csv", )
X_OR.to_csv("/Users/Documents/PostdocUW/proc_data/X_OR.csv", )
X_NCA.to_csv("/Users/Documents/PostdocUW/proc_data/X_NCA.csv", )
X_SCA.to_csv("/Users/Documents/PostdocUW/proc_data/X_SCA.csv", )


# Histogram of response variable

