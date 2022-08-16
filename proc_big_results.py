#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 08:00:34 2021

@author: N
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

#################################################################################################
#################################################################################################
####
#### WASHINGTON
#################################################################################################
#################################################################################################

with open("/Users/rbani20/Documents/PostdocUW/proc_data/comb_var_wa.txt", "rb") as fp:
        comb_var_wa = pickle.load(fp)

Results1 = pd.DataFrame(np.zeros((len(comb_var_wa),4)), columns = ['R2_train', 'R2_test', 'count','Nvar'])

    
file  = '/Users/rbani20/Documents/PostdocUW/proc_data/WA_results_1001.csv'

with open(file) as fl:
    chunk_iter = pd.read_csv(fl, chunksize = len(comb_var_wa))
    for chunk in chunk_iter:
        chunk1 = chunk[['R2_train','R2_test']]
        chunk1[chunk1<0] = 0
        Results1['R2_test']  += chunk1['R2_test'].tolist()
        Results1['R2_train'] += chunk1['R2_train'].tolist()

        
        # code to count how many times the model would appear if my seletion process applied
        chunk2 = chunk[chunk['DeltaAICc']<2]
        chunk2[chunk2<0] = 0
        chunk2 = chunk2.sort_values('DeltaR2',ascending=False)
        chunk2 = chunk2.reset_index()
        if len(chunk2)>5:
            chunk2 = chunk2.loc[0:5,]
        
        chunk2 = chunk2.reset_index()
        for i in range(len(chunk2)):
            Results1['count'][chunk2['modelid'][i]] += 1
for i in Results1.index:
    Results1['Nvar'][i]=len(comb_var_wa[i])  

Results1['R2_train'] = (1/1000)*Results1['R2_train'] 
Results1['R2_test'] = (1/1000)*Results1['R2_test']
Results1['count']    =10*Results1['count']      

Results1 = Results1.sort_values('count',ascending=False)

br_wa = np.arange(len(Results1))

#################################################################################################
#################################################################################################
####
#### Oregon
#################################################################################################
#################################################################################################

with open("/Users/rbani20/Documents/PostdocUW/proc_data/comb_var_or.txt", "rb") as fp:
        comb_var_or = pickle.load(fp)

Results2 = pd.DataFrame(np.zeros((len(comb_var_or),4)), columns = ['R2_train', 'R2_test', 'count','Nvar'])

    
file  = '/Users/rbani20/Documents/PostdocUW/proc_data/OR_results_1004.csv'

with open(file) as fl:
    chunk_iter = pd.read_csv(fl, chunksize = len(comb_var_or))
    for chunk in chunk_iter:
        chunk1 = chunk[['R2_train','R2_test']]
        chunk1[chunk1<0] = 0
        Results2['R2_test']  += chunk1['R2_test'].tolist()
        Results2['R2_train'] += chunk1['R2_train'].tolist()

        
        # code to count how many times the model would appear if my seletion process applied
        chunk2 = chunk[chunk['DeltaAICc']<2]
        chunk2[chunk2<0] = 0
        chunk2 = chunk2.sort_values('DeltaR2',ascending=False)
        chunk2 = chunk2.reset_index()
        if len(chunk2)>5:
            chunk2 = chunk2.loc[0:5,]
        
        chunk2 = chunk2.reset_index()
        for i in range(len(chunk2)):
            Results2['count'][chunk2['modelid'][i]] += 1
for i in Results2.index:
    Results2['Nvar'][i]=len(comb_var_or[i])  

Results2['R2_train'] = (1/1000)*Results2['R2_train'] 
Results2['R2_test'] = (1/1000)*Results2['R2_test']
Results2['count']    =10*Results2['count']      

Results2 = Results2.sort_values('count',ascending=False)

br_or = np.arange(len(Results2))

#################################################################################################
#################################################################################################
####
#### Northern California
#################################################################################################
#################################################################################################

with open("/Users/rbani20/Documents/PostdocUW/proc_data/comb_var_nca.txt", "rb") as fp:
        comb_var_nca = pickle.load(fp)

Results3 = pd.DataFrame(np.zeros((len(comb_var_nca),4)), columns = ['R2_train', 'R2_test', 'count','Nvar'])

    
file  = '/Users/rbani20/Documents/PostdocUW/proc_data/NCA_results_1002.csv'

with open(file) as fl:
    chunk_iter = pd.read_csv(fl, chunksize = len(comb_var_nca))
    for chunk in chunk_iter:
        chunk1 = chunk[['R2_train','R2_test']]
        chunk1[chunk1<0] = 0
        Results3['R2_test']  += chunk1['R2_test'].tolist()
        Results3['R2_train'] += chunk1['R2_train'].tolist()

        
        # code to count how many times the model would appear if my seletion process applied
        chunk2 = chunk[chunk['DeltaAICc']<2]
        chunk2[chunk2<0] = 0
        chunk2 = chunk2.sort_values('DeltaR2',ascending=False)
        chunk2 = chunk2.reset_index()
        if len(chunk2)>5:
            chunk2 = chunk2.loc[0:5,]
        
        chunk2 = chunk2.reset_index()
        for i in range(len(chunk2)):
            Results3['count'][chunk2['modelid'][i]] += 1
for i in Results3.index:
    Results3['Nvar'][i]=len(comb_var_nca[i])  

Results3['R2_train'] = (1/1000)*Results3['R2_train'] 
Results3['R2_test'] = (1/1000)*Results3['R2_test']
Results3['count']    =10*Results3['count']      

Results3 = Results3.sort_values('count',ascending=False)

br_nca = np.arange(len(Results3))

#################################################################################################
#################################################################################################
####
#### Central California
#################################################################################################
#################################################################################################

with open("/Users/rbani20/Documents/PostdocUW/proc_data/comb_var_sca.txt", "rb") as fp:
        comb_var_sca = pickle.load(fp)

Results4 = pd.DataFrame(np.zeros((len(comb_var_sca),4)), columns = ['R2_train', 'R2_test', 'count','Nvar'])

    
file  = '/Users/rbani20/Documents/PostdocUW/proc_data/SCA_results_1004.csv'

with open(file) as fl:
    chunk_iter = pd.read_csv(fl, chunksize = len(comb_var_sca))
    for chunk in chunk_iter:
        chunk1 = chunk[['R2_train','R2_test']]
        chunk1[chunk1<0] = 0
        Results4['R2_test']  += chunk1['R2_test'].tolist()
        Results4['R2_train'] += chunk1['R2_train'].tolist()

        
        # code to count how many times the model would appear if my seletion process applied
        chunk2 = chunk[chunk['DeltaAICc']<2]
        chunk2[chunk2<0] = 0
        chunk2 = chunk2.sort_values('DeltaR2',ascending=False)
        chunk2 = chunk2.reset_index()
        if len(chunk2)>5:
            chunk2 = chunk2.loc[0:5,]
        
        chunk2 = chunk2.reset_index()
        for i in range(len(chunk2)):
            Results4['count'][chunk2['modelid'][i]] += 1
for i in Results4.index:
    Results4['Nvar'][i]=len(comb_var_sca[i])  

Results4['R2_train'] = (1/1000)*Results4['R2_train'] 
Results4['R2_test'] = (1/1000)*Results4['R2_test'] 
Results4['count']    =10*Results4['count'] 

Results4 = Results4.sort_values('count',ascending=False)

br_sca = np.arange(len(Results4))

#################################################################################################
#################################################################################################

fig = plt.figure()

ax = plt.subplot(4,3,1)
plt.barh(br_wa, Results1['count'].values,height = 0.4, label='training')
plt.yticks(br_wa,Results1.index,fontsize = 1)
plt.ylabel("Model ID")
ax = plt.subplot(4,3,2)
plt.barh(br_wa, Results1['R2_train'].values,height = 0.4, label='training')
plt.barh(br_wa+0.3, Results1['R2_test'],height = 0.4, label='testing')
plt.yticks(br_wa,Results1.index,fontsize = 1)
leg =  plt.legend(bbox_to_anchor=(0, 1), loc='upper left',fontsize = 6)
ax = plt.subplot(4,3,3)
plt.barh(br_wa, Results1['Nvar'].values,height = 0.4, label='training')
plt.yticks(br_wa,Results1.index,fontsize = 1)

ax = plt.subplot(4,3,4)
plt.barh(br_or, Results2['count'].values,height = 0.4, label='training')
plt.yticks(br_or,Results2.index,fontsize = 1)
plt.ylabel("Model ID")
ax = plt.subplot(4,3,5)
plt.barh(br_or, Results2['R2_train'].values,height = 0.4, label='training')
plt.barh(br_or+0.3, Results2['R2_test'],height = 0.4, label='testing')
plt.yticks(br_or,Results2.index,fontsize = 1)
ax = plt.subplot(4,3,6)
plt.barh(br_or, Results2['Nvar'].values,height = 0.4, label='training')
plt.yticks(br_or,Results2.index,fontsize = 1)


ax = plt.subplot(4,3,7)
plt.barh(br_nca, Results3['count'].values,height = 0.4, label='training')
plt.yticks(br_nca,Results3.index,fontsize = 1)
plt.ylabel("Model ID")
ax = plt.subplot(4,3,8)
plt.barh(br_nca, Results3['R2_train'].values,height = 0.4, label='training')
plt.barh(br_nca+0.3, Results3['R2_test'],height = 0.4, label='testing')
plt.yticks(br_nca,Results3.index,fontsize = 1)
ax = plt.subplot(4,3,9)
plt.barh(br_nca, Results3['Nvar'].values,height = 0.4, label='training')
plt.yticks(br_nca,Results3.index,fontsize = 1)


ax = plt.subplot(4,3,10)
plt.barh(br_sca, Results4['count'].values,height = 0.4, label='training')
plt.yticks(br_sca,Results4.index,fontsize = 1)
plt.ylabel("Model ID")
ax = plt.subplot(4,3,11)
plt.barh(br_sca, Results4['R2_train'].values,height = 0.4, label='training')
plt.barh(br_sca+0.3, Results4['R2_test'],height = 0.4, label='testing')
plt.yticks(br_sca,Results4.index,fontsize = 1)
plt.xlabel("Averged R2")
ax = plt.subplot(4,3,12)
plt.barh(br_sca, Results4['Nvar'].values,height = 0.4, label='training')
plt.yticks(br_sca,Results4.index,fontsize = 1)
plt.xlabel("# of variables")




fig.set_size_inches(9,16)
fig.savefig('/Users/rbani20/Documents/PostdocUW/figure/R2_freq_MS.pdf', format='pdf', dpi=1000)

#################################################################################################
#################################################################################################

fig1 = plt.figure()

ax = plt.subplot(4,3,1)
plt.barh(br_wa[:50], Results1['count'].values[:50],height = 0.4, label='training')
plt.yticks(br_wa[:50],Results1.index[:50],fontsize = 4)
plt.ylabel("Model ID")
plt.title("a)")
ax = plt.subplot(4,3,2)
plt.barh(br_wa[:50], Results1['R2_train'].values[:50],height = 0.4, label='training')
plt.barh(br_wa[:50]+0.3, Results1['R2_test'][:50],height = 0.4, label='testing')
plt.yticks(br_wa[:50],Results1.index[:50],fontsize = 4)
leg =  plt.legend(bbox_to_anchor=(0.7, 1), loc='upper left',fontsize = 6)
plt.title("e)")
ax = plt.subplot(4,3,3)
plt.barh(br_wa[:50], Results1['Nvar'].values[:50],height = 0.4, label='training')
plt.yticks(br_wa[:50],Results1.index[:50],fontsize = 4)
plt.title("i)")

ax = plt.subplot(4,3,4)
plt.barh(br_or[:50], Results2['count'].values[:50],height = 0.4, label='training')
plt.yticks(br_or[:50],Results2.index[:50],fontsize = 4)
plt.ylabel("Model ID")
plt.title("b)")
ax = plt.subplot(4,3,5)
plt.barh(br_or[:50], Results2['R2_train'].values[:50],height = 0.4, label='training')
plt.barh(br_or[:50]+0.3, Results2['R2_test'][:50],height = 0.4, label='testing')
plt.yticks(br_or[:50],Results2.index[:50],fontsize = 4)
plt.title("f)")
ax = plt.subplot(4,3,6)
plt.barh(br_or[:50], Results2['Nvar'].values[:50],height = 0.4, label='training')
plt.yticks(br_or[:50],Results2.index[:50],fontsize = 4)
plt.title("j)")


ax = plt.subplot(4,3,7)
plt.barh(br_nca[:50], Results3['count'].values[:50],height = 0.4, label='training')
plt.yticks(br_nca[:50],Results3.index[:50],fontsize = 4)
plt.ylabel("Model ID")
plt.title("c)")
ax = plt.subplot(4,3,8)
plt.barh(br_nca[:50], Results3['R2_train'].values[:50],height = 0.4, label='training')
plt.barh(br_nca[:50]+0.3, Results3['R2_test'][:50],height = 0.4, label='testing')
plt.yticks(br_nca[:50],Results3.index[:50],fontsize = 4)
plt.title("g)")
ax = plt.subplot(4,3,9)
plt.barh(br_nca[:50], Results3['Nvar'].values[:50],height = 0.4, label='training')
plt.yticks(br_nca[:50],Results3.index[:50],fontsize = 4)
plt.title("k)")


ax = plt.subplot(4,3,10)
plt.barh(br_sca[:50], Results4['count'].values[:50],height = 0.4, label='training')
plt.yticks(br_sca[:50],Results4.index[:50],fontsize = 4)
plt.ylabel("Model ID")
plt.xlabel("count")
plt.title("d)")
ax = plt.subplot(4,3,11)
plt.barh(br_sca[:50], Results4['R2_train'].values[:50],height = 0.4, label='training')
plt.barh(br_sca[:50]+0.3, Results4['R2_test'][:50],height = 0.4, label='testing')
plt.yticks(br_sca[:50],Results4.index[:50],fontsize = 4)
plt.title("h)")
plt.xlabel("Averaged R2")
ax = plt.subplot(4,3,12)
plt.barh(br_sca[:50], Results4['Nvar'].values[:50],height = 0.4, label='training')
plt.yticks(br_sca[:50],Results4.index[:50],fontsize = 4)
plt.xlabel("# of variables")
plt.title("l)")




fig1.set_size_inches(9,16)
fig1.savefig('/Users/rbani20/Documents/PostdocUW/figure/R2_freq_MS_20.pdf', format='pdf', dpi=1000)

