#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 15:29:02 2021

@author: N
"""

import pandas as pd
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import pickle
import statsmodels.api as sm
from pandas import read_csv



#################################################################################################
#################################################################################################
####
#### WASHINGTON
#################################################################################################
#################################################################################################

WA  = read_csv("/Users/Documents/PostdocUW/proc_data/WA_results_1001.csv")


with open("/Users/documents/postdocuw/proc_data/comb_var_wa.txt", "rb") as fp:
        comb_var_wa = pickle.load(fp)

best_var_wa = WA[(WA['R2_train']>0) & (WA['R2_test']>0)]

best_var_wa = WA.reset_index()

uni_wa = np.unique(best_var_wa['modelid'])

count_modelid_wa = pd.DataFrame(0, index = [i for i in uni_wa], columns = ["count","Nvar"])

count_mean_pergroup_wa = best_var_wa.groupby('modelid').mean()


for i in best_var_wa['modelid']:
    for j in uni_wa:
        if i == j:
            count_modelid_wa['count'][i] += 1
        
for i in count_modelid_wa.index:
    count_modelid_wa['Nvar'][i]=len(comb_var_wa[int(i)])
    
count_modelid_wa['R2_train'] = count_mean_pergroup_wa['R2_train']
count_modelid_wa['R2_test'] = count_mean_pergroup_wa['R2_test']

count_modelid_wa = count_modelid_wa.sort_values('count', ascending=False)

br_wa = np.arange(len(count_modelid_wa))

best_models_wa = [int(i) for i in count_modelid_wa['count'][:20].index.tolist()]
best_models_var_wa = [ comb_var_wa[i] for i in best_models_wa]
Best_models_var_wa = pd.DataFrame(best_models_var_wa, index = best_models_wa)

Best_models_var_wa.columns = ["variable "+ str(i+1) for i in range(len(Best_models_var_wa.columns))]
Best_models_var_wa.index =[str(i) for i in range(1,len(best_models_wa)+1)]
Best_models_var_wa['Model ID'] = best_models_wa

Best_models_var_wa.to_csv("/Users/Documents/PostdocUW/proc_data/Best_models_var_WA.csv", )

# i this section i will try to fit top 6 models and return data of coeff, Aicc, R2 in a table

X_WA = read_csv("/Users/Documents/PostdocUW/proc_data/X_WA.csv")
list_col = list(X_WA)

Xt1 = X_WA[list_col[1:len(list_col)-2]]
Xt4 = X_WA[list_col[len(list_col)-2:]]

Xt1 = Xt1[1:31].reset_index()
Xt4 = Xt4[5:35].reset_index()

Xt1 = Xt1.drop(columns = ['index'])
Xt4 = Xt4.drop(columns = ['index'])

X = Xt1.join(Xt4, how='left')

Results = pd.DataFrame([])
for i in best_models_wa:
    
    X_in                  = sm.add_constant(X[list(comb_var_wa[i])])
    Y_log_in              = np.log(X['Y'])
    
    model                 = sm.OLS(Y_log_in , X_in).fit()
    
    res1                  = pd.DataFrame(model.params).T
    res1["R2"]      = model.rsquared
    res1['AICc']          = sm.tools.eval_measures.aicc(model.llf,model.nobs,len(model.params))
    res1['Model ID']       = i
    
    Results = Results.append(res1)
    
 
Results = Results[['const']+list(set(Results.columns.tolist()) -set(['const', 'AICc', 'R2'])) + ['R2', 'AICc']]
Results = Results.set_index('Model ID')
Results_wa = Results

Results_wa.to_csv("/Users/Documents/PostdocUW/proc_data/Model_table20_WA.csv", )
#################################################################################################
#################################################################################################
####
#### Oregon
#################################################################################################
#################################################################################################

OR  = read_csv("/Users/Documents/PostdocUW/proc_data/OR_results_1000.csv")


with open("/Users/documents/postdocuw/proc_data/comb_var_or.txt", "rb") as fp:
        comb_var_or = pickle.load(fp)

best_var_or = OR[(OR['R2_test']>0) & (OR['R2_train']>0)]

best_var_or = best_var_or.reset_index()

uni_or = np.unique(best_var_or['modelid'])

count_modelid_or = pd.DataFrame(0, index = [i for i in uni_or], columns = ["count","Nvar"])

count_mean_pergroup_or = best_var_or.groupby('modelid').mean()


for i in best_var_or['modelid']:
    for j in uni_or:
        if i == j:
            count_modelid_or['count'][i] += 1
        
for i in count_modelid_or.index:
    count_modelid_or['Nvar'][i]=len(comb_var_or[int(i)])
    
count_modelid_or['R2_train'] = count_mean_pergroup_or['R2_train']
count_modelid_or['R2_test'] = count_mean_pergroup_or['R2_test']

count_modelid_or = count_modelid_or.sort_values('count', ascending=False)

br_or = np.arange(len(count_modelid_or))

best_models_or = [int(i) for i in count_modelid_or['count'][:20].index.tolist()]
best_models_var_or = [ comb_var_or[i] for i in best_models_or]
Best_models_var_or = pd.DataFrame(best_models_var_or, index = best_models_or)

Best_models_var_or.columns = ["variable "+ str(i+1) for i in range(len(Best_models_var_or.columns))]
Best_models_var_or.index =[str(i) for i in range(1,len(best_models_or)+1)]
Best_models_var_or['Model ID'] = best_models_or

Best_models_var_or.to_csv("/Users/Documents/PostdocUW/proc_data/Best_models_var_OR.csv", )

# i this section i will try to fit top 6 models and return data of coeff, Aicc, R2 in a table

X_WA = read_csv("/Users/Documents/PostdocUW/proc_data/X_OR.csv")
list_col = list(X_WA)

Xt1 = X_WA[list_col[1:len(list_col)-2]]
Xt4 = X_WA[list_col[len(list_col)-2:]]

Xt1 = Xt1[1:31].reset_index()
Xt4 = Xt4[5:35].reset_index()

Xt1 = Xt1.drop(columns = ['index'])
Xt4 = Xt4.drop(columns = ['index'])

X = Xt1.join(Xt4, how='left')

Results = pd.DataFrame([])
for i in best_models_or:
    
    X_in                  = sm.add_constant(X[list(comb_var_or[i])])
    Y_log_in              = np.log(X['Y'])
    
    model                 = sm.OLS(Y_log_in , X_in).fit()
    
    res1                  = pd.DataFrame(model.params).T
    res1["R2"]      = model.rsquared
    res1['AICc']          = sm.tools.eval_measures.aicc(model.llf,model.nobs,len(model.params))
    res1['Model ID']       = i
    
    Results = Results.append(res1)
    
 
Results = Results[['const']+list(set(Results.columns.tolist()) -set(['const', 'AICc', 'R2'])) + ['R2', 'AICc']]
Results = Results.set_index('Model ID')
Results_or = Results

Results_or.to_csv("/Users/Documents/PostdocUW/proc_data/Model_table20_OR.csv", )

#################################################################################################
#################################################################################################
####
#### Northern California
#################################################################################################
#################################################################################################

NCA  = read_csv("/Users/Documents/PostdocUW/proc_data/NCA_results_1000.csv")
NCA1  = read_csv("/Users/Documents/PostdocUW/proc_data/NCA_results_502.csv")
NCA = NCA.append(NCA1)

with open("/Users/documents/postdocuw/proc_data/comb_var_nca.txt", "rb") as fp:
        comb_var_nca = pickle.load(fp)

best_var_nca = NCA[(NCA['R2_test']>0) & (NCA['R2_train']>0)]

best_var_nca = best_var_nca.reset_index()

uni_nca = np.unique(best_var_nca['modelid'])

count_modelid_nca = pd.DataFrame(0, index = [i for i in uni_nca], columns = ["count","Nvar"])

count_mean_pergroup_nca = best_var_nca.groupby('modelid').mean()


for i in best_var_nca['modelid']:
    for j in uni_nca:
        if i == j:
            count_modelid_nca['count'][i] += 1
        
for i in count_modelid_nca.index:
    count_modelid_nca['Nvar'][i]=len(comb_var_nca[int(i)])
    
count_modelid_nca['R2_train'] = count_mean_pergroup_nca['R2_train']
count_modelid_nca['R2_test'] = count_mean_pergroup_nca['R2_test']

count_modelid_nca = count_modelid_nca.sort_values('count', ascending=False)

br_nca = np.arange(len(count_modelid_nca))

best_models_nca = [int(i) for i in count_modelid_nca['count'][:20].index.tolist()]
best_models_var_nca = [ comb_var_nca[i] for i in best_models_nca]
Best_models_var_nca = pd.DataFrame(best_models_var_nca, index = best_models_nca)

Best_models_var_nca.columns = ["variable "+ str(i+1) for i in range(len(Best_models_var_nca.columns))]
Best_models_var_nca.index =[str(i) for i in range(1,len(best_models_nca)+1)]
Best_models_var_nca['Model ID'] = best_models_nca

Best_models_var_nca.to_csv("/Users/Documents/PostdocUW/proc_data/Best_models_var_NCA.csv", )

# i this section i will try to fit top 6 models and return data of coeff, Aicc, R2 in a table

X_WA = read_csv("/Users/Documents/PostdocUW/proc_data/X_NCA.csv")
list_col = list(X_WA)

Xt1 = X_WA[list_col[1:len(list_col)-2]]
Xt4 = X_WA[list_col[len(list_col)-2:]]

Xt1 = Xt1[1:31].reset_index()
Xt4 = Xt4[5:35].reset_index()

Xt1 = Xt1.drop(columns = ['index'])
Xt4 = Xt4.drop(columns = ['index'])

X = Xt1.join(Xt4, how='left')

Results = pd.DataFrame([])
for i in best_models_nca:
    
    X_in                  = sm.add_constant(X[list(comb_var_nca[i])])
    Y_log_in              = np.log(X['Y'])
    
    model                 = sm.OLS(Y_log_in , X_in).fit()
    
    res1                  = pd.DataFrame(model.params).T
    res1["R2"]      = model.rsquared
    res1['AICc']          = sm.tools.eval_measures.aicc(model.llf,model.nobs,len(model.params))
    res1['Model ID']       = i
    
    Results = Results.append(res1)
    
 
Results = Results[['const']+list(set(Results.columns.tolist()) -set(['const', 'AICc', 'R2'])) + ['R2', 'AICc']]
Results = Results.set_index('Model ID')
Results_nca = Results

Results_nca.to_csv("/Users/Documents/PostdocUW/proc_data/Model_table20_NCA.csv", )

#################################################################################################
#################################################################################################
####
#### Central California
#################################################################################################
#################################################################################################

SCA  = read_csv("/Users/Documents/PostdocUW/proc_data/SCA_results_1000.csv")


with open("/Users/documents/postdocuw/proc_data/comb_var_sca.txt", "rb") as fp:
        comb_var_sca = pickle.load(fp)

best_var_sca = SCA[(SCA['R2_test']>0) & (SCA['R2_train']>0)]

best_var_sca = best_var_sca.reset_index()

uni_sca = np.unique(best_var_sca['modelid'])

count_modelid_sca = pd.DataFrame(0, index = [i for i in uni_sca], columns = ["count","Nvar"])

count_mean_pergroup_sca = best_var_sca.groupby('modelid').mean()


for i in best_var_sca['modelid']:
    for j in uni_sca:
        if i == j:
            count_modelid_sca['count'][i] += 1
        
for i in count_modelid_sca.index:
    count_modelid_sca['Nvar'][i]=len(comb_var_sca[int(i)])
    
count_modelid_sca['R2_train'] = count_mean_pergroup_sca['R2_train']
count_modelid_sca['R2_test'] = count_mean_pergroup_sca['R2_test']

count_modelid_sca = count_modelid_sca.sort_values('count', ascending=False)

br_sca = np.arange(len(count_modelid_sca))

best_models_sca = [int(i) for i in count_modelid_sca['count'][:20].index.tolist()]
best_models_var_sca = [ comb_var_sca[i] for i in best_models_sca]
Best_models_var_sca = pd.DataFrame(best_models_var_sca, index = best_models_sca)

Best_models_var_sca.columns = ["variable "+ str(i+1) for i in range(len(Best_models_var_sca.columns))]
Best_models_var_sca.index =[str(i) for i in range(1,len(best_models_sca)+1)]
Best_models_var_sca['Model ID'] = best_models_sca

Best_models_var_sca.to_csv("/Users/Documents/PostdocUW/proc_data/Best_models_var_SCA.csv", )

# i this section i will try to fit top 6 models and return data of coeff, Aicc, R2 in a table

X_WA = read_csv("/Users/Documents/PostdocUW/proc_data/X_SCA.csv")
list_col = list(X_WA)

Xt1 = X_WA[list_col[1:len(list_col)-2]]
Xt4 = X_WA[list_col[len(list_col)-2:]]

Xt1 = Xt1[1:31].reset_index()
Xt4 = Xt4[5:35].reset_index()

Xt1 = Xt1.drop(columns = ['index'])
Xt4 = Xt4.drop(columns = ['index'])

X = Xt1.join(Xt4, how='left')

Results = pd.DataFrame([])
for i in best_models_sca:
    
    X_in                  = sm.add_constant(X[list(comb_var_sca[i])])
    Y_log_in              = np.log(X['Y'])
    
    model                 = sm.OLS(Y_log_in , X_in).fit()
    
    res1                  = pd.DataFrame(model.params).T
    res1["R2"]      = model.rsquared
    res1['AICc']          = sm.tools.eval_measures.aicc(model.llf,model.nobs,len(model.params))
    res1['Model ID']       = i
    
    Results = Results.append(res1)
    
 
Results = Results[['const']+list(set(Results.columns.tolist()) -set(['const', 'AICc', 'R2'])) + ['R2', 'AICc']]
Results = Results.set_index('Model ID')
Results_sca = Results

Results_sca.to_csv("/Users/Documents/PostdocUW/proc_data/Model_table20_SCA.csv", )

#################################################################################################
#################################################################################################
fig = plt.figure()

ax1 = plt.subplot(4,3,1)
ax1 = plt.barh(br_wa[:20], count_modelid_wa['count'][:20],height = 0.4,label='# of samples')
plt.yticks(br_wa[:20],count_modelid_wa.index[:20],fontsize = 6)
plt.ylabel("Model ID")
plt.title("Number of appearances")
ax1 = plt.subplot(4,3,2)
plt.barh(br_wa[:20], count_modelid_wa['R2_train'].values[:20],height = 0.4, label='training')
plt.barh(br_wa[:20]+0.3, count_modelid_wa['R2_test'][:20],height = 0.4, label='testing')
plt.yticks(br_wa[:20],count_modelid_wa.index[:20],fontsize = 6)
plt.title("Averaged R2")
leg =  plt.legend(bbox_to_anchor=(0.6, 1), loc='upper left',fontsize = 6)
ax1 = plt.subplot(4,3,3)
ax1 = plt.barh(br_wa[:20], count_modelid_wa['Nvar'][:20],height = 0.4,label='# of samples')
plt.title("# of variables in model")


ax1 = plt.subplot(4,3,4)
ax1 = plt.barh(br_or[:20], count_modelid_or['count'][:20],height = 0.4,label='# of samples')
plt.yticks(br_or[:20],count_modelid_or.index[:20],fontsize = 6)
plt.ylabel("Model ID")
ax1 = plt.subplot(4,3,5)
plt.barh(br_or[:20], count_modelid_or['R2_train'].values[:20],height = 0.4, label='R2 on training')
plt.barh(br_or[:20]+0.3, count_modelid_or['R2_test'][:20],height = 0.4, label='R2 on testing')
plt.yticks(br_or[:20],count_modelid_or.index[:20],fontsize = 6)
ax1 = plt.subplot(4,3,6)
ax1 = plt.barh(br_or[:20], count_modelid_or['Nvar'][:20],height = 0.4,label='# of samples')
plt.yticks(br_or[:20],count_modelid_or.index[:20],fontsize = 6)

ax1 = plt.subplot(4,3,7)
ax1 = plt.barh(br_nca[:20], count_modelid_nca['count'][:20],height = 0.4,label='# of samples')
plt.yticks(br_nca[:20],count_modelid_nca.index[:20],fontsize = 6)
plt.ylabel("Model ID")
ax1 = plt.subplot(4,3,8)
plt.barh(br_nca[:20], count_modelid_nca['R2_train'].values[:20],height = 0.4, label='R2 on training')
plt.barh(br_nca[:20]+0.3, count_modelid_nca['R2_test'][:20],height = 0.4, label='R2 on testing')
plt.yticks(br_nca[:20],count_modelid_nca.index[:20],fontsize = 6)
ax1 = plt.subplot(4,3,9)
ax1 = plt.barh(br_nca[:20], count_modelid_nca['Nvar'][:20],height = 0.4,label='# of samples')
plt.yticks(br_nca[:20],count_modelid_nca.index[:20],fontsize = 6)

ax1 = plt.subplot(4,3,10)
ax1 = plt.barh(br_sca[:20], count_modelid_sca['count'][:20],height = 0.4,label='# of samples')
plt.yticks(br_sca[:20],count_modelid_sca.index[:20],fontsize = 6)
plt.ylabel("Model ID")
plt.xlabel("count")
ax1 = plt.subplot(4,3,11)
plt.barh(br_sca[:20], count_modelid_sca['R2_train'].values[:20],height = 0.4, label='R2 on training')
plt.barh(br_sca[:20]+0.3, count_modelid_sca['R2_test'][:20],height = 0.4, label='R2 on testing')
plt.yticks(br_sca[:20],count_modelid_sca.index[:20],fontsize = 6)
plt.xlabel("R2")
ax1 = plt.subplot(4,3,12)
ax1 = plt.barh(br_sca[:20], count_modelid_sca['Nvar'][:20],height = 0.4,label='# of samples')
plt.xlabel("count")
plt.yticks(br_sca[:20],count_modelid_sca.index[:20],fontsize = 6)



fig.set_size_inches(8,12)
fig.savefig('/Users/Documents/PostdocUW/figure/Best_model_number.pdf', format='pdf', dpi=1000)
