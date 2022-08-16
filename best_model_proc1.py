
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:40:54 2021

@author: N
"""

import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

plt.rc("figure", figsize=(16,8))
plt.rc("font", size=14)


fig1 = plt.figure()
for i in range(1,5):
    if i == 0:
        X1 = read_csv("/Users/Documents/PostdocUW/proc_data/X_WAq.csv")
        list_core_var = ['meanCST_z1','meanCST_z12','maxTLT_z3','meanBLT_precond','maxTLT_mg']
    if i == 1:
        X1 = read_csv("/Users/Documents/PostdocUW/proc_data/X_WA.csv")
        list_core_var = ['meanCST_z1','meanBLT_jvnl0','maxTLT_z5','meanBLT_precond','maxTLT_mg']
        
        
    if i == 2:
        X1 = read_csv("/Users/Documents/PostdocUW/proc_data/X_OR.csv")
        list_core_var = ['meanCST_z5','meanLST_mg', 'meanLST_z1', 'maxTLT_z3']
    if i == 3:
        X1 = read_csv("/Users/Documents/PostdocUW/proc_data/X_NCA.csv")
        list_core_var = ['maxTLT_z5','maxTLT_z3']

       
    if i == 4:
        X1 = read_csv("/Users/Documents/PostdocUW/proc_data/X_SCA.csv")
        list_core_var = ['meanBLT_spwn', 'maxTLT_z4', 'maxTLT_z1', 'meanCST_z3', 'meanLST_z2']
        
    
    list_col = list(X1)
    Xt1 = X1[list_col[1:len(list_col)-2]]
    Xt4 = X1[list_col[len(list_col)-2:len(list_col)-1]]
    
    Xt1 = Xt1[1:31].reset_index()
    Xt4 = Xt4[5:35].reset_index()
    
    X = Xt1.drop(columns = ['index'])
    Y = np.log(Xt4.drop(columns = ['index']))
    
    if i ==0:
        X['meanCST_z12'] = X['meanCST_z1']**2
    # 5 core variables 
    
    
    X = X[list_core_var]
    
    X_in = sm.add_constant(X)
    
    X['year'] = range(1985,2015)
    
    model     = sm.OLS(Y, X_in).fit()
    
    st, data, ss2 = summary_table(model, alpha=0.05)
    
    J_year = pd.DataFrame([np.NAN]*len(X_in), columns = ['Y'])
    for j in range(len(X)):
        
        X_innew = X_in.drop(labels = j, axis = 0)
        Ynew = Y.drop(labels = j, axis = 0)
        
        modelnew     = sm.OLS(Ynew, X_innew).fit()
        
        J_year.loc[j] = modelnew.predict(X_in.loc[j].values)
        
    X_innew6 = X_in[0:24]
    Ynew6 = Y[0:24]
    
    modelnew6     = sm.OLS(Ynew6, X_innew6).fit()
    predict6      = modelnew6.predict(X_in.loc[25:30].values)
    
    fittedvalues = data[:, 2]
    predict_mean_se  = data[:, 3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
    predict_ci_low, predict_ci_upp = data[:, 6:8].T
    
    model.summary()
    Ypred = model.predict(X)
    
    # fig = sm.graphics.influence_plot(model, criterion="cooks")
    
    ax = plt.subplot(2,2,i)
    
    plt.plot(X['year'], np.exp([Y.mean()]*len(X['year'])), 'k-', label='Mean Obs.')
    
    plt.scatter(X['year'], np.exp(Y), marker = 'o', c = 'black', s = 50, label='Obs.')
    plt.plot(X['year'], np.exp(fittedvalues), 'r-', lw=1, label='mean fitted')
    plt.fill_between(X['year'], np.exp(predict_mean_ci_low), np.exp(predict_mean_ci_upp),color='red',alpha=0.3,
                     label='C.I. fitted')
    plt.fill_between(X['year'], np.exp([Y.mean()[0]-Y.std()[0]]*len(X['year'])), 
                     np.exp([Y.mean()[0]+Y.std()[0]]*len(X['year'])),color='gray',alpha=0.3, label='C.I. Obs.')
    plt.scatter(X['year'], np.exp(J_year), marker = 'o', c = 'blue', s = 50, label='Jackknife')
    plt.scatter(X['year'][25:30], np.exp(predict6), marker = 'o', c = 'green', s = 50, label='Predected \n last 6 yrs')
    if i ==1 or i ==3:
        plt.ylabel("Abundance \n (million of lbs)", fontsize=18)
    if i == 3 or i == 4:
        plt.xlabel("Year", fontsize=18)
    if i == 4:
        plt.ylim(-1, 15);
    if i == 1:
        plt.ylim(0,25)
    if i == 2:
        plt.ylim(0,20)
    if i == 3:
        plt.ylim(0,11)
    if i ==2:
        leg =  plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
        
    if i == 1:
        ax.text(1990, 22, 'Washington')
        #ax.annotate('Washington', xy = (1985, 0.2), xytext=(1985, 0.2))
    if i == 2:
        ax.text(1990, 18, 'Oregon')
    if i == 3:
        ax.text(1990, 10, 'Northern California')
    if i == 4:
        ax.text(1990, 13, 'Central California')
    #leg =  plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left', ncol=4)
    #plt.show()
    #fig.tight_layout(pad=1.0)
    
    

    
plt.show()   
fig1.set_size_inches(12,6)
fig1.savefig('/Users/Documents/PostdocUW/figure/Fitted_data.pdf',bbox_inches='tight', format='pdf', dpi=1000)
    


