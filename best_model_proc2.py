#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 06:52:12 2021

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


for i in range(4):
    if i == 0:
        X11 = read_csv("/Users/Documents/PostdocUW/proc_data/X_WA.csv")
        list_core_var = ['meanCST_z1','meanBLT_jvnl0','maxTLT_z5','meanBLT_precond','maxTLT_mg']
        
        list_col = list(X11)
        Xt1 = X11[list_col[1:len(list_col)-2]]
        Xt4 = X11[list_col[len(list_col)-2:len(list_col)-1]]
        
        Xt1 = Xt1[1:31].reset_index()
        Xt4 = Xt4[5:35].reset_index()
        
        X1 = Xt1.drop(columns = ['index'])
        Y1 = np.log(Xt4.drop(columns = ['index']))
        
        X1 = X1[list_core_var]
        
        X_in1 = sm.add_constant(X1)
        
        X1['year'] = range(1985,2015)
        
        model1     = sm.OLS(Y1, X_in1).fit()        
        st1, data1, ss1 = summary_table(model1, alpha=0.05)
        fittedvalues1 = data1[:, 2]
        Ypred1 = model1.predict(X1)
        
    if i == 1:
        X11 = read_csv("/Users/Documents/PostdocUW/proc_data/X_OR.csv")
        list_core_var = ['meanLST_mg', 'meanLST_z1', 'meanCST_z5', 'maxTLT_z3']
        
        list_col = list(X11)
        Xt1 = X11[list_col[1:len(list_col)-2]]
        Xt4 = X11[list_col[len(list_col)-2:len(list_col)-1]]
        
        Xt1 = Xt1[1:31].reset_index()
        Xt4 = Xt4[5:35].reset_index()
        
        X2 = Xt1.drop(columns = ['index'])
        Y2 = np.log(Xt4.drop(columns = ['index']))
        
        X2 = X2[list_core_var]
        
        X_in2 = sm.add_constant(X2)
        
        X2['year'] = range(1985,2015)
        
        model2    = sm.OLS(Y2, X_in2).fit()        
        st2, data2, ss2 = summary_table(model2, alpha=0.05)
        fittedvalues2 = data2[:, 2]
        Ypred2 = model2.predict(X2)
        

    if i == 2:
        X11 = read_csv("/Users/Documents/PostdocUW/proc_data/X_NCA.csv")
        list_core_var = ['maxTLT_z4','meanBLT_spwn']

        list_col = list(X11)
        Xt1 = X11[list_col[1:len(list_col)-2]]
        Xt4 = X11[list_col[len(list_col)-2:len(list_col)-1]]
        
        Xt1 = Xt1[1:31].reset_index()
        Xt4 = Xt4[5:35].reset_index()
        
        X3 = Xt1.drop(columns = ['index'])
        Y3 = np.log(Xt4.drop(columns = ['index']))
        
        X3 = X3[list_core_var]
        
        X_in3 = sm.add_constant(X3)
        
        X3['year'] = range(1985,2015)
        
        model3     = sm.OLS(Y3, X_in3).fit()        
        st1, data3, ss1 = summary_table(model3, alpha=0.05)
        fittedvalues3 = data3[:, 2]
        Ypred3 = model3.predict(X3)
       
    if i == 3:
        X11 = read_csv("/Users/Documents/PostdocUW/proc_data/X_SCA.csv")
        list_core_var = ['meanBLT_spwn', 'maxTLT_z4', 'maxTLT_z1', 'meanCST_z3', 'meanLST_z2']

        list_col = list(X11)
        Xt1 = X11[list_col[1:len(list_col)-2]]
        Xt4 = X11[list_col[len(list_col)-2:len(list_col)-1]]
        
        Xt1 = Xt1[1:31].reset_index()
        Xt4 = Xt4[5:35].reset_index()
        
        X4 = Xt1.drop(columns = ['index'])
        Y4 = np.log(Xt4.drop(columns = ['index']))
        
        X4 = X4[list_core_var]
        
        X_in4 = sm.add_constant(X4)
        
        X4['year'] = range(1985,2015)
        
        model4     = sm.OLS(Y4, X_in4).fit()        
        st1, data4, ss1 = summary_table(model4, alpha=0.05)
        fittedvalues4 = data4[:, 2]
        Ypred4 = model4.predict(X4)        
   

fig2 = plt.figure()

ax1 = plt.subplot(2,2,1)
ax1.scatter(np.exp(fittedvalues1), model1.resid, marker = 'o', c = 'k', s = 50)
ax1.plot([np.exp(Y1.min()),np.exp(Y1.max())],[0,0],'k--')
plt.ylabel("Residuals", fontsize=18)
for i, label in enumerate(X1['year']):
    plt.annotate(label, (np.exp(fittedvalues1[i]), model1.resid[i]), fontsize=8)

ax2 = plt.subplot(2,2,2)
ax2.scatter(np.exp(fittedvalues2), model2.resid, marker = 'o', c = 'k', s = 50)
ax2.plot([np.exp(Y2.min()),np.exp(Y2.max())],[0,0],'k--')
for i, label in enumerate(X2['year']):
    plt.annotate(label, (np.exp(fittedvalues2[i]), model2.resid[i]), fontsize=8)

ax3 = plt.subplot(2,2,3)
ax3.scatter(np.exp(fittedvalues3), model3.resid, marker = 'o', c = 'k', s = 50)
ax3.plot([np.exp(Y3.min()),np.exp(Y3.max())],[0,0],'k--')
plt.xlabel("Fitted \n (million of lbs)", fontsize=18)
plt.ylabel("Residuals", fontsize=18)
for i, label in enumerate(X3['year']):
    plt.annotate(label, (np.exp(fittedvalues3[i]), model3.resid[i]), fontsize=8)

ax4 = plt.subplot(2,2,4)
ax4.scatter(np.exp(fittedvalues4), model4.resid, marker = 'o', c = 'k', s = 50)
ax4.plot([np.exp(Y4.min()),np.exp(Y4.max())],[0,0],'k--')
plt.xlabel("Fitted \n (million of lbs)", fontsize=18)
for i, label in enumerate(X4['year']):
    plt.annotate(label, (np.exp(fittedvalues4[i]), model4.resid[i]), fontsize=8)


    
    
plt.show()   
fig2.set_size_inches(8,6)
fig2.savefig('/Users/Documents/PostdocUW/figure/Fitted_resid.pdf',bbox_inches='tight', format='pdf', dpi=1000)
    

