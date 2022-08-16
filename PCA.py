#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:16:59 2021

@author: N
"""

from pandas import read_csv
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pandas.plotting import table

pca = PCA(n_components=5)



X_WA = read_csv('/Users/Documents/PostdocUW/proc_data/X_WA.csv')

X_OR = read_csv('/Users/Documents/PostdocUW/proc_data/X_OR.csv')
X_NCA = read_csv('/Users/Documents/PostdocUW/proc_data/X_NCA.csv')

X_SCA = read_csv('/Users/Documents/PostdocUW/proc_data/X_SCA.csv')


def model_output(argument): 
       switcher = { 
           0: X_WA['Y'], 
           1: X_OR['Y'], 
           2: X_NCA['Y'], 
           3: X_SCA['Y'],
       } 
       return switcher.get(argument) 
def model_input(argument):
    switcher = {
        0: X_WA,
        1: X_OR,
        2: X_NCA,
        3: X_SCA
    }
    return switcher.get(argument) 
results = []

for i in range(0,4):
    X = model_input(i)
    Y = model_output(i)

    X = X[1:31]
    Y = Y[5:35]
    
    X = X.drop(columns = ['date','Y','Y_SD'])
    #Y = Y.drop(columns = ['Unnamed: 0', 'index', 'date'])
    
    X = X.reset_index()
    Y = Y.reset_index()
    
    X = X.drop(columns = ['index'])
    Y = Y.drop(columns = ['index'])
    

    
    X_trans = StandardScaler().fit_transform(X)
    principalComponents = pca.fit_transform(X_trans)
    
    loadings = pd.DataFrame(np.round(pca.components_.T,2), columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'], index=list(X.head(0)))
    results.append(pca.explained_variance_ratio_)
    
    fig0 = plt.figure()
    ax1 = plt.subplot(212, frame_on=False) # no visible frame
    ax1.xaxis.set_visible(False)  # hide the x axis
    ax1.yaxis.set_visible(False)  # hide the y axis

    table(ax1, loadings)  # where df is your data frame
    
    ax2 = plt.subplot(211)
    plt.pcolor(loadings.abs())
    plt.yticks(np.arange(0.5, len(loadings.index), 1), list(X.head(0)))
    plt.xticks(np.arange(0.5, len(loadings.columns), 1), loadings.columns)
    plt.show()
    fig0.tight_layout(pad=1.0)
    fig0.set_size_inches(14,18)
    fig0.savefig('/Users/Documents/PostdocUW/figure/SCA_PC_Loadings.pdf',bbox_inches='tight', format='pdf', dpi=500)
    
    
    
    
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
    #principalDf.index=loadings.index
    finalDf = pd.concat([principalDf, Y], axis = 1)
    
    finalDf.to_csv("/Users/Documents/PostdocUW/proc_data/PCA_"+str(i)+"_.csv", )
    
    fig = plt.figure()
    ax = plt.subplot(2,2,i+1) 
    ax.set_xlabel('PC 1', fontsize = 15)
    ax.set_ylabel('PC 2', fontsize = 15)
    #ax.set_title('2 component PCA', fontsize = 20)
    ax.scatter(finalDf['PC1']
           , finalDf['PC2']
           , s = 50)
        
plt.show()        
results = np.transpose(results)
results = np.round(results*100,2)
    
R = pd.DataFrame(results, columns=['WA','OR','NCA','SCA'], index=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5'])

display(R)
from tabulate import tabulate 
print(tabulate(R, headers = 'keys', tablefmt = 'psql')) 
