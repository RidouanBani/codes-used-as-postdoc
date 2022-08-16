#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:17:19 2021

@author: N
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from pandas import read_csv
import matplotlib.pyplot as plt

model = LinearRegression(normalize = True)

X_WA = read_csv("/Users/Documents/PostdocUW/proc_data/X_WA.csv")


X_OR = read_csv("/Users/Documents/PostdocUW/proc_data/X_OR.csv")


X_NCA = read_csv("/Users/Documents/PostdocUW/proc_data/X_NCA.csv")


X_SCA = read_csv("/Users/Documents/PostdocUW/proc_data/X_SCA.csv")

###########################################
list_col = list(X_WA)
Xt1 = X_WA[list_col[1:len(list_col)-2]]
Xt4 = X_WA[list_col[len(list_col)-2:len(list_col)-1]]

Xt1 = Xt1[1:31].reset_index()
Xt4 = Xt4[5:35].reset_index()

X_WA = Xt1.drop(columns = ['index'])
Y_WA = Xt4.drop(columns = ['index'])

############################################
Xt1 = X_OR[list_col[1:len(list_col)-2]]
Xt4 = X_OR[list_col[len(list_col)-2:len(list_col)-1]]

Xt1 = Xt1[1:31].reset_index()
Xt4 = Xt4[5:35].reset_index()

X_OR = Xt1.drop(columns = ['index'])
Y_OR = Xt4.drop(columns = ['index'])

###############################################
Xt1 = X_NCA[list_col[1:len(list_col)-2]]
Xt4 = X_NCA[list_col[len(list_col)-2:len(list_col)-1]]

Xt1 = Xt1[1:31].reset_index()
Xt4 = Xt4[5:35].reset_index()

X_NCA = Xt1.drop(columns = ['index'])
Y_NCA = Xt4.drop(columns = ['index'])

###############################################
Xt1 = X_SCA[list_col[1:len(list_col)-2]]
Xt4 = X_SCA[list_col[len(list_col)-2:len(list_col)-1]]

Xt1 = Xt1[1:31].reset_index()
Xt4 = Xt4[5:35].reset_index()

X_SCA = Xt1.drop(columns = ['index'])
Y_SCA = Xt4.drop(columns = ['index'])


def model_output(argument): 
    switcher = { 
        0: Y_WA, 
        1: Y_OR, 
        2: Y_NCA, 
        3: Y_SCA,
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

def polyfit1(x, y, degree):
    #results = {}
    coeffs = np.polyfit(x, y, degree).reshape(1,-1)[0].tolist()
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    #results['r_squared'] = ssreg / sstot

    return ssreg / sstot

Rlinear=[]
Rquadr =[]
for state in range(0,4):
        X = model_input(state)
        Y = model_output(state)
        ii = 1
        fig = plt.figure()
        for var in X.columns:
           
            model.fit(X[var].values.reshape(-1, 1), Y.values)
            Y_predict = model.predict(X[var].values.reshape(-1, 1))
            r_sq = model.score(X[var].values.reshape(-1, 1), Y.values)
            coef = model.coef_
            
            model1 = np.poly1d(np.polyfit(X[var].values, Y.values, 2).reshape(1,-1)[0].tolist())
            polyline = np.linspace(min(X[var].values), max(X[var].values), 30)
            ax = plt.subplot(5, 6, ii)
            plt.scatter(X[var].values.reshape(-1, 1), Y.values,  color='black')
            plt.plot(X[var].values.reshape(-1, 1), Y_predict, color='blue', linewidth=3)
            plt.plot(polyline, model1(polyline), color='red', linewidth=3)
            
            
            r_sq2=  polyfit1(X[var].values, Y.values, 2)
            if ii in [1,7,13,19]:
               plt.ylabel('abundance')
            plt.xlabel(var)
            ax.xaxis.set_label_coords(.5, -0.12)
            #    plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_title(f'line = {round(r_sq*100)} \n quad={round(r_sq2*100)}', y=0.8, pad=-8)
        
            # else:
            #    plt.setp(ax.get_xticklabels(), visible=False)
            #    plt.setp(ax.get_yticklabels(), visible=False)
            #    ax.set_title(f'{var} \n R_sq={round(r_sq*100)} \n coef ={round(coef[0][0])}', y=0.9, pad=-8)
            ii+=1
            
        plt.show()
        fig.set_size_inches(14,14)
        fig.savefig("/Users/Documents/PostdocUW/figure/LnReg"+str(state)+".pdf",bbox_inches='tight', format='pdf', dpi=1000)


