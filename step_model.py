#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:52:34 2021

@author: N
"""
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tools import eval_measures


X_WA = read_csv('/Users/Documents/PostdocUW/proc_data/X_WA.csv')
Y_WA = read_csv('/Users/Documents/PostdocUW/proc_data/Y_WA.csv')
X_WA['Yt_3']= X_WA['feml_prey_jvl0']

X_WA = X_WA[1:31]
Y_WA = Y_WA[1:31]

X = X_WA.Yt_3.values
y= Y_WA.SB_WA.values

X_train, X_test = X[1:len(X)-5], X[len(X)-5:]
y_train, y_test = y[1:len(X)-5], y[len(X)-5:]

# add constant to X
X = sm.add_constant(X_train) 

est=sm.OLS(y_train, X)
est = est.fit()

est.summary()

eval_measures.aicc(est.llf, est.nobs, est.df_model+1)

plt.plot
plt.plot(Y_WA.date.values[1:len(X)-5], y_train)
plt.plot(Y_WA.date.values[1:len(X)-5], est.predict(sm.add_constant(X_train)) )
plt.plot(Y_WA.date.values[len(X)-5:], y_test)
plt.plot(Y_WA.date.values[len(X)-5:], est.predict(sm.add_constant(X_test)) )