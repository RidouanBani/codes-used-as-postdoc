#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:52:11 2021

@author: N
"""

import numpy as np
import pandas as pd
from pandas import read_csv
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
model1 = LinearRegression(normalize = True)
model2 = LinearRegression(normalize = True)
model3 = LinearRegression(normalize = True)
model4 = LinearRegression(normalize = True)


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

X_NCA= X_NCA.join(Y_NCA, how='left')

X_NCA['meanLST_z1or'] = X_OR['meanLST_z1']
X_NCA['meanLST_z2or'] = X_OR['meanLST_z2']
X_NCA['meanLST_z3or'] = X_OR['meanLST_z3']
X_NCA['meanLST_z4or'] = X_OR['meanLST_z4']
X_NCA['meanLST_z5or'] = X_OR['meanLST_z5']
X_NCA['meanLST_mgor'] = X_OR['meanLST_mg']

X_NCA['year'] = list(range(1985,2015))

###############################################
Xt1 = X_SCA[list_col[1:len(list_col)-2]]
Xt4 = X_SCA[list_col[len(list_col)-2:len(list_col)-1]]

Xt1 = Xt1[1:31].reset_index()
Xt4 = Xt4[5:35].reset_index()

X_SCA = Xt1.drop(columns = ['index'])
Y_SCA = Xt4.drop(columns = ['index'])
##################################################

X_ORNCA = (X_NCA.stack()+X_OR.stack())/2
X_ORNCA = X_ORNCA.unstack()

Y_ORNCA = Y_NCA+Y_OR

X_ORNCA['Y']= Y_ORNCA

X_ORNCA.to_csv("/Users/Documents/PostdocUW/proc_data/X_ORNCA.csv", )


model1.fit(Y_NCA.values, Y_OR.values)
Y_predict1 = model1.predict(Y_NCA.values)
r_sq1 = model1.score(Y_NCA.values, Y_OR.values)
coef1 = model1.coef_

model2.fit(Y_OR.values, Y_WA.values)
Y_predict2 = model1.predict(Y_OR.values)
r_sq2 = model2.score(Y_OR.values, Y_WA.values)
coef2 = model2.coef_

model3.fit(Y_SCA.values, Y_NCA.values)
Y_predict3 = model1.predict(Y_SCA.values)
r_sq3 = model3.score(Y_SCA.values, Y_NCA.values)
coef3 = model3.coef_

fig = plt.figure()
ax1 = plt.subplot(2, 2, 1)
plt.scatter(Y_OR.values, Y_WA.values,  color='black')
plt.plot(Y_OR.values, Y_predict2, color='blue', linewidth=3)
plt.xlabel("OR abundance")
plt.ylabel("WA abundance")
plt.text(5, 19, f'Coef = {round(coef2[0][0],2)} \n R^2={round(r_sq2*100,2)}', fontsize=10)


ax2 = plt.subplot(2, 2, 2)
plt.scatter(Y_NCA.values, Y_OR.values,  color='black')
plt.plot(Y_NCA.values, Y_predict1, color='blue', linewidth=3)
plt.xlabel("North CA abundance")
plt.ylabel("OR abundance")
plt.text(2.5, 15, f'Coef = {round(coef1[0][0],2)} \n R^2={round(r_sq1*100,2)}', fontsize=10)

ax3 = plt.subplot(2, 2, 3)
plt.scatter(Y_SCA.values, Y_NCA.values,  color='black')
plt.plot(Y_SCA.values, Y_predict3, color='blue', linewidth=3)
plt.xlabel("Central CA abundance")
plt.ylabel("North CA abundance")
plt.text(2.5, 11, f'Coef = {round(coef3[0][0],2)} \n R^2={round(r_sq3*100,2)}', fontsize=10)

plt.show()
fig.set_size_inches(8,8)
fig.savefig("/Users/Documents/PostdocUW/figure/regress_region.pdf",bbox_inches='tight', format='pdf', dpi=1000)

