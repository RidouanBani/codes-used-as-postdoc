#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:41:50 2021

@author: N
"""

from pandas import read_csv

import statsmodels.api as sm
import numpy as np

import matplotlib.pyplot as plt

X_WA = read_csv('/Users/Documents/PostdocUW/proc_data/X_WA.csv')
Y_WA = read_csv('/Users/Documents/PostdocUW/proc_data/Y_WA.csv')

X_OR = read_csv('/Users/Documents/PostdocUW/proc_data/X_OR.csv')
Y_OR = read_csv('/Users/Documents/PostdocUW/proc_data/Y_OR.csv')

X_NCA = read_csv('/Users/Documents/PostdocUW/proc_data/X_NCA.csv')
Y_NCA = read_csv('/Users/Documents/PostdocUW/proc_data/Y_NCA.csv')

X_SCA = read_csv('/Users/Documents/PostdocUW/proc_data/X_SCA.csv')
Y_SCA = read_csv('/Users/Documents/PostdocUW/proc_data/Y_SCA.csv')

X_WA = X_WA.drop(columns = ['Unnamed: 0'])
Y_WA = Y_WA.drop(columns = ['Unnamed: 0', 'index', 'date'])

X_OR = X_OR.drop(columns = ['Unnamed: 0'])
Y_OR = Y_OR.drop(columns = ['Unnamed: 0', 'index', 'date'])

X_NCA = X_NCA.drop(columns = ['Unnamed: 0'])
Y_NCA = Y_NCA.drop(columns = ['Unnamed: 0', 'index', 'date'])

X_SCA = X_SCA.drop(columns = ['Unnamed: 0'])
Y_SCA = Y_SCA.drop(columns = ['Unnamed: 0', 'index', 'date'])

X_WA = X_WA[1:31]
Y_WA = Y_WA[1:31]

X_OR = X_OR[1:31]
Y_OR = Y_OR[1:31]

X_NCA = X_NCA[1:31]
Y_NCA = Y_NCA[1:31]

X_SCA = X_SCA[1:31]
Y_SCA = Y_SCA[1:31]

Data_WA = read_csv('/Users/Documents/PostdocUW/data_proc/data0.csv')
Data_OR = read_csv('/Users/Documents/PostdocUW/data_proc/data1.csv')
Data_NCA = read_csv('/Users/Documents/PostdocUW/data_proc_2-21/data2.csv')
Data_SCA = read_csv('/Users/Documents/PostdocUW/data_proc_2-21/data3.csv')

Data_WA = Data_WA.drop(columns = ['Unnamed: 0'])
Data_OR = Data_OR.drop(columns = ['Unnamed: 0'])
Data_NCA = Data_NCA.drop(columns = ['Unnamed: 0'])
Data_SCA = Data_SCA.drop(columns = ['Unnamed: 0'])

# compare performance of models by comparing min AIC
P_WA = Data_WA.min()
P_OR = Data_OR.min()
P_NCA = Data_NCA.min()
P_SCA = Data_SCA.min()

# GLM Gamma performed better and returned lower AIC for WA, OR, NCA, SCA

ind_WA_gamma = Data_WA.index[Data_WA['gamma'] == min(Data_WA['gamma'])]
ind_OR_gamma = Data_OR.index[Data_OR['gamma'] == min(Data_OR['gamma'])]
ind_NCA_gamma = Data_NCA.index[Data_NCA['gamma'] == min(Data_NCA['gamma'])]
ind_SCA_gamma = Data_SCA.index[Data_SCA['gamma'] == min(Data_SCA['gamma'])]


lis_var_gamma_WA  = Data_WA['var_comb'][ind_WA_gamma [0]]
lis_var_gamma_WA  = lis_var_gamma_WA[2:len(lis_var_gamma_WA)-2]
lis_var_gamma_WA  = lis_var_gamma_WA.split("', '")

lis_var_gamma_OR  = Data_OR['var_comb'][ind_OR_gamma [0]]
lis_var_gamma_OR  = lis_var_gamma_OR[2:len(lis_var_gamma_OR)-2]
lis_var_gamma_OR  = lis_var_gamma_OR.split("', '")

lis_var_gamma_NCA  = Data_NCA['var_comb'][ind_NCA_gamma [0]]
lis_var_gamma_NCA  = lis_var_gamma_NCA[2:len(lis_var_gamma_NCA)-2]
lis_var_gamma_NCA  = lis_var_gamma_NCA.split("', '")

lis_var_gamma_SCA  = Data_SCA['var_comb'][ind_SCA_gamma [0]]
lis_var_gamma_SCA  = lis_var_gamma_SCA[2:len(lis_var_gamma_SCA)-2]
lis_var_gamma_SCA  = lis_var_gamma_SCA.split("', '")


X_WA_gamma = X_WA[lis_var_gamma_WA]
X_OR_gamma = X_OR[lis_var_gamma_OR]
X_NCA_gamma = X_NCA[lis_var_gamma_NCA]
X_SCA_gamma = X_SCA[lis_var_gamma_SCA]

X_WA_gamma  = sm.add_constant(X_WA_gamma)
X_OR_gamma  = sm.add_constant(X_OR_gamma)
X_NCA_gamma  = sm.add_constant(X_NCA_gamma)
X_SCA_gamma  = sm.add_constant(X_SCA_gamma)


glm_gamma_WA= sm.GLM(Y_WA, X_WA_gamma, family=sm.families.Gamma(sm.families.links.log()))
glm_gamma_OR= sm.GLM(Y_OR, X_OR_gamma, family=sm.families.Gamma(sm.families.links.log()))
glm_gamma_NCA= sm.GLM(Y_NCA, X_NCA_gamma, family=sm.families.Gamma(sm.families.links.log()))
glm_gamma_SCA= sm.GLM(Y_SCA, X_SCA_gamma, family=sm.families.Gamma(sm.families.links.log()))

res_gamma_WA= glm_gamma_WA.fit()
res_gamma_OR= glm_gamma_OR.fit()
res_gamma_NCA= glm_gamma_NCA.fit()
res_gamma_SCA= glm_gamma_SCA.fit()


print(res_gamma_WA.summary())
print(res_gamma_OR.summary())
print(res_gamma_NCA.summary())
print(res_gamma_SCA.summary())


plt.rc('figure', figsize=(8, 4))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(res_gamma_WA.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('/Users/Documents/PostdocUW/figure/res_gamma_WA_summary.pdf')

plt.rc('figure', figsize=(8, 4))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(res_gamma_OR.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('/Users/Documents/PostdocUW/figure/res_gamma_OR_summary.pdf')

plt.rc('figure', figsize=(8, 4))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(res_gamma_NCA.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('/Users/Documents/PostdocUW/figure/res_gamma_NCA_summary.pdf')

plt.rc('figure', figsize=(8, 4))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(res_gamma_SCA.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('/Users/Documents/PostdocUW/figure/res_gamma_SCA_summary.pdf')

#####################################
X_wa = X_WA_gamma.to_numpy()
proba_wa = res_gamma_WA.predict(X_wa)
cov_wa = res_gamma_WA.cov_params()
gradient_wa = (proba_wa * (1 - proba_wa) * X_wa.T).T 
std_errors_wa = np.array([np.sqrt(np.dot(np.dot(g, cov_wa), g)) for g in gradient_wa])

c = 1 # multiplier for confidence interval
upper_wa = np.maximum(0,proba_wa + c*std_errors_wa )
lower_wa = np.maximum(0,proba_wa - c*std_errors_wa )
#####################################
X_or = X_OR_gamma.to_numpy()
proba_or = res_gamma_OR.predict(X_or)
cov_or = res_gamma_OR.cov_params()
gradient_or = (proba_or * (1 - proba_or) * X_or.T).T 
std_errors_or = np.array([np.sqrt(np.dot(np.dot(g, cov_or), g)) for g in gradient_or])

c = 1 # multiplier for confidence interval
upper_or = np.maximum(0,proba_or + c*std_errors_or )
lower_or = np.maximum(0,proba_or - c*std_errors_or )
#####################################
X_nca = X_NCA_gamma.to_numpy()
proba_nca = res_gamma_NCA.predict(X_nca)
cov_nca = res_gamma_NCA.cov_params()
gradient_nca = (proba_nca * (1 - proba_nca) * X_nca.T).T 
std_errors_nca = np.array([np.sqrt(np.dot(np.dot(g, cov_nca), g)) for g in gradient_nca])

c = 1 # multiplier for confidence interval
upper_nca = np.maximum(0,proba_nca + c*std_errors_nca )
lower_nca = np.maximum(0,proba_nca - c*std_errors_nca )
#####################################
X_sca = X_SCA_gamma.to_numpy()
proba_sca = res_gamma_SCA.predict(X_sca)
cov_sca = res_gamma_SCA.cov_params()
gradient_sca = (proba_sca * (1 - proba_sca) * X_sca.T).T 
std_errors_sca = np.array([np.sqrt(np.dot(np.dot(g, cov_sca), g)) for g in gradient_sca])

c = 1 # multiplier for confidence interval
upper_sca = np.maximum(0,proba_sca + c*std_errors_sca )
lower_sca = np.maximum(0,proba_nca - c*std_errors_nca )

#######################################################################################

fig0 = plt.figure()

ax1 = plt.subplot(221)
plt.plot(X_WA['date'], Y_WA['SB_WA'], color="k", label='Obs', linewidth=2, linestyle="-")
plt.plot(X_WA['date'], proba_wa, color="r", label='pred', linewidth=2, linestyle="-")
plt.plot(X_WA['date'], lower_wa, color="r", label='+std', linewidth=1, linestyle="dotted")
plt.plot(X_WA['date'], upper_wa, color="r", label='-std', linewidth=1, linestyle="dotted")
leg0 =  plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left')

ax2 = plt.subplot(212)
plt.plot(X_OR['date'], Y_OR['SB_OR'], color="k", label='Obs', linewidth=2, linestyle="-")
plt.plot(X_OR['date'], proba_or, color="r", label='pred', linewidth=2, linestyle="-")
plt.plot(X_OR['date'], lower_or, color="r", label='+std', linewidth=1, linestyle="dotted")
plt.plot(X_OR['date'], upper_or, color="r", label='-std', linewidth=1, linestyle="dotted")
#leg0 =  plt.legend(loc='upper right')
leg0 =  plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left')

ax3 = plt.subplot(223)
plt.plot(range(1981,2011), Y_NCA['SB_NCA'], color="k", label='Obs', linewidth=2, linestyle="-")
plt.plot(range(1981,2011), proba_nca, color="r", label='pred', linewidth=1, linestyle="-")
plt.plot(range(1981,2011), lower_nca, color="r", label='-std', linewidth=1, linestyle="dotted")
plt.plot(range(1981,2011), upper_nca, color="r", label='+std', linewidth=1, linestyle="dotted")
#leg0 =  plt.legend(loc='upper right')

ax4 = plt.subplot(224)
plt.plot(range(1981,2011), Y_SCA['SB_SCA'], color="k", label='Obs', linewidth=2, linestyle="-")
plt.plot(range(1981,2011), proba_sca, color="r", label='pred', linewidth=1, linestyle="-")
plt.plot(range(1981,2011), lower_sca, color="r", label='-std', linewidth=1, linestyle="dotted")
plt.plot(range(1981,2011), upper_sca, color="r", label='+std', linewidth=1, linestyle="dotted")
leg0 =  plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left')
plt.show()

fig0.set_size_inches(14,7)
fig0.savefig('/Users/Documents/PostdocUW/figure/pred_SB_gamma.pdf',bbox_inches='tight', format='pdf', dpi=1000)

#############################################

fig1 = plt.figure()

ax1 = plt.subplot(221)
plt.plot(range(1981,2011), Y_WA['SB_WA'], color="k", label='Obs', linewidth=2, linestyle="-")
plt.plot(range(1981,2011), Y_WA['SB_WA']-proba_wa, color="r", label='Obs-pred', linewidth=2, linestyle="-")
ax1.set_title('WA')

ax2 = plt.subplot(222)
plt.plot(range(1981,2011), Y_OR['SB_OR'], color="k", label='Obs', linewidth=2, linestyle="-")
plt.plot(range(1981,2011), Y_OR['SB_OR']-proba_or, color="r", label='Obs-pred', linewidth=2, linestyle="-")
ax2.set_title('OR')
#leg0 =  plt.legend(loc='upper right')

ax3 = plt.subplot(223)
plt.plot(range(1981,2011), Y_NCA['SB_NCA'], color="k", label='Obs', linewidth=2, linestyle="-")
plt.plot(range(1981,2011), Y_NCA['SB_NCA']-proba_nca, color="r", label='Obs-pred', linewidth=2, linestyle="-")
ax3.set_title('North CA')
#leg0 =  plt.legend(loc='upper right')

ax4 = plt.subplot(224)
plt.plot(range(1981,2011), Y_SCA['SB_SCA'], color="k", label='Obs', linewidth=2, linestyle="-")
plt.plot(range(1981,2011), Y_SCA['SB_SCA']-proba_sca, color="r", label='Obs-pred', linewidth=2, linestyle="-")
ax4.set_title('Central CA')
leg0 =  plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left')
plt.show()

fig1.set_size_inches(14,7)
fig1.savefig('/Users/Documents/PostdocUW/figure/Obs-pred_gamma.pdf',bbox_inches='tight', format='pdf', dpi=1000)


#############################################################################

####
#plotting

nobs = res_poisson.nobs
y = Y_WA['SB_WA'] # /Y_WA['SB_WA'].sum()
yhat_gamma = res_gamma_log.mu # /res_poisson.mu.sum()
yhat_gaussian = res_gaussian.mu

##############################################################################
from statsmodels.graphics.api import abline_plot

fig1, ax1 = plt.subplots()
ax1.scatter(yhat_gamma, y)
ax1.scatter(yhat_gaussian, y)
line_fit1 = sm.OLS(y, sm.add_constant(yhat_gamma, prepend=True)).fit()
line_fit2 = sm.OLS(y, sm.add_constant(yhat_gaussian, prepend=True)).fit()
abline_plot(model_results = line_fit1, color="r", ax=ax1)
abline_plot(model_results=line_fit2, color="b", ax=ax1)


ax1.set_title('Model Fit Plot: WA')
ax1.set_ylabel('Observed values')
ax1.set_xlabel('Fitted values');

##############################################################################
fig2, ax2 = plt.subplots()

ax2.scatter(yhat, res_poisson.resid_pearson)
ax2.hlines(0, 0, 20)
ax2.set_xlim(0, 20)
ax2.set_title('Residual Dependence Plot')
ax2.set_ylabel('Pearson Residuals')
ax2.set_xlabel('Fitted values')

##############################################################################
from scipy import stats

fig3, ax3 = plt.subplots()

resid = res_poisson.resid_deviance.copy()
resid_std = stats.zscore(resid)
ax3.hist(resid_std, bins=25)
ax3.set_title('Histogram of standardized deviance residuals');


##############################################################################
from statsmodels import graphics
graphics.gofplots.qqplot(resid, line='r')