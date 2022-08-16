#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:09:14 2021

@author: N
"""

import statsmodels.api as sm
import numpy as np

def glm_comb(Y, combo, X):
    
    print(f'Proccessing {str(combo)}')
    Xnew = X[[combo[u] for u in range(len(combo))]]
    # Corr = Xnew.corr()
    # np.fill_diagonal(Corr.values, 0)
    
    exog  = sm.add_constant(Xnew)
     
    #glm_poisson = sm.GLM(Y, exog, family=sm.families.Poisson())
    glm_gamma_log = sm.GLM(Y, exog, family=sm.families.Gamma(sm.families.links.log()))
    #glm_gaussian = sm.GLM(Y, exog, family = sm.families.Gaussian())
    #glm_invgaussian = sm.GLM(Y, exog, family = sm.families.InverseGaussian(sm.families.links.log()))
    #glm_negbinomial = sm.GLM(Y, exog, family = sm.families.NegativeBinomial(alpha=1))
    #glm_tweedie = sm.GLM(Y, exog, family = sm.families.Tweedie())
     
    #res_poisson = glm_poisson.fit()
    res_gamma = glm_gamma_log.fit()
    #res_gaussian = glm_gaussian.fit()
    #res_invgaussian = glm_invgaussian.fit()
    #res_negbinomial = glm_negbinomial.fit()
    #res_tweedie = glm_tweedie.fit()
       
    return res_gamma.aic #[res_gamma.aic, res_gaussian.aic, res_negbinomial.aic, res_tweedie.aic]

def check_corr(Y, combo, X):
    Xnew = X[[combo[u] for u in range(len(combo))]]
    Corr = Xnew.corr()
    np.fill_diagonal(Corr.values, 0)
    # if Corr[Corr.values> 0.75].empty == True:
    #    results = glm_comb(Y, combo, Xnew)
    # else:
    #    results = 0
       
    return Corr[Corr.values> 0.75].empty
       
    