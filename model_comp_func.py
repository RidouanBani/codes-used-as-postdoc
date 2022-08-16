#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:59:06 2021

@author: N
"""

from pandas import read_csv
import pandas as pd
import numpy as np

import itertools

import sys
sys.path.append('/home/rbani20/projects/def-guichard/rbani20/postdoc')
#sys.path.append('/Users/Documents/PostdocUW/codes')
from func_model_comp import glm_comb, check_corr

#import multiprocessing as mp

def main():

    # load data
    X_WA = read_csv('/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/X_WA.csv') 
    Y_WA = read_csv('/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/Y_WA.csv') 
    
    # X_WA = read_csv('/Users/Documents/PostdocUW/proc_data/X_WA.csv')
    # Y_WA = read_csv('/Users/Documents/PostdocUW/proc_data/Y_WA.csv')
    
    X_OR = read_csv('/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/X_OR.csv') 
    Y_OR = read_csv('/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/Y_OR.csv') 
    
    X_NCA = read_csv('/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/X_NCA.csv') 
    Y_NCA = read_csv('/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/Y_NCA.csv') 
    #X_NCA = read_csv('/Users/Documents/PostdocUW/proc_data/X_NCA.csv') 
    #Y_NCA = read_csv('/Users/Documents/PostdocUW/proc_data/Y_NCA.csv') 
    
    X_SCA = read_csv('/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/X_SCA.csv') 
    Y_SCA = read_csv('/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/Y_SCA.csv')
    #X_SCA = read_csv('/Users/Documents/PostdocUW/proc_data/X_SCA.csv') 
    #Y_SCA = read_csv('/Users/Documents/PostdocUW/proc_data/Y_SCA.csv')
    
    # remove inecessary columns
    X_WA = X_WA.drop(columns = ['Unnamed: 0','date'])
    Y_WA = Y_WA.drop(columns = ['Unnamed: 0', 'index', 'date'])
    
    X_OR = X_OR.drop(columns = ['Unnamed: 0','date'])
    Y_OR = Y_OR.drop(columns = ['Unnamed: 0', 'index', 'date'])
    
    X_NCA = X_NCA.drop(columns = ['Unnamed: 0','date'])
    Y_NCA = Y_NCA.drop(columns = ['Unnamed: 0', 'index', 'date'])
    
    X_SCA = X_SCA.drop(columns = ['Unnamed: 0','date'])
    Y_SCA = Y_SCA.drop(columns = ['Unnamed: 0', 'index', 'date'])
    
    # remove rows with nans
    X_WA = X_WA[1:31]
    Y_WA = Y_WA[1:31]
    
    X_OR = X_OR[1:31]
    Y_OR = Y_OR[1:31]
    
    X_NCA = X_NCA[1:31]
    Y_NCA = Y_NCA[1:31]
    
    X_SCA = X_SCA[1:31]
    Y_SCA = Y_SCA[1:31]
    
    # X_WA = X_WA[['meanBLT_precond_WA', 'meanBLT_spwn_WA', 'maxTLT_z1_WA', 'maxTLT_z3_WA',
    #              'maxTLT_mg_WA', 'meanLST_z1_WA', 'meanLST_z3_WA', 'meanLST_mg_WA',
    #              'meanCST_z1_WA', 'meanCST_z3_WA', 'meanCST_mg_WA']]
    
    # X_OR = X_OR[['meanBLT_precond_OR', 'meanBLT_spwn_OR', 'maxTLT_z1_OR', 'maxTLT_z3_OR',
    #              'maxTLT_mg_OR', 'meanLST_z1_OR', 'meanLST_z3_OR', 'meanLST_mg_OR',
    #              'meanCST_z1_OR', 'meanCST_z3_OR', 'meanCST_mg_OR']]
    
    # X_NCA = X_NCA[['meanBLT_precond_NCA', 'meanBLT_spwn_NCA', 'maxTLT_z1_NCA', 'maxTLT_z3_NCA',
    #              'maxTLT_mg_NCA', 'meanLST_z1_NCA', 'meanLST_z3_NCA', 'meanLST_mg_NCA',
    #              'meanCST_z1_NCA', 'meanCST_z3_NCA', 'meanCST_mg_NCA']]
    
    # X_SCA = X_SCA[['meanBLT_precond_SCA', 'meanBLT_spwn_SCA', 'maxTLT_z1_SCA', 'maxTLT_z3_SCA',
    #              'maxTLT_mg_SCA', 'meanLST_z1_SCA', 'meanLST_z3_SCA', 'meanLST_mg_SCA',
    #              'meanCST_z1_SCA', 'meanCST_z3_SCA', 'meanCST_mg_SCA']]
    
    ##
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
    
    ST = ['var_comb', 'gamma'] #, 'gaussian', 'negbinomial', 'tweedie']
    
    for state in range(0,4):
        X = model_input(state)
        Y = model_output(state)
        #list_comb =[]
        Data = pd.DataFrame([], columns = ST)
        for i in range(2,len(X.columns)+1): 
            lis = list(itertools.combinations(X.columns, i))
            #list_comb += [str(u) for u in lis]
            results =[]
            for combo in lis:
                #pool = mp.Pool(mp.cpu_count())           
                #results = [pool.apply(glm_comb, args =(Y, combo, X)) for combo in lis]  
                if check_corr(Y, combo, X) == True:
                   results += [[str(combo), glm_comb(Y, combo, X)]]
                #pool.close() 
            
            #results = np.array(results)
            #Data1 = pd.DataFrame(results, columns = ST)
            Data = Data.append(pd.DataFrame(results, columns = ST))
            
        #Data['Var_comb'] = list_comb
        #df1.to_csv("/home/rbani20/projects/def-guichard/rbani20/postdoc/data_proc/var_com"+str(state)+".csv", )
        Data.to_csv("/home/rbani20/projects/def-guichard/rbani20/postdoc/data_proc/data"+str(state)+".csv", )   
            
if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!



