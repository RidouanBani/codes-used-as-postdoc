#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:39:27 2021

@author: N
"""
import itertools
from pandas import read_csv
import pickle



def Var_comb(X,N_var,Cor_var,list_var_exclude):
    
    # remove the unwanted variabes from data including Y, date,...
    X_without_list_var_exclude = X.drop(columns = list_var_exclude)
    # find the names of remaining variables
    list_col = list(X_without_list_var_exclude)
    # now iterate among the remaining variables to form combinations 
    
    # first create empty results container
    results = []
    for i in list(range(1,N_var+1)): 
        
        list_var = list(itertools.combinations(list_col, i))
        # for combinations of a single variable no need to check for Cor_var
        if i ==1:
           for var_name in list_var:
               # append single variable models to results empty container
               results += [var_name]
        # if i>2 need to check for correlation between variabels 
        # any model with high correlations than the threshold is not included 
        # in results models container
        else:
            for var_name in list_var:
                # find correlation matrix
                corr_var_name = X[list(var_name)].corr()
                # replace diagonals with 0
                corr_var_name[corr_var_name == 1] = 0
                # fill correlation higher than threshold with 1 
                corr_var_name[abs(corr_var_name) >= Cor_var] = 1
                # fill correlation lower than threshold with 0 
                corr_var_name[abs(corr_var_name) < Cor_var] = 0
                
                if sum(sum(corr_var_name.values)) < 1:
                    results+= [var_name]
    return(results)
  
X_WA = read_csv("/Users/Documents/PostdocUW/proc_data/X_SCA.csv")
#X_WA = read_csv("/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/X_WA.csv")    
#######################################################################  
# Clean Data 
list_col = list(X_WA)
Xt1 = X_WA[list_col[1:len(list_col)-2]]
Xt4 = X_WA[list_col[len(list_col)-2:]]

Xt1 = Xt1[1:31].reset_index()
Xt4 = Xt4[5:35].reset_index()

Xt1 = Xt1.drop(columns = ['index'])
Xt4 = Xt4.drop(columns = ['index'])

X = Xt1.join(Xt4, how='left')

X['year']=list(range(1985,2015))

comb_var_ncaor = Var_comb(X,5,0.75,["Y","Y_SD","year"])

with open("/Users/Documents/PostdocUW/proc_data/comb_var_ncaor.txt", "wb") as fp:
    pickle.dump(comb_var_ncaor, fp)
