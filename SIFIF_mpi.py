#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:54:40 2021

@author: Ridouan Bani
This codes contains 4 functions
the top 2 are the main fucntions
The first function apply the model selection on randomly selected training data set and then 
evaluate the models on test  data set and returns the best (up to 5) models with low Delta AICC 
and Delat R2.
The second function check for collinearity and retun possible variables combinations
with control corelations.
"""

import itertools
import numpy as np
from random import sample
import statsmodels.api as sm
import pandas as pd
import multiprocessing as mp
from pandas import read_csv
import pickle


def  SIFIF_func(X, N_var, Cor_var, list_var_exclude, training_percent, results):

    # This fucntion randomly sample traing and testing data sets  
    # then applies OLS method to all possible variables combinations 
    # and returns top best (up to five models) from all possible combinations

    
    
    # X ................... data including Y response
    # N_var ............... maximum number of variables in a model
    # Cor_var ............. correlation threshold between variables to avoid collinearity
    # list_var_exclude..... list of variables name to exclude from the analysis
    # training_percent .... the percentage of data that will be trawn randomly for training data

    
    # find training and testing set and index
    training_index = sample(list(X.index),round(len(list(X.index))*training_percent))
    testing_index = list(set(list(X.index))-set(training_index))
        
    # make an empty Dataframe to return the results
    # the goal is to have a table with all variables 
    # All NANa except when variable is included in the model
    # on top of const, the table should have important metrics such 
    # as AICc, R2_train, R2_adj_train, R2_test, R2_adj_test
    X_without_list_var_exclude = X.drop(columns = list_var_exclude)
    Res = pd.DataFrame([], columns = ['const'] + list(X_without_list_var_exclude) +["R2_train", "R2adj_train", "AICc", "R2_test","R2adj_test"])
    
    # go throughout the models 
    for model_var in results:
        
        # restrecting the data to model only
        X_training = X.loc[training_index,list(model_var)]
        X_testing  = X.loc[testing_index,list(model_var)]
        
        Y_training = X.loc[training_index,"Y"]
        Y_testing = X.loc[testing_index,"Y"]
        
        Y_SD_train = X.loc[training_index,"Y_SD"]
        
        
        ## apply the random sampling of Y after transforming Y
        
        Y_log_train = np.log(gen_lognor_Y(Y_training,Y_SD_train,training_index))
        Y_log_testing   = np.log(Y_testing)
        
        # add const to input matrix
        X_in = sm.add_constant(X_training)
        # rest_index in X_in
        X_in = X_in.reset_index()
        # drop the 'index' column
        X_in = X_in.drop(columns = ['index'])
        
        # apply ordinary least square method (OLS) to log Y 
        model     = sm.OLS(Y_log_train , X_in).fit()
        
        res1                  = pd.DataFrame(model.params).T
        res1['R2adj_train']   = model.rsquared_adj
        res1["R2_train"]      = model.rsquared
        res1['AICc']          = sm.tools.eval_measures.aicc(model.llf,model.nobs,len(model.params))
        
        ## NOW the long calculation of R2 on test data (R2adj_test)
        ## first SS.total
        SS_total     =   sum((Y_log_testing.values-np.mean(Y_log_testing.values))*np.transpose(Y_log_testing.values-np.mean(Y_log_testing.values))) 
        SS_residual  = sum((Y_log_testing.values-model.predict(sm.add_constant(X_testing)))*np.transpose(Y_log_testing.values-model.predict(sm.add_constant(X_testing))))
        
        R2_testing   = 1-(SS_residual/SS_total)
        R2adj_test   = 1-(((1-R2_testing)*(model.nobs-1))/(model.nobs-len(model.params)-1))
        
        # Add values to putput
        res1['R2_test']      = R2_testing
        res1['R2adj_test']   = R2adj_test
        
        Res  = Res.append(res1)
        
    # now after all the potential models are evaluated 
    # we sort the resulrs Res by AICc
    Res['DeltaAICc'] = Res['AICc']-min(Res['AICc'])
    Res['DeltaR2']   = Res['R2adj_test']-Res['R2adj_train']
    
    # sorte dataframe from small to big delta AICc
    Res = Res.sort_values('DeltaAICc')
    Res = Res[Res['DeltaAICc']<2]
    Res = Res.sort_values('DeltaR2',ascending=False)
    if len(Res)>5:
        Res = Res.loc[0:5,]
    
    # Sort agin using Delta R2
    #Res = Res.sort_values('DeltaAICc',ascending=False)
    
    return(Res.values)
    
def Var_comb(X,N_var,Cor_var,list_var_exclude):
    # This function returns all possible combinations of variables with lower collinearity
    # This result of this function will be feed to function SIFIF_func
    
    # X .................. data 
    # N_var .............. number of variables in a combination
    # Cor_var ............ correlation threshold 
    # list_var_exclude ... list of variables to exludes from the process (e.g. Y, index,...)
    
    
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
def gen_lognor_Y(Y_training,Y_SD_train,training_index):
    Y_n= [np.random.normal(Y_training[i], Y_SD_train[i], 1)[0] for i in training_index]        
    Y_new = pd.DataFrame(Y_n, columns = ['Y'])
    return Y_new

######################################################################
# Import ra data          
#X_WA = read_csv("/Users/Documents/PostdocUW/proc_data/X_WA.csv")
X_WA = read_csv("/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/X_WA.csv")    
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
#######################################################################
# run the codes


if __name__ == '__main__':
    
    with open("/home/rbani20/projects/def-guichard/rbani20/postdoc/comb_var.txt", "rb") as fp:
        results = pickle.load(fp)
    
    pool = mp.Pool(40)
    
    Results1= [pool.apply_async(SIFIF_func, args =(X, 5, 0.75, ["Y","Y_SD","year"], 0.8, results)) for i in range(0,1001)]
    
    Results2 = [R.get() for R in Results1]
    pool.close()
    pool.join()
    
    Results3 =[j for i in Results2 for j in i]
    X_without_list_var_exclude = X.drop(columns = ["Y","Y_SD","year"])
    Results = pd.DataFrame(Results3, columns = ['const'] + list(X_without_list_var_exclude)+["R2_train", "R2adj_train", "AICc", "R2_test","R2adj_test","DeltaAICc","DeltaR2"])


    Results.to_csv("/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/WA_results_MSP1000.csv", )
    #Results.to_csv("/Users/Documents/PostdocUW/proc_data/WA_results_MSP.csv", )
            
  
            

            
