#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:54:40 2021

@author: Ridouan Bani
function similar to dgredge function in R
first I have to build a function to form all possible independent variables combination.
the number of possible variables in a model is controlled (N_var) as well as 
the level of correlations (Cor_var) between the variables of a potential model.
The control of Cor_var to avoid collinearity in model variables.
Then function to calculate the model performance and the compare
"""

import itertools
import numpy as np
from random import sample
import statsmodels.api as sm
import pandas as pd
import multiprocessing as mp
from pandas import read_csv
import timeit


def  SIFIF_func(X, N_var, Cor_var, list_var_exclude, training_percent):
    # X data including Y response
    # N_var maximum number of variables in a model
    # Cor_var correlation threshold between variables to avoid collinearity
    # list_var_exclude list of variables name to exclude from the analysis
    # training_percent the percentage of data that will be trawn randomly for training data

    
    # find training and testing set and index
    training_index = sample(list(X.index),round(len(list(X.index))*training_percent))
    testing_index = list(set(list(X.index))-set(training_index))
    
    # find all possible controled variables combinations
    results = Var_comb(X, N_var, Cor_var, list_var_exclude)
    
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
        
        # # reset_index for Y_trainign and Y_SD_train
        # Y_training = Y_training.reset_index()
        # Y_SD_train = Y_SD_train.reset_index()
        
        # # drop index column
        # Y_training = Y_training.drop(columns = ['index'])
        # Y_SD_train = Y_SD_train.drop(columns = ['index'])
        
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
            
def get_result(result):
    global Results 
    Results = pd.DataFrame([], columns = ['const'] + ['meanBLT_precond',
     'meanBLT_spwn',
     'maxTLT_z1',
     'meanLST_z1',
     'meanCST_z1',
     'maxTLT_z2',
     'meanLST_z2',
     'meanCST_z2',
     'maxTLT_z3',
     'meanLST_z3',
     'meanCST_z3',
     'maxTLT_z4',
     'meanLST_z4',
     'meanCST_z4',
     'maxTLT_z5',
     'meanLST_z5',
     'meanCST_z5',
     'maxTLT_mg',
     'meanLST_mg',
     'meanCST_mg',
     'meanBLT_jvnl0']+["R2_train", "R2adj_train", "AICc", "R2_test","R2adj_test"])
    Results.append(result)
    
    return(Results)


######################################################################
# Import ra data          
X_WA = read_csv("/Users/Documents/PostdocUW/proc_data/X_WA.csv")
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
#######################################################################
# run the codes

start = timeit.default_timer()
if __name__ == '__main__':
    
    pool = mp.Pool(10)
    
    Results1= [pool.apply_async(SIFIF_func, args = (X, 5, 0.75, ["Y","Y_SD","year"], 0.8)) for i in range(0,1)]
    
    Results2 = [R.get() for R in Results1]
    pool.close()
    pool.join()
    
    Results = pd.DataFrame(Results2[0], columns = ['const'] + ['meanBLT_precond',
                                                    'meanBLT_spwn',
                                                    'maxTLT_z1',
                                                    'meanLST_z1',
                                                    'meanCST_z1',
                                                    'maxTLT_z2',
                                                    'meanLST_z2',
                                                    'meanCST_z2',
                                                    'maxTLT_z3',
                                                    'meanLST_z3',
                                                    'meanCST_z3',
                                                    'maxTLT_z4',
                                                    'meanLST_z4',
                                                    'meanCST_z4',
                                                    'maxTLT_z5',
                                                    'meanLST_z5',
                                                    'meanCST_z5',
                                                    'maxTLT_mg',
                                                    'meanLST_mg',
                                                    'meanCST_mg',
                                                    'meanBLT_jvnl0']+["R2_train", "R2adj_train", "AICc", "R2_test","R2adj_test","DeltaAICc","DeltaR2"])


    #Results.to_csv("/home/rbani20/projects/def-guichard/rbani20/postdoc/proc_data/WA_results_MSP.csv", )
    Results.to_csv("/Users/Documents/PostdocUW/proc_data/WA_results_MSP.csv", )
            

stop = timeit.default_timer()
print('Time: ', stop - start)
            
