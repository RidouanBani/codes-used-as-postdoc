#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:28:52 2021

@author: N
"""

from pydap.client import open_url
from pydap.cas.urs import setup_session
import numpy as np
from siphon.catalog import TDSCatalog
from siphon import catalog, ncss
import pickle
import multiprocessing as mp
import pandas as pd

WA_grid_LS=[[[163,98],[169,100]],
         [[169,97],[173,99]],
         [[173,95],[178,97]],
         [[178,93],[180,95]]]

OR_grid_LS=[[[120,95],[122,97]],         
         [[122,95],[127,96]],                
         [[127,94],[129,95]],         
         [[129,94],[132,96]],         
         [[132,95],[134,96]],         
         [[134,96],[135,97]],         
         [[135,97],[138,98]],         
         [[138,98],[144,99]],         
         [[144,99],[151,100]],
         [[151,99],[163,101]]]

NCA_grid_LS=[[[88,103],[89,104]],         
         [[89,102],[90,103]],                
         [[90,101],[93,103]],                
         [[93,101],[98,102]],          
         [[98,100],[99,101]],         
         [[99,99],[100,100]],         
         [[100,98],[101,99]],          
         [[101,97],[102,98]],
         [[102,96],[104,97]],
         [[104,95],[106,96]],
         [[106,96],[107,97]],
         [[107,97],[108,98]],
         [[108,98],[113,99]],
         [[113,98],[116,100]],
         [[116,97],[120,99]]]

SCA_grid_LS=[[[27,138],[28,139]],
         [[30,136],[32,139]],
         [[33,137],[34,138]],
         [[36,137],[39,142]],
         [[32,141],[35,147]],
         [[28,143],[31,148]],
         [[25,142],[28,145]],
         [[25,146],[28,150]],
         [[27,153],[31,157]],
         [[25,159],[29,163]],
         [[25,165],[32,169]],
         [[32,163],[34,166]],
         [[32,152],[36,159]],
         [[33,147],[38,151]],
         [[34,157],[37,163]],
         [[37,153],[39,156]],
         [[39,134],[41,156]],
         [[41,146],[44,148]],
         [[44,134],[45,145]],
         [[45,133],[46,135]],
         [[47,125],[51,128]],
         [[46,132],[52,134]],   
         [[52,130],[55,131]],
         [[55,128],[56,130]],
         [[56,126],[58,128]],
         [[58,124],[59,126]],
         [[59,123],[61,125]],
         [[61,122],[62,124]],         
         [[62,121],[63,123]],                
         [[63,120],[66,121]],         
         [[66,120],[67,122]],         
         [[67,121],[68,122]],          
         [[68,121],[69,122]],         
         [[69,118],[70,122]],         
         [[70,117],[71,118]],      
         [[71,116],[72,117]],
         [[72,115],[75,116]],
         [[75,114],[77,115]],
         [[77,114],[78,115]],
         [[78,113],[79,115]],
         [[79,111],[80,113]],
         [[80,109],[83,111]],
         [[83,109],[84,110]],
         [[84,108],[85,109]],
         [[85,106],[86,107]],
         [[86,105],[87,106]],
         [[87,104],[88,105]]]

def read_temp(u, Temp):
    T_store = []
    T_store += Temp[0:16,1,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
    reture T_store

def BLT_LS_func(u, WA_grid_LS, OR_grid_LS, NCA_grid_LS, SCA_grid_LS):
    
    opendap_url = "https://oceanmodeling.ucsc.edu:8443/thredds/dodsC/wc12.0_ccsra31_01/posterior/"+str(u)
    dataset = open_url(opendap_url)
    Temp = dataset['temp']    
    
    # WA bottom Temperature  in shallow watters 0-10km off-shore 
    T_store1 =[]
    for ii in WA_grid_LS:
        T_store1+=Temp[0:16,1,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
    T_store1 = np.array(T_store1)
    T_store1 = T_store1.astype('float')
    T_store1[T_store1>50] = np.nan
    
    # OR bottom Temperature  in shallow watters 0-10km off-shore 
    T_store2 =[]
    for ii in OR_grid_LS:
        T_store2+=Temp[0:16,1,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
    T_store2 = np.array(T_store2)
    T_store2 = T_store2.astype('float')
    T_store2[T_store2>50] = np.nan

    
    # NCA bottom Temperature in shallow watters 0-10km off-shore 
    T_store3 =[]
    for ii in NCA_grid_LS:
        T_store3+=Temp[0:16,1,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
    T_store3 = np.array(T_store3)
    T_store3 = T_store3.astype('float')
    T_store3[T_store3>50] = np.nan
    
    # SCA   bottom Temperature in shallow watters 0-10km off-shore 
    T_store4 =[]
    for ii in SCA_grid_LS:
        T_store4+=Temp[0:16,1,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
    T_store4 = np.array(T_store4)
    T_store4 = T_store4.astype('float')
    T_store4[T_store4>50] = np.nan
    
    BLT_LS =[ np.nanmean(T_store1), np.nanmean(T_store2), np.nanmean(T_store3),np.nanmean(T_store4)]
    return BLT_LS
    
data_url = ('https://oceanmodeling.ucsc.edu:8443/thredds/catalog/wc12.0_ccsra31_01/posterior/catalog.xml')

cat = catalog.TDSCatalog(data_url)
datasets = list(cat.datasets)

# Parallelizing using Pool.apply()

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `BLT_LS_func()`
ST=['WA','OR','NCA','SCA'];
results = [pool.apply(BLT_LS_func, args=(u, WA_grid_LS, OR_grid_LS, NCA_grid_LS, SCA_grid_LS)) for u in datasets]

# Step 3: Don't forget to close
pool.close()  


results = np.array(results)
BLT = pd.DataFrame(results, columns = ST)
BLT.to_csv("BLT.csv", )
