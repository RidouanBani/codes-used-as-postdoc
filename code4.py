#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 23:00:36 2021

@author: N
"""

from pydap.client import open_url
import numpy as np
import multiprocessing as mp
import pandas as pd
import csv
import sys
sys.path.append('/home/rbani20/projects/def-guichard/rbani20/postdoc')
from BLT_LS_func import TLT_LS_func


WA_grid=[[[163,94],[165,101]],
         [[165,93],[166,101]],
         [[166,91],[168,101]],
         [[168,90],[169,101]],
         [[169,90],[171,101]],
         [[171,90],[173,98]],
         [[173,92],[174,98]],
         [[174,92],[175,97]],
         [[175,89],[178,97]],
         [[178,87],[180,95]]]
OR_grid=[[[120,93],[122,97]],         
         [[122,92],[127,96]],                
         [[127,91],[129,95]],         
         [[129,91],[132,96]],         
         [[132,92],[134,96]],         
         [[134,92],[135,97]],         
         [[135,93],[138,98]],         
         [[138,90],[144,99]],         
         [[144,91],[150,100]],
         [[150,92],[151,100]],
         [[151,93],[154,101]],
         [[154,92],[161,101]],
         [[161,93],[163,101]]]
NCA_grid=[[[88,100],[89,104]],         
         [[89,99],[90,103]],                
         [[90,100],[91,103]],         
         [[91,99],[93,103]],         
         [[93,99],[98,102]],          
         [[98,98],[99,101]],         
         [[99,97],[100,100]],         
         [[100,96],[101,99]],          
         [[101,95],[102,98]],
         [[102,92],[103,97]],
         [[103,95],[104,97]],
         [[104,93],[105,96]],
         [[105,93],[106,96]],
         [[106,94],[107,97]],
         [[107,94],[108,98]],
         [[108,94],[110,99]],
         [[110,95],[113,99]],
         [[113,94],[116,100]],
         [[116,94],[120,99]]]
SCA_grid=[[[27,138],[28,139]],
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
         [[41,133],[44,148]],
         [[44,132],[45,145]],
         [[45,131],[46,135]],
         [[47,125],[51,128]],
         [[46,129],[52,134]],   
         [[52,127],[55,131]],
         [[55,126],[56,130]],
         [[56,124],[58,128]],
         [[58,124],[59,126]],
         [[59,123],[61,125]],
         [[61,122],[62,124]],         
         [[62,119],[63,123]],                
         [[63,119],[66,121]],         
         [[66,119],[67,120]],         
         [[67,120],[68,122]],          
         [[68,118],[69,122]],         
         [[69,115],[70,122]],         
         [[70,113],[71,118]],          
         [[71,111],[72,117]],
         [[72,110],[75,116]],
         [[75,109],[77,115]],
         [[77,107],[78,115]],
         [[78,106],[79,115]],
         [[79,105],[80,113]],
         [[80,105],[83,111]],
         [[83,104],[84,110]],
         [[84,103],[85,109]],
         [[85,102],[86,107]],
         [[86,102],[87,106]],
         [[87,101],[88,105]]]

WA_grid_250 = np.zeros(np.array(WA_grid).shape).tolist()
for i in range(len(WA_grid_250)):
    WA_grid_250[i][0][1] = WA_grid[i][1][1]-25
    WA_grid_250[i][0][0] = WA_grid[i][0][0]
    WA_grid_250[i][1][0] = WA_grid[i][1][0]
    WA_grid_250[i][1][1] = WA_grid[i][1][1]
    

OR_grid_250 = np.zeros(np.array(OR_grid).shape).tolist()
for i in range(len(OR_grid_250)):
    OR_grid_250[i][0][1] = OR_grid[i][1][1]-25
    OR_grid_250[i][0][0] = OR_grid[i][0][0]
    OR_grid_250[i][1][0] = OR_grid[i][1][0]
    OR_grid_250[i][1][1] = OR_grid[i][1][1]
    
    
NCA_grid_250 = np.zeros(np.array(NCA_grid).shape).tolist()
for i in range(len(NCA_grid_250)):
    NCA_grid_250[i][0][1] = NCA_grid[i][1][1]-25
    NCA_grid_250[i][0][0] = NCA_grid[i][0][0]
    NCA_grid_250[i][1][0] = NCA_grid[i][1][0]
    NCA_grid_250[i][1][1] = NCA_grid[i][1][1]
    
    
SCA_grid_250 = np.zeros(np.array(SCA_grid).shape).tolist()
for i in range(len(SCA_grid_250)):
    SCA_grid_250[i][0][1] = SCA_grid[i][1][1]-25
    SCA_grid_250[i][0][0] = SCA_grid[i][0][0]
    SCA_grid_250[i][1][0] = SCA_grid[i][1][0]
    SCA_grid_250[i][1][1] = SCA_grid[i][1][1]


with open('datasets.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    datasets = list (csv_reader)

# Parallelizing using Pool.apply()

# Step 1: Init multiprocessing.Pool()
#pool = mp.Pool(40)
# Step 2: `pool.apply` the `BLT_LS_func()`
#results = [pool.apply(TLT_LS_func, args=(u, WA_grid_250, OR_grid_250, NCA_grid_250, SCA_grid_250)) for u in datasets]
#results = [TLT_LS_func(u, WA_grid_250, OR_grid_250, NCA_grid_250, SCA_grid_250) for u in datasets]
# Step 3: Don't forget to close
#pool.close()  


ST=['count','date','WA_TLT','OR_TLT','NCA_TLT','SCA_TLT','WA_LS','OR_LS','NCA_LS','SCA_LS','WA_CS','OR_CS','NCA_CS','SCA_CS'];

l1 = [0,942]
l2 = [942,1884]
l3 = [1884,2827]
lall = [0, 2827]
l = l3

Data_250 = pd.DataFrame([], columns = ST)
for k in range(l[0],l[1]):
    u = datasets[k]
    results = [k,u[0][17:25]]+TLT_LS_func(u, WA_grid_250, OR_grid_250, NCA_grid_250, SCA_grid_250)
    results = np.array(results)
    Data_250 = Data_250.append(pd.Series(results, index=ST), ignore_index=True)
    Data_250.to_csv("data/Data_250_"+str(l[0])+"_to_"+str(l[1])+".csv", )