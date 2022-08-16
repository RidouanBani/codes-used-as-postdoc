#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:47:11 2021

@author: N
"""

from pydap.client import open_url
import numpy as np
import pandas as pd
import csv

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

WA_grid_150 = np.zeros(np.array(WA_grid).shape).tolist()
for i in range(len(WA_grid_150)):
    WA_grid_150[i][0][1] = WA_grid[i][1][1]-15
    WA_grid_150[i][0][0] = WA_grid[i][0][0]
    WA_grid_150[i][1][0] = WA_grid[i][1][0]
    WA_grid_150[i][1][1] = WA_grid[i][1][1]
    

OR_grid_150 = np.zeros(np.array(OR_grid).shape).tolist()
for i in range(len(OR_grid_150)):
    OR_grid_150[i][0][1] = OR_grid[i][1][1]-15
    OR_grid_150[i][0][0] = OR_grid[i][0][0]
    OR_grid_150[i][1][0] = OR_grid[i][1][0]
    OR_grid_150[i][1][1] = OR_grid[i][1][1]
    
    
NCA_grid_150 = np.zeros(np.array(NCA_grid).shape).tolist()
for i in range(len(NCA_grid_150)):
    NCA_grid_150[i][0][1] = NCA_grid[i][1][1]-15
    NCA_grid_150[i][0][0] = NCA_grid[i][0][0]
    NCA_grid_150[i][1][0] = NCA_grid[i][1][0]
    NCA_grid_150[i][1][1] = NCA_grid[i][1][1]
    
    
SCA_grid_150 = np.zeros(np.array(SCA_grid).shape).tolist()
for i in range(len(SCA_grid_150)):
    SCA_grid_150[i][0][1] = SCA_grid[i][1][1]-15
    SCA_grid_150[i][0][0] = SCA_grid[i][0][0]
    SCA_grid_150[i][1][0] = SCA_grid[i][1][0]
    SCA_grid_150[i][1][1] = SCA_grid[i][1][1]

def TLT_LS_CS_150_func(u, WA_grid_150, OR_grid_150, NCA_grid_150, SCA_grid_150):
    
    opendap_url = "https://oceanmodeling.ucsc.edu:8443/thredds/dodsC/wc12.0_ccsra31_01/posterior/"+str(u)
    dataset = open_url(opendap_url)
    
    Temp = dataset['temp']
    U_momntum = dataset['u']
    V_momntum = dataset['v']

     # WA top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store1 =[]
    CS_store1 =[]
    LS_store1 =[]
    for ii in WA_grid_150:
        T_store1+=Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        LS_store1+=U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        CS_store1+=V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        
    T_store1 = np.array(T_store1)
    T_store1 = T_store1.astype('float')
    T_store1[T_store1>50] = np.nan  

    LS_store1 = np.array(LS_store1)
    LS_store1 = LS_store1.astype('float')
    LS_store1[LS_store1>50] = np.nan  

    CS_store1 = np.array(CS_store1)
    CS_store1 = CS_store1.astype('float')
    CS_store1[CS_store1>50] = np.nan  

    
    # OR top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store2 =[]
    CS_store2 =[]
    LS_store2 =[]
    for ii in OR_grid_150:
        T_store2+=Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        LS_store2+=U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        CS_store2+=V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        
    T_store2 = np.array(T_store2)
    T_store2 = T_store2.astype('float')
    T_store2[T_store2>50] = np.nan  

    LS_store2 = np.array(LS_store2)
    LS_store2 = LS_store2.astype('float')
    LS_store2[LS_store2>50] = np.nan  

    CS_store2 = np.array(CS_store2)
    CS_store2 = CS_store2.astype('float')
    CS_store2[CS_store2>50] = np.nan  

    
    # NCA top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store3 =[]
    CS_store3 =[]
    LS_store3 =[]
    for ii in NCA_grid_150:
        T_store3+=Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        LS_store3+=U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        CS_store3+=V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        
    T_store3 = np.array(T_store3)
    T_store3 = T_store3.astype('float')
    T_store3[T_store3>50] = np.nan  

    LS_store3 = np.array(LS_store3)
    LS_store3 = LS_store3.astype('float')
    LS_store3[LS_store3>50] = np.nan  

    CS_store3 = np.array(CS_store3)
    CS_store3 = CS_store3.astype('float')
    CS_store3[CS_store3>50] = np.nan  

    
    # SCA top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store4 =[]
    CS_store4 =[]
    LS_store4 =[]
    for ii in SCA_grid_150:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store4+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store4+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store4+=vemp.data.flatten().tolist()
    T_store4 = np.array(T_store4)
    T_store4 = T_store4.astype('float')
    T_store4[T_store4>50] = np.nan  

    LS_store4 = np.array(LS_store4)
    LS_store4 = LS_store4.astype('float')
    LS_store4[LS_store4>50] = np.nan  

    CS_store4 = np.array(CS_store4)
    CS_store4 = CS_store4.astype('float')
    CS_store4[CS_store4>50] = np.nan  
    
    ALL_CS = [ np.nanmean(T_store1), np.nanmean(T_store2), np.nanmean(T_store3), np.nanmean(T_store4), np.nanmean(LS_store1), np.nanmean(LS_store2), np.nanmean(LS_store3), np.nanmean(LS_store4), np.nanmean(CS_store1), np.nanmean(CS_store2), np.nanmean(CS_store3), np.nanmean(CS_store4)] 
 
    return ALL_CS

with open('datasets.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    datasets = list (csv_reader)


ST=['WA_TLT','OR_TLT','NCA_TLT','SCA_TLT','WA_LS','OR_LS','NCA_LS','SCA_LS','WA_CS','OR_CS','NCA_CS','SCA_CS'];
results = [TLT_LS_CS_150_func(u, WA_grid_150, OR_grid_150, NCA_grid_150, SCA_grid_150) for u in datasets]


results = np.array(results)
Data_150 = pd.DataFrame(results, columns = ST)
Data_150.to_csv("Data_150_serial.csv", )
