#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:31:19 2021

@author: N
"""

from pydap.client import open_url
import numpy as np
import datetime
from netCDF4 import Dataset
import multiprocessing as mp
import pandas as pd
    
def BLT_LS_func(u, WA_grid_LS, OR_grid_LS, NCA_grid_LS, SCA_grid_LS):  
    print(f'processing {u[0]}')
    opendap_url = "https://oceanmodeling.ucsc.edu:8443/thredds/dodsC/wc12.0_ccsra31_01/posterior/"+str(u[0])
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

def TLT_LS_func(u, WA_grid, OR_grid, NCA_grid, SCA_grid):
    print(f'processing {u[0]}')
    
    opendap_url = "https://oceanmodeling.ucsc.edu:8443/thredds/dodsC/wc12.0_ccsra31_01/posterior/"+str(u[0])
    dataset = open_url(opendap_url)
    
    Temp = dataset['temp']
    U_momntum = dataset['u']
    V_momntum = dataset['v']
    
     # WA top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store1 =[]
    CS_store1 =[]
    LS_store1 =[]
    for ii in WA_grid:
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
    for ii in OR_grid:
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
    for ii in NCA_grid:
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
    for ii in SCA_grid:
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

def TLT_func(l, datasets, Grid, area_number):
    results = []
    for k in range(l[0],l[1]):
        u = datasets[k]
        print(f'processing {u[0]}')
    
        opendap_url = "https://oceanmodeling.ucsc.edu:8443/thredds/dodsC/wc12.0_ccsra31_01/posterior/"+str(u[0])
        dataset = open_url(opendap_url)
    
        Temp = dataset['temp']
        U_momntum = dataset['u']
        V_momntum = dataset['v']
 
        for area in range(len(Grid)):
            grid = Grid[area]
            T_store2 =[]
            CS_store2 =[]
            LS_store2 =[]
            for ii in grid:
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
  
            results += [[u[0][17:25]] + [area_number[area], np.nanmean(T_store2), np.nanmean(LS_store2), np.nanmean(CS_store2)]]
    results = np.array(results)
    Data = pd.DataFrame(results, columns = ['date','area','TLT','TLS','TCS'])
    Data.to_csv("data_shrimp/TL_per_area_shrimp_150"+str(l[0])+"_to_"+str(l[1])+".csv", )  

def TLT_LS_nc_func(day, Grid):
    print(f'processing day = {day}')
    Data = Dataset('ROMS_data_nc.nc', 'r')
    Temp = Data.variables['temp']
    U_momntum = Data.variables['u']
    V_momntum = Data.variables['v']
    
     # WA top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store1 =[]
    CS_store1 =[]
    LS_store1 =[]
    for ii in Grid:
        T_store1  += extract_func(day,Temp,ii)
        LS_store1 += extract_func(day,U_momntum,ii)
        CS_store1 += extract_func(day,V_momntum,ii)
    
    T_store1 = np.array(T_store1)
    #T_store1 = T_store1.astype('float')
    T_store1 = T_store1[~np.isnan(T_store1)] 
    T_store1[T_store1>50] = np.nan  

    LS_store1 = np.array(LS_store1)
    #LS_store1 = LS_store1.astype('float')
    LS_store1 = LS_store1[~np.isnan(LS_store1)] 
    LS_store1[LS_store1>50] = np.nan  

    CS_store1 = np.array(CS_store1)
    #CS_store1 = CS_store1.astype('float')
    CS_store1 = CS_store1[~np.isnan(CS_store1)] 
    CS_store1[CS_store1>50] = np.nan  
    
    ALL_CS = [np.nanmean(T_store1), np.nanmean(LS_store1), np.nanmean(CS_store1)] 
    Data.close()
    return ALL_CS

def extract_func(day,V,ii):
    return V[day:day+3,0,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()

def put_together_nc(day,Grid1, Grid2, Grid3, Grid4):
    return [datetime.datetime.strptime('2011-01-02', "%Y-%m-%d") + datetime.timedelta(days=day)] + TLT_LS_nc_func(day, Grid1) + TLT_LS_nc_func(day, Grid2) + TLT_LS_nc_func(day, Grid3) + TLT_LS_nc_func(day, Grid4)

def put_together_nc2(day,Grid1, Grid2, Grid3, Grid4):
    return [datetime.datetime.strptime('2011-01-02', "%Y-%m-%d") + datetime.timedelta(days=day)] + [BLT_nc_func(day, Grid1), BLT_nc_func(day, Grid2), BLT_nc_func(day, Grid3), BLT_nc_func(day, Grid4)]

def BLT_nc_func(day, Grid):
    print(f'processing day = {day}')
    Data = Dataset('ROMS_data_2.nc', 'r')
    Temp = Data.variables['temp']
    
     # WA top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store1 =[]

    for ii in Grid:
        T_store1  += extract_func(day,Temp,ii)

    
    T_store1 = np.array(T_store1)
    #T_store1 = T_store1.astype('float')
    T_store1 = T_store1[~np.isnan(T_store1)] 
    T_store1[T_store1>50] = np.nan  
    
    ALL_CS = np.nanmean(T_store1)
    Data.close()
    return ALL_CS

def Temp_per_grid(ii,Temp):
    return Temp[0:16,1,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()

def temp_per_area(u, area, area_number, Grid, Temp):
    T_store1 =[]
    grid = Grid[area]
    for ii in grid:
        T_store1+=Temp_per_grid(ii,Temp)
    
    T_store1 = np.array(T_store1)
    T_store1 = T_store1.astype('float')
    T_store1[T_store1>50] = np.nan
    return [[u[0][17:25]] + [area_number[area], np.nanmean(T_store1)]]
    

def BLT_extract(l, datasets, Grid, area_number): 
    results = []
    for k in range(l[0],l[1]):
        u = datasets[k]
        print(f'processing {u[0]}')
        opendap_url = "https://oceanmodeling.ucsc.edu:8443/thredds/dodsC/wc12.0_ccsra31_01/posterior/"+str(u[0])
        dataset = open_url(opendap_url, timeout=200)
        Temp = dataset['temp'] 
        
        for area in range(len(Grid)):
            results+= temp_per_area(u, area, area_number, Grid, Temp)
 
        
    results = np.array(results)
    Data = pd.DataFrame(results, columns = ['date','area','BLT'])
    Data.to_csv("data_shrimp/BLT_per_area_shrimp_"+str(l[0])+"_to_"+str(l[1])+".csv", ) 