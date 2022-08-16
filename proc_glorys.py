#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:58:11 2022

@author: rbani20
"""

import os
import numpy as np
from netCDF4 import Dataset
import pandas as pd

os.chdir('/Users/rbani20/Documents/PostdocUW/GLORYS/opernicus-marine-data/regrid')

years = list(range(1993,2020))

apr       = ["-04-01","-04-30"]
may       = ["-05-01","-05-31"]
june      = ["-06-01","-06-30"]
july      = ["-07-01","-07-31"]
august    = ["-08-01","-08-31"]

grid_32=[[[172,91],[173,94]],
         [[173,92],[174,94]],
         [[174,91],[175,94]],
         [[175,90],[176,93]],
         [[176,89],[178,92]],
         [[178,88],[179,92]],
         [[179,85],[180,91]]]

grid_30 =[[[170,90],[172,94]],
          [[168,91],[170,95]],
          [[167,91],[168,96]],
          [[166,92],[167,96]],
          [[164,94],[166,96]],
          [[163,94],[164,97]],
          ]
          

grid_29 =[
    [[162,95],[163,97]],
    [[160,93],[162,98]],
    ]

grid_28 = [
    [[159,93],[160,98]],
    [[158,93],[159,99]],
    [[157,94],[158,99]]
    ]

grid_26 =[
    [[156,95],[157,99]],
    [[153,96],[156,99]],
    [[151,97],[153,99]],
    [[149,96],[151,99]]
    ]

grid_24 =[
    [[148,95],[149,98]],
    [[147,94],[148,97]],
    [[146,93],[147,97]],
    [[145,94],[146,95]],
    [[143,92],[145,95]],
    [[140,90],[143,96]]
    ]

grid_22 = [
    [[139,92],[140,97]],
    [[138,94],[139,96]],
    [[135,94],[138,97]],
    [[134,94],[135,96]],
    [[133,93],[134,96]],
    [[132,93],[133,95]],
    [[131,93],[132,94]]
    ]

grid_21 = [
    [[130,91],[131,95]],
    [[129,91],[130,94]],
    [[128,92],[129,94]],
    [[127,93],[128,94]],
    [[126,93],[127,95]]
    ]

grid_20 = [
    [[125,93],[126,94]],
    [[124,92],[125,94]],
    [[123,93],[124,95]],
    [[122,94],[123,96]]
    ]

grid_19 = [
    [[119,94],[122,96]],
    [[118,95],[119,96]]
    ]

grid_18 = [
    [[113,95],[118,97]],
    [[109,96],[113,97]],
    [[108,95],[109,97]],
    [[107,95],[108,96]],
    [[107,95],[108,96]],
    [[106,94],[107,96]],
    [[105,93],[106,95]],
    [[104,94],[105,96]],
    [[103,95],[104,96]]    
    ]

grid_12 = [
    [[102,95],[103,96]],
    [[101,96],[102,98]],
    [[99,98],[101,99]],
    [[97,99],[99,101]],
    [[94,100],[97,102]]
    ]

grid_32_150=[[[172,83],[173,98]],
         [[173,82],[174,97]],
         [[174,82],[175,97]],
         [[175,82],[176,97]],
         [[176,81],[178,96]],
         [[178,80],[179,95]],
         [[179,79],[180,94]]]

grid_30_150 =[[[170,84],[172,99]],
          [[168,84],[170,99]],
          [[167,84],[168,99]],
          [[166,84],[167,99]],
          [[164,84],[166,99]],
          [[163,84],[164,99]],
          ]

grid_29_150 =[
    [[162,85],[163,100]],
    [[160,85],[162,100]],
    ]
          
grid_28_150 = [
    [[159,85],[160,100]],
    [[158,85],[159,100]],
    [[157,85],[158,100]]
    ]

grid_26_150 =[
    [[156,85],[157,100]],
    [[153,85],[156,100]],
    [[151,85],[153,100]],
    [[149,85],[151,100]]
    ]

grid_24_150 =[
    [[148,84],[149,99]],
    [[147,84],[148,99]],
    [[146,84],[147,99]],
    [[145,84],[146,99]],
    [[143,84],[145,99]],
    [[140,84],[143,99]]
    ]

grid_22_150 = [
    [[139,83],[140,98]],
    [[138,83],[139,98]],
    [[135,83],[138,98]],
    [[134,82],[135,97]],
    [[133,82],[134,97]],
    [[132,81],[133,96]],
    [[131,81],[132,96]]
    ]

grid_21_150 = [
    [[130,80],[131,95]],
    [[129,80],[130,95]],
    [[128,80],[129,95]],
    [[127,80],[128,95]],
    [[126,80],[127,95]]
    ]

grid_20_150 = [
    [[125,81],[126,96]],
    [[124,81],[125,96]],
    [[123,81],[124,96]],
    [[122,81],[123,96]]
    ]

grid_19_150 = [
    [[119,82],[122,97]],
    [[118,83],[119,98]]
    ]

grid_18_150 = [
    [[113,84],[118,99]],
    [[109,84],[113,99]],
    [[108,84],[109,99]],
    [[107,83],[108,98]],
    [[107,82],[108,97]],
    [[106,82],[107,97]],
    [[105,81],[106,96]],
    [[104,81],[105,96]],
    [[103,81],[104,96]]    
    ]

grid_12_150 = [
    [[102,82],[103,97]],
    [[101,83],[102,98]],
    [[99,84],[101,99]],
    [[97,86],[99,101]],
    [[94,87],[97,102]]
    ]


Grid_all =grid_32+grid_30+grid_29+grid_28+grid_26+grid_24+grid_22+grid_21+grid_20+grid_19+grid_18+grid_12
Grid_OR =grid_28+grid_26+grid_24+grid_22+grid_21+grid_20+grid_19

Grid_all_150 =grid_32_150+grid_30_150+grid_29_150+grid_28_150+grid_26_150+grid_24_150+grid_22_150+grid_21_150+grid_20_150+grid_19_150+grid_18_150+grid_12_150
Grid_OR_150 =grid_28_150+grid_26_150+grid_24_150+grid_22_150+grid_21_150+grid_20_150+grid_19_150

Tap1 =[]
Tm1 =[]
Tj1 = []
Tju1 =[]
Ta1 =[]

Tap2 =[]
Tm2 =[]
Tj2 = []
Tju2 =[]
Ta2 =[]

Uap = []
Um = []
Uj =[]
Uju =[]
Ua =[]

Vap =[]
Vm = []
Vj = []
Vju =[]
Va =[]

for year in years:
    
    file_name_apr = "CMEMS_Temperature_"+str(year)+"-04.nc"
    file_name_may = "CMEMS_Temperature_"+str(year)+"-05.nc"
    file_name_jun = "CMEMS_Temperature_"+str(year)+"-06.nc"
    file_name_jul = "CMEMS_Temperature_"+str(year)+"-07.nc"
    file_name_aug = "CMEMS_Temperature_"+str(year)+"-08.nc"
    
    file_name_U_apr = "CMEMS_U_velocity_" + str(year) +"-04.nc"
    file_name_U_may = "CMEMS_U_velocity_" + str(year) +"-05.nc"
    file_name_U_jun = "CMEMS_U_velocity_" + str(year) +"-06.nc"
    file_name_U_jul = "CMEMS_U_velocity_" + str(year) +"-07.nc"
    file_name_U_aug = "CMEMS_U_velocity_" + str(year) +"-08.nc"
    
    file_name_V_apr= "CMEMS_V_velocity_" + str(year) +"-05.nc"
    file_name_V_may = "CMEMS_V_velocity_" + str(year) +"-05.nc"
    file_name_V_jun = "CMEMS_V_velocity_" + str(year) +"-06.nc"
    file_name_V_jul = "CMEMS_V_velocity_" + str(year) +"-07.nc"
    file_name_V_aug = "CMEMS_V_velocity_" + str(year) +"-08.nc"

    if os.path.exists(file_name_apr) == True:
    
       Data_may = Dataset(file_name_apr, 'r')
       Temp_may = Data_may.variables['thetao']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Tap1  += [round(np.mean(T_store1_may),4)]
       Tap2  += [round(np.max(T_store1_may),4)]
       
    else:
        Tap1 += [np.nan]
        Tap2 += [np.nan]
    
    if os.path.exists(file_name_may) == True:
    
       Data_may = Dataset(file_name_may, 'r')
       Temp_may = Data_may.variables['thetao']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Tm1  += [round(np.mean(T_store1_may),4)]
       Tm2  += [round(np.max(T_store1_may),4)]
       
    else:
        Tm1 += [np.nan]
        Tm2 += [np.nan]
        
    if os.path.exists(file_name_jun) == True:
    
       Data_jun = Dataset(file_name_jun, 'r')
       Temp_jun = Data_jun.variables['thetao']
       
       T_store1_jun =[]
       
       for ii in Grid_OR_150:
           T_store1_jun+=Temp_jun[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Tj1  += [round(np.mean(T_store1_jun),4)]
       Tj2  += [round(np.max(T_store1_jun),4)]
       
    else:
        Tj1 += [np.nan]
        Tj2 += [np.nan]


    if os.path.exists(file_name_jul) == True:
    
       Data_jul = Dataset(file_name_jul, 'r')
       Temp_jul = Data_jul.variables['thetao']
       
       T_store1_jul =[]
       
       for ii in Grid_OR_150:
           T_store1_jul+=Temp_jul[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Tju1  += [round(np.mean(T_store1_jul),4)]
       Tju2  += [round(np.max(T_store1_jul),4)]
       
    else:
        Tju1 += [np.nan]  
        Tju2 += [np.nan] 
        
        
    if os.path.exists(file_name_aug) == True:
    
       Data_aug = Dataset(file_name_aug, 'r')
       Temp_aug = Data_aug.variables['thetao']
       
       T_store1_aug =[]
       
       for ii in Grid_OR_150:
           T_store1_aug+=Temp_aug[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Ta1  += [round(np.mean(T_store1_aug),4)]
       Ta2  += [round(np.max(T_store1_aug),4)]
       
    else:
        Ta1 += [np.nan]
        Ta2 += [np.nan]
        
        
        
#################################
## U and V
    if os.path.exists(file_name_U_apr) == True:
    
       Data_may = Dataset(file_name_U_apr, 'r')
       Temp_may = Data_may.variables['uo']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Uap  += [round(np.mean(T_store1_may),4)]

    else:
        Uap += [np.nan]


    if os.path.exists(file_name_U_may) == True:
    
       Data_may = Dataset(file_name_U_may, 'r')
       Temp_may = Data_may.variables['uo']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Um  += [round(np.mean(T_store1_may),4)]

    else:
        Um += [np.nan]
        
    if os.path.exists(file_name_U_jun) == True:
    
       Data_may = Dataset(file_name_U_jun, 'r')
       Temp_may = Data_may.variables['uo']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Uj  += [round(np.mean(T_store1_may),4)]

    else:
        Uj += [np.nan]       

    if os.path.exists(file_name_U_jul) == True:
    
       Data_may = Dataset(file_name_U_jul, 'r')
       Temp_may = Data_may.variables['uo']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Uju  += [round(np.mean(T_store1_may),4)]

    else:
        Uju += [np.nan] 

    if os.path.exists(file_name_U_aug) == True:
    
       Data_may = Dataset(file_name_U_aug, 'r')
       Temp_may = Data_may.variables['uo']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Ua  += [round(np.mean(T_store1_may),4)]

    else:
        Ua += [np.nan] 
        
###########################
# V

    if os.path.exists(file_name_V_apr) == True:
    
       Data_may = Dataset(file_name_V_apr, 'r')
       Temp_may = Data_may.variables['vo']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Vap  += [round(np.mean(T_store1_may),4)]

    else:
        Vap += [np.nan]

 
    if os.path.exists(file_name_V_may) == True:
    
       Data_may = Dataset(file_name_V_may, 'r')
       Temp_may = Data_may.variables['vo']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Vm  += [round(np.mean(T_store1_may),4)]

    else:
        Vm += [np.nan]

    if os.path.exists(file_name_V_jun) == True:
    
       Data_may = Dataset(file_name_V_jun, 'r')
       Temp_may = Data_may.variables['vo']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Vj  += [round(np.mean(T_store1_may),4)]

    else:
        Vj += [np.nan]

    if os.path.exists(file_name_V_jul) == True:
    
       Data_may = Dataset(file_name_V_jul, 'r')
       Temp_may = Data_may.variables['vo']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Vju  += [round(np.mean(T_store1_may),4)]

    else:
        Vju += [np.nan]

    if os.path.exists(file_name_V_aug) == True:
    
       Data_may = Dataset(file_name_V_aug, 'r')
       Temp_may = Data_may.variables['vo']
       
       T_store1_may =[]
       
       for ii in Grid_OR_150:
           T_store1_may+=Temp_may[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
           
       Va  += [round(np.mean(T_store1_may),4)]

    else:
        Va += [np.nan]

feml_precond = ["-04-01","-09-31"]

T_bottom_femprecond =[]
for year in years:
    T_store1 =[]

    for month in ["04","05","06","07","08","09"]:
        file_name = "CMEMS_Temperature_Bottom_"+str(year-1)+"-"+month+".nc"
        
        if os.path.exists(file_name) == True:
        
           Data = Dataset(file_name, 'r')
           Temp = Data.variables['bottomT']
           
           for ii in Grid_OR:
               T_store1+=Temp[:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        else:
            T_store1+= [np.nan]
   

    T_bottom_femprecond  += [round(np.mean(T_store1),4)]  

brood        = ["-10-01","-02-28"]
T_bottom_brood =[]
for year in years[:len(years)]:
    T_store1 =[]

    
    file_name_10 = "CMEMS_Temperature_Bottom_"+str(year-1)+"-10.nc"
    file_name_11 = "CMEMS_Temperature_Bottom_"+str(year-1)+"-11.nc"
    file_name_12 = "CMEMS_Temperature_Bottom_"+str(year-1)+"-12.nc"        
    file_name_01 = "CMEMS_Temperature_Bottom_"+str(year)+"-01.nc"
    file_name_02 = "CMEMS_Temperature_Bottom_"+str(year)+"-02.nc"
    for file in [file_name_10, file_name_11, file_name_12, file_name_01, file_name_02]:
        if os.path.exists(file) == True: 
    
           Data = Dataset(file, 'r')
           Temp = Data.variables['bottomT']
       
           for ii in Grid_OR:
               T_store1+=Temp[:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        else:
            T_store1+= [np.nan]
   

    T_bottom_brood  += [round(np.nanmax(T_store1),4)] 

########################################################################
######## Spawning
    
Spwn         = ["-02-01", "-03-31"]   
T_bottom_spawn =[]
for year in years[:len(years)]:
    T_store1 =[]

    file_name_02 = "CMEMS_Temperature_Bottom_"+str(year)+"-02.nc"
    file_name_03 = "CMEMS_Temperature_Bottom_"+str(year)+"-03.nc"
    # file_name_04 = "CMEMS_Temperature_Bottom_"+str(year)+"-04.nc"
    # file_name_05 = "CMEMS_Temperature_Bottom_"+str(year)+"-05.nc"        
    # file_name_06 = "CMEMS_Temperature_Bottom_"+str(year)+"-06.nc"

    for file in [file_name_02, file_name_03]:
        if os.path.exists(file) == True: 
    
           Data = Dataset(file, 'r')
           Temp = Data.variables['bottomT']
       
           for ii in Grid_OR:
               T_store1+=Temp[:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        else:
            T_store1+= [np.nan]
   

    T_bottom_spawn += [round(np.max(T_store1),4)] 
    
########################################################################
######## over the whole larval phase
    
lrvl_phase         = ["-04-01", "-08-31"]   

T_top_lrvl_phase =[]
T_top_lrvl_phase2 =[]
for year in years:
    T_store1 =[]

    for month in ["04","05","06","07","08"]:
        file_name = "CMEMS_Temperature_"+str(year)+"-"+month+".nc"
        
        
        if os.path.exists(file_name) == True:
        
           Data = Dataset(file_name, 'r')
           Temp = Data.variables['thetao']
           
           for ii in Grid_OR_150:
               T_store1+=Temp[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        else:
            T_store1+= [np.nan]
   

    T_top_lrvl_phase  += [round(np.nanmean(T_store1),4)]  
    T_top_lrvl_phase2  += [round(np.nanmax(T_store1),4)]  


U_top_lrvl_phase =[]

for year in years:
    T_store1 =[]

    for month in ["04","05","06","07","08"]:
        file_name = "CMEMS_U_velocity_"+str(year)+"-"+month+".nc"
        
        
        if os.path.exists(file_name) == True:
        
           Data = Dataset(file_name, 'r')
           Temp = Data.variables['uo']
           
           for ii in Grid_OR_150:
               T_store1+=Temp[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        else:
            T_store1+= [np.nan]
   

    U_top_lrvl_phase  += [round(np.nanmean(T_store1),4)]  
    

V_top_lrvl_phase =[]

for year in years:
    T_store1 =[]

    for month in ["04","05","06","07","08"]:
        file_name = "CMEMS_V_velocity_"+str(year)+"-"+month+".nc"
        
        
        if os.path.exists(file_name) == True:
        
           Data = Dataset(file_name, 'r')
           Temp = Data.variables['vo']
           
           for ii in Grid_OR_150:
               T_store1+=Temp[:,:,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]].data.flatten().tolist()
        else:
            T_store1+= [np.nan]
   

    V_top_lrvl_phase  += [round(np.nanmean(T_store1),4)] 


######################################      
X_shrimp_glorys_150 = pd.DataFrame({'year':list(range(1993,2020)),
                                                'meanTLT_apr': Tap1,
                                                'meanTLT_may': Tm1,
                                                'meanTLT_jun':Tj1,
                                                'meanTLT_jul':Tju1,
                                                'meanTLT_aug':Ta1,
                                                'maxTLT_apr': Tap2,
                                                'maxTLT_may': Tm2,
                                                'maxTLT_jun':Tj2,
                                                'maxTLT_jul':Tju2,
                                                'maxTLT_aug':Ta2,
                                                'meanCST_apr': Uap,
                                                'meanCST_may': Um,
                                                'meanCST_jun': Uj,
                                                'meanCST_jul': Uju,
                                                'meanCST_aug': Ua,
                                                'meanLST_apr': Vap,
                                                'meanLST_may': Vm,
                                                'meanLST_jun': Vj,
                                                'meanLST_jul': Vju,
                                                'meanLST_aug': Va,
                                                'meanTLT_lrvl_phase': T_top_lrvl_phase,
                                                'maxTLT_lrvl_phase': T_top_lrvl_phase2,
                                                'meanCST_lrvl_phase' : U_top_lrvl_phase,
                                                'meanLST_lrvl_phase' : U_top_lrvl_phase,
                                                'meanBLT_precond':T_bottom_femprecond,
                                                'maxBLT_brood': T_bottom_brood,
                                                'maxBLT_spwn': T_bottom_spawn})

X_shrimp_glorys_150.to_csv('/Users/rbani20/Dropbox/X_shrimp_glorys_150.csv',)