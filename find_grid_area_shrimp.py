#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:33:39 2021

@author: N
"""
import os
import sys
import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from pydap.client import open_url
import xarray as xr


## Areas along US West coast
os.chdir('/Users/Documents/PostdocUW/shrimp_data')
Ar = pd.read_csv('area_locations.csv')
Ar['lon'] = [-125.5, -125.5, -125.5, -125.5, -125.5, -125.5, -125.5, -125.5, -125.5, -125.5, -125.5, -125.5]       


# Bathemetry
os.chdir('/Users/Documents/PostdocUW/gebco_bath')
ds = xr.open_dataset('gebco.nc')
df = ds.to_dataframe()
df.reset_index(inplace=True)

os.chdir('/Users/Documents/PostdocUW/data')
# Read the -91 m isocliane

for i in range(len(Ar)):
    if i ==0:
        A = df[(df['elevation'] == -91) & (df['lat'] <= Ar['Lat1'][i]) & (df['lat'] >= Ar['Lat2'][i]) & (df['lon'] < -124)]
        B = df[(df['elevation'] == -183) & (df['lat'] <= Ar['Lat1'][i]) & (df['lat'] >= Ar['Lat2'][i]) & (df['lon'] < -124)]
    else:
        A = df[(df['elevation'] == -91) & (df['lat'] <= Ar['Lat1'][i]) & (df['lat'] >= Ar['Lat2'][i])]
        B = df[(df['elevation'] == -183) & (df['lat'] <= Ar['Lat1'][i]) & (df['lat'] >= Ar['Lat2'][i])]
            
    A.to_csv("iso_91_"+str(Ar['Area'][i])+".csv", )
    B.to_csv("iso_183_"+str(Ar['Area'][i])+".csv", )            
    
    
    
df91 = df[(df['elevation'] == -91)]

# Read the -183 m isocliane
df183 = df[df['elevation'] == -183]

df91.reset_index(inplace=True)
df183.reset_index(inplace=True)

# Shoreline
os.chdir('/Users/Documents/Data_for_P1/maps')

sf1= shp.Reader("us_medium_shoreline")

os.chdir('/Users/Documents/PostdocUW')

def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

# reshape the shape data into dataframe

df1 = read_shapefile(sf1)
df_pac_shore = df1[df1.REGIONS == 'P']
df_ps =df_pac_shore.reset_index()


WA_coords = [[48.5,-123.5],[46.3,-123.5]]
OR_coords = [[46.3,-123.5],[42,-123.5]]
NCA_coords = [[42,-123.6],[38.8,-123.6]]
SCA_coords = [[38.8,-117.1],[32.5,-117.1]]

WA_ps = pd.DataFrame({'coords' : []})
OR_ps = pd.DataFrame({'coords' : []})
NCA_ps =pd.DataFrame({'coords' : []})  

for i in range(len(df_ps)):
        l1=[]; l2=[]; l3=[]; 
        B = df_ps.coords[i]
        for j in range(len(df_ps.coords[i])):
            if (B[j][1] <= WA_coords[0][0]) & (B[j][1] >= WA_coords[1][0]) & (B[j][0] < -123.5):
                l1.append(j)
            if (B[j][1] <= OR_coords[0][0]) & (B[j][1] >= OR_coords[1][0]) & (B[j][0] < -123.5):
                l2.append(j)                
            if (B[j][1] <= NCA_coords[0][0]) & (B[j][1] >= NCA_coords[1][0]) & (B[j][0] < -123.5):
                l3.append(j)               

                    
        if (len(l1) <= len(B)) & (len(l1)>0):
            WA_ps.coords[i] = [i for j, i in enumerate(B) if j in l1]       
        if (len(l2) <= len(B)) & (len(l2)>0):
            OR_ps.coords[i] = [i for j, i in enumerate(B) if j in l2] 
        if (len(l3) <= len(B)) & (len(l3)>0):
            NCA_ps.coords[i] = [i for j, i in enumerate(B) if j in l3] 

WC_US_ps=[WA_ps,OR_ps,NCA_ps]



# ROMS spatial Grid
opendap_url = 'https://oceanmodeling.ucsc.edu:8443/thredds/dodsC/wc12.0_ccsra31_01/prior/wc12_ccsra31_fwd_000_20101221_20101229.nc'
dataset = open_url(opendap_url)

lat = dataset['lat_rho']
lon = dataset['lon_rho']
Alat = lon[:,:].data[0,:]
Alon = lat[:,:].data[:,0] 


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

Grid =[grid_32,grid_30,grid_29,grid_28,grid_26,grid_24,grid_22,grid_21,grid_20,grid_19,grid_18,grid_12]
A=[]
B=[]


os.chdir('/Users/Documents/PostdocUW/data')
fig1 = plt.figure()

for i in range(len(Ar)):
    A = pd.read_csv("iso_91_"+str(Ar['Area'][i])+".csv")
    B = pd.read_csv("iso_183_"+str(Ar['Area'][i])+".csv")
    plt.plot(A['lon'],A['lat'],'b')
    plt.plot(B['lon'],B['lat'],'r')
for t in Grid:        
    for w in t:
        for i in range(w[0][0],w[1][0]):
            for j in range(w[0][1],w[1][1]):
                y = [Alon[i],Alon[i+1],Alon[i+1],Alon[i],Alon[i]]
                x = [Alat[j],Alat[j],Alat[j+1],Alat[j+1],Alat[j]]
                plt.plot(x, y, 'k',linewidth=0.5)
                
for r in WC_US_ps:
    for c in r.coords:
        x = [i[0] for i in c]
        y = [i[1] for i in c]
        plt.plot(x,y,'k',linewidth=1)   
        
for i in range(len(Ar)):
    y1 = [Ar['lon'][i],Ar['lon'][i]+0.5]
    x1 = [Ar['Lat1'][i],Ar['Lat1'][i]]
    plt.plot(y1,x1,'k',linewidth=1.5)  
    plt.text(-125.5, Ar['Centroid'][i], Ar['Area'][i], fontsize=8)
      
             
os.chdir('/Users/Documents/PostdocUW/figure')
fig1.set_size_inches(6, 8)
fig1.savefig('Shrimp_91_183_grid.pdf', format='pdf', dpi=2000)

#plt.ylim(Ar['Lat2'][0],Ar['Lat1'][0])