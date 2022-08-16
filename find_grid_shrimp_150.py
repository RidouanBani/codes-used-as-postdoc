#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 18:27:36 2021

@author: N
"""

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

Grid =[grid_32_150,grid_30_150,grid_29_150,grid_28_150,grid_26_150,grid_24_150,grid_22_150,grid_21_150,grid_20_150,grid_19_150,grid_18_150,grid_12_150]
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
        
# for i in range(len(Ar)):
#     y1 = [Ar['lon'][i],Ar['lon'][i]+0.5]
#     x1 = [Ar['Lat1'][i],Ar['Lat1'][i]]
#     plt.plot(y1,x1,'k',linewidth=1.5)  
#     plt.text(-125.5, Ar['Centroid'][i], Ar['Area'][i], fontsize=8)
      
             
os.chdir('/Users/Documents/PostdocUW/figure')
fig1.set_size_inches(6, 8)
fig1.savefig('Shrimp_150_grid.pdf', format='pdf', dpi=2000)

#plt.ylim(Ar['Lat2'][0],Ar['Lat1'][0])