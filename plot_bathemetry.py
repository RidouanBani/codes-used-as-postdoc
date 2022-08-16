#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 22:15:56 2021

@author: N
"""
import sys
import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from pydap.client import open_url


opendap_url = 'https://oceanmodeling.ucsc.edu:8443/thredds/dodsC/wc12.0_ccsra31_01/prior/wc12_ccsra31_fwd_000_20101221_20101229.nc'

dataset = open_url(opendap_url)

cd '/Users/Documents/Data_for_P1/maps'

# read US west coast bathemetry
sf = shp.Reader("caorwall")
# read the US coastal line
sf1= shp.Reader("us_medium_shoreline")

cd '/Users/Documents/PostdocUW'

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
df = read_shapefile(sf)
df1 = read_shapefile(sf1)

# Read the -200 m isocliane
df_200 = df[df.DEPTH == -500]
df_200 = df_200.reset_index()


# read the US west coast shoreline
df_pac_shore = df1[df1.REGIONS == 'P']
df_ps =df_pac_shore.reset_index()

lat = dataset['lat_rho']
lon = dataset['lon_rho']
Alat = lon[:,:].data[0,:]
Alon = lat[:,:].data[:,0] 

WA_coords = [[48.5,-123.5],[46.3,-123.5]]
OR_coords = [[46.3,-123.5],[42,-123.5]]
NCA_coords = [[42,-123.6],[38.8,-123.6]]
SCA_coords = [[38.8,-117.1],[32.5,-117.1]]

Boundary =[WA_coords, OR_coords, NCA_coords, SCA_coords]
def find_coor_peth(df_200, df_ps, WA_coords, OR_coords, NCA_coords, SCA_coords):
    WA_200 = pd.DataFrame({'coords' : []})
    WA_ps = pd.DataFrame({'coords' : []})
    OR_200 = pd.DataFrame({'coords' : []})
    OR_ps = pd.DataFrame({'coords' : []})
    NCA_200 = pd.DataFrame({'coords' : []})
    NCA_ps =pd.DataFrame({'coords' : []})
    SCA_200 = pd.DataFrame({'coords' : []})    
    SCA_ps =  pd.DataFrame({'coords' : []})
    
    for i in range(len(df_200)):
        l1=[]; l2=[]; l3=[]; l4=[];
        A = df_200.coords[i]
        for j in range(len(A)):
            
            if (A[j][1] <= WA_coords[0][0]) & (A[j][1] >= WA_coords[1][0]) & (A[j][0] > -127):
                l1.append(j)
            if (A[j][1] <= OR_coords[0][0]) & (A[j][1] >= OR_coords[1][0]) & (A[j][0] > -127):
                l2.append(j)               
            if (A[j][1] <= NCA_coords[0][0]) & (A[j][1] >= NCA_coords[1][0]) & (A[j][0] > -127):
                l3.append(j)             
            if (A[j][1] <= SCA_coords[0][0]) & (A[j][1] >= SCA_coords[1][0])& (A[j][0] > -127):
                l4.append(j)           
        if (len(l1) <= len(A)) & (len(l1)>0):
            WA_200.coords[i] = [i for j, i in enumerate(A) if j in l1]       
        if (len(l2) <= len(A)) & (len(l2)>0):
            OR_200.coords[i] = [i for j, i in enumerate(A) if j in l2] 
        if (len(l3) <= len(A)) & (len(l3)>0):
            NCA_200.coords[i] = [i for j, i in enumerate(A) if j in l3] 
        if (len(l4) <= len(A)) & (len(l4)>0):
            SCA_200.coords[i] = [i for j, i in enumerate(A) if j in l4] 
        
                                
    for i in range(len(df_ps)):
        l1=[]; l2=[]; l3=[]; l4=[];
        B = df_ps.coords[i]
        for j in range(len(df_ps.coords[i])):
            if (B[j][1] <= WA_coords[0][0]) & (B[j][1] >= WA_coords[1][0]) & (B[j][0] < -123.5):
                l1.append(j)
            if (B[j][1] <= OR_coords[0][0]) & (B[j][1] >= OR_coords[1][0]) & (B[j][0] < -123.5):
                l2.append(j)                
            if (B[j][1] <= NCA_coords[0][0]) & (B[j][1] >= NCA_coords[1][0]) & (B[j][0] < -123.5):
                l3.append(j)               
            if (B[j][1] <= SCA_coords[0][0]) & (B[j][1] >= SCA_coords[1][0]):
                l4.append(j)
                    
        if (len(l1) <= len(B)) & (len(l1)>0):
            WA_ps.coords[i] = [i for j, i in enumerate(B) if j in l1]       
        if (len(l2) <= len(B)) & (len(l2)>0):
            OR_ps.coords[i] = [i for j, i in enumerate(B) if j in l2] 
        if (len(l3) <= len(B)) & (len(l3)>0):
            NCA_ps.coords[i] = [i for j, i in enumerate(B) if j in l3] 
        if (len(l4) <= len(B)) & (len(l4)>0):
            SCA_ps.coords[i] = [i for j, i in enumerate(B) if j in l4] 
    
    WA_200= WA_200.coords.to_frame()
    WA_ps = WA_ps.coords.to_frame()
    OR_200= OR_200.coords.to_frame()
    OR_ps = OR_ps.coords.to_frame()
    NCA_200= NCA_200.coords.to_frame()
    NCA_ps = NCA_ps.coords.to_frame()
    SCA_200= SCA_200.coords.to_frame()
    SCA_ps = SCA_ps.coords.to_frame()
    
    WA_200 = WA_200.reset_index()
    WA_ps = WA_ps.reset_index()
    OR_200 = OR_200.reset_index()
    OR_ps = OR_ps.reset_index()
    NCA_200 = NCA_200.reset_index()
    NCA_ps = NCA_ps.reset_index()
    SCA_200 = SCA_200.reset_index()
    SCA_ps = SCA_ps.reset_index()            
    return WA_200, WA_ps, OR_200, OR_ps, NCA_200, NCA_ps, SCA_200, SCA_ps

WA_200, WA_ps, OR_200, OR_ps, NCA_200, NCA_ps, SCA_200, SCA_ps = find_coor_peth(df_200, df_ps, WA_coords, OR_coords, NCA_coords, SCA_coords)

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
WA_grid_LS=[[[163,98],[169,100]],
         [[169,97],[173,99]],
         [[173,95],[178,97]],
         [[178,93],[180,95]]]
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
SCA_grid_NISL=[[[25,165],[32,169]],
         [[32,163],[34,166]],
         [[34,157],[37,163]],
         [[37,153],[39,156]],
         [[39,134],[41,156]],
         [[41,133],[44,148]],
         [[44,132],[45,145]],
         [[45,131],[46,135]],
         [[46,129],[52,134]],   
         [[52,127],[55,131]],
         [[55,126],[56,130]],
         [[56,124],[58,128]],
         [[58,124],[59,126]],
         [[59,123],[61,125]],
         [[61,122],[62,124]],         
         [[62,119],[63,123]],                
         [[63,119],[66,121]],         
         [[66,119],[67,103]],         
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

WC_US_200=[]
WC_US_200.append(WA_200);WC_US_200.append(OR_200);WC_US_200.append(NCA_200);WC_US_200.append(SCA_200)

WC_US_ps=[]
WC_US_ps.append(WA_ps);WC_US_ps.append(OR_ps);WC_US_ps.append(NCA_ps);WC_US_ps.append(SCA_ps);

WC_US_grid=[]
WC_US_grid.append(WA_grid);WC_US_grid.append(OR_grid);WC_US_grid.append(NCA_grid);WC_US_grid.append(SCA_grid);

WC_US_grid_LS=[]
WC_US_grid_LS.append(WA_grid_LS);WC_US_grid_LS.append(OR_grid_LS);WC_US_grid_LS.append(NCA_grid_LS);WC_US_grid_LS.append(SCA_grid_LS);

WC_US_grid_150 = []
WC_US_grid_150.append(WA_grid_150);WC_US_grid_150.append(OR_grid_150);WC_US_grid_150.append(NCA_grid_150);WC_US_grid_150.append(SCA_grid_150);

WC_US_grid_250 = []
WC_US_grid_250.append(WA_grid_250);WC_US_grid_250.append(OR_grid_250);WC_US_grid_250.append(NCA_grid_250);WC_US_grid_250.append(SCA_grid_250);


def plot_grid(WC_US_200, WC_US_ps, WC_US_grid_LS, WC_US_grid, WC_US_grid_150, WC_US_grid_250):
    fig = plt.figure()
    plt.subplot(2, 2,1)
    for r in WC_US_200:
        for c in r.coords:
            x = [i[0] for i in c]
            y = [i[1] for i in c]
            plt.plot(x,y,'b',linewidth=1)
    for r in WC_US_ps:
        for c in r.coords:
            x = [i[0] for i in c]
            y = [i[1] for i in c]
            plt.plot(x,y,'k',linewidth=1)
    for r in WC_US_grid_LS:
        for w in r:
            for i in range(w[0][0],w[1][0]):
                for j in range(w[0][1],w[1][1]):
                    y = [Alon[i],Alon[i+1],Alon[i+1],Alon[i],Alon[i]]
                    x = [Alat[j],Alat[j],Alat[j+1],Alat[j+1],Alat[j]]
                    plt.plot(x, y, 'k',linewidth=0.2)
    
    plt.subplot(2, 2,2)
    for r in WC_US_200:
        for c in r.coords:
            x = [i[0] for i in c]
            y = [i[1] for i in c]
            plt.plot(x,y,'b',linewidth=1)
    for r in WC_US_ps:
        for c in r.coords:
            x = [i[0] for i in c]
            y = [i[1] for i in c]
            plt.plot(x,y,'k',linewidth=1)
    for r in WC_US_grid:
        for w in r:
            for i in range(w[0][0],w[1][0]):
                for j in range(w[0][1],w[1][1]):
                    y = [Alon[i],Alon[i+1],Alon[i+1],Alon[i],Alon[i]]
                    x = [Alat[j],Alat[j],Alat[j+1],Alat[j+1],Alat[j]]
                    plt.plot(x, y, 'k',linewidth=0.2)
                    
    plt.subplot(2, 2,3)
    for r in WC_US_200:
        for c in r.coords:
            x = [i[0] for i in c]
            y = [i[1] for i in c]
            plt.plot(x,y,'b',linewidth=1)
    for r in WC_US_ps:
        for c in r.coords:
            x = [i[0] for i in c]
            y = [i[1] for i in c]
            plt.plot(x,y,'k',linewidth=1)
    for r in WC_US_grid_150:
        for w in r:
            for i in range(w[0][0],w[1][0]):
                for j in range(w[0][1],w[1][1]):
                    y = [Alon[i],Alon[i+1],Alon[i+1],Alon[i],Alon[i]]
                    x = [Alat[j],Alat[j],Alat[j+1],Alat[j+1],Alat[j]]
                    plt.plot(x, y, 'k',linewidth=0.2)
    plt.subplot(2, 2,4)
    for r in WC_US_200:
        for c in r.coords:
            x = [i[0] for i in c]
            y = [i[1] for i in c]
            plt.plot(x,y,'b',linewidth=1)
    for r in WC_US_ps:
        for c in r.coords:
            x = [i[0] for i in c]
            y = [i[1] for i in c]
            plt.plot(x,y,'k',linewidth=1)
    for r in WC_US_grid_250:
        for w in r:
            for i in range(w[0][0],w[1][0]):
                for j in range(w[0][1],w[1][1]):
                    y = [Alon[i],Alon[i+1],Alon[i+1],Alon[i],Alon[i]]
                    x = [Alat[j],Alat[j],Alat[j+1],Alat[j+1],Alat[j]]
                    plt.plot(x, y, 'k',linewidth=0.2)
    plt.show()
    return fig

fig1 = plot_grid(WC_US_200, WC_US_ps, WC_US_grid_LS, WC_US_grid, WC_US_grid_150, WC_US_grid_250)
fig1.savefig('Grid_range.pdf', format='pdf', dpi=2000)
for c in SCA_200.coords:
    x = [i[0] for i in c]
    y = [i[1] for i in c]
    plt.plot(x,y,'b',linewidth=1)
for c in WA_ps.coords:
    x = [i[0] for i in c]
    y = [i[1] for i in c]
    plt.plot(x,y,'k',linewidth=1)

for w in WA_grid:
    for i in range(w[0][0],w[1][0]):
        for j in range(w[0][1],w[1][1]):
            y = [Alon[i],Alon[i+1],Alon[i+1],Alon[i],Alon[i]]
            x = [Alat[j],Alat[j],Alat[j+1],Alat[j+1],Alat[j]]
            plt.plot(x, y, 'k',linewidth=0.7)
            
plt.xlim(-122.,-117)
plt.ylim(32,35.5)
plt.show()

##########################################################################
##########################################################################
# retrieve the data
# first i define function to go throu regions 


from pydap.client import open_url
from pydap.cas.urs import setup_session
import numpy as np
from siphon.catalog import TDSCatalog
from siphon import catalog, ncss
import csv 

data_url = ('https://oceanmodeling.ucsc.edu:8443/thredds/catalog/wc12.0_ccsra31_01/posterior/catalog.xml')

cat = TDSCatalog(data_url)
datasets = list(cat.datasets)

WA_BLT_LS=[];  OR_BLT_LS=[]; NCA_BLT_LS=[]; SCA_BLT_LS=[];
WA_TLT_CS=[];  OR_TLT_CS=[]; NCA_TLT_CS=[]; SCA_TLT_CS=[];
WA_TLT_150=[];  OR_TLT_150=[]; NCA_TLT_150=[]; SCA_TLT_150=[];
WA_TLT_250=[];  OR_TLT_250=[]; NCA_TLT_250=[]; SCA_TLT_250=[];
WA_LS_CS=[];  OR_LS_CS=[]; NCA_LS_CS=[]; SCA_LS_CS=[];
WA_LS_150=[];  OR_LS_150=[]; NCA_LS_150=[]; SCA_LS_150=[];
WA_LS_250=[];  OR_LS_250=[]; NCA_LS_250=[]; SCA_LS_250=[];
WA_CS_CS=[];  OR_CS_CS=[]; NCA_CS_CS=[]; SCA_CS_CS=[];
WA_CS_150=[];  OR_CS_150=[]; NCA_CS_150=[]; SCA_CS_150=[];
WA_CS_250=[];  OR_CS_250=[]; NCA_CS_250=[]; SCA_CS_250=[];
for u in datasets:
    opendap_url = "https://oceanmodeling.ucsc.edu:8443/thredds/dodsC/wc12.0_ccsra31_01/posterior/"+str(u)
    dataset = open_url(opendap_url)

    Temp = dataset['temp']
    U_momntum = dataset['u']
    V_momntum = dataset['v']
    
    # WA bottom Temperature  in shallow watters 0-10km off-shore 
    T_store =[]
    for ii in WA_grid_LS:
        temp = Temp[0:16,0,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan
    WA_BLT_LS.append(np.nanmean(T_store))
    
    # OR bottom Temperature  in shallow watters 0-10km off-shore 
    T_store =[]
    for ii in OR_grid_LS:
        temp = Temp[0:16,1,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan
    OR_BLT_LS.append(np.nanmean(T_store))
    
    # NCA bottom Temperature in shallow watters 0-10km off-shore 
    T_store =[]
    for ii in NCA_grid_LS:
        temp = Temp[0:16,1,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan
    NCA_BLT_LS.append(np.nanmean(T_store))
    
    # SCA   bottom Temperature in shallow watters 0-10km off-shore 
    T_store =[]
    for ii in SCA_grid_LS:
        temp = Temp[0:16,1,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan
    SCA_BLT_LS.append(np.nanmean(T_store))
       
    # WA top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in WA_grid:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    WA_TLT_CS.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    WA_LS_CS.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    WA_CS_CS.append(np.nanmean(CS_store))
    
    # OR top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in OR_grid:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    OR_TLT_CS.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    OR_LS_CS.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    OR_CS_CS.append(np.nanmean(CS_store))
    
    # NCA top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in NCA_grid:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    NCA_TLT_CS.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    NCA_LS_CS.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    NCA_CS_CS.append(np.nanmean(CS_store))
    
    # SCA top Temperature, cross-shelf transp & along shelf transp  on continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in SCA_grid:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    SCA_TLT_CS.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    SCA_LS_CS.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    SCA_CS_CS.append(np.nanmean(CS_store))       
            
    # WA top Temperature, cross-shelf transp & along shelf transp  on 150km of continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in WA_grid_150:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    WA_TLT_150.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    WA_LS_150.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    WA_CS_150.append(np.nanmean(CS_store))       
            
    # OR top Temperature, cross-shelf transp & along shelf transp  on 150km of continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in OR_grid_150:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    OR_TLT_150.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    OR_LS_150.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    OR_CS_150.append(np.nanmean(CS_store)) 
    
    # NCA top Temperature, cross-shelf transp & along shelf transp  on 150km of continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in NCA_grid_150:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    NCA_TLT_150.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    NCA_LS_150.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    NCA_CS_150.append(np.nanmean(CS_store)) 
    
    # SCA top Temperature, cross-shelf transp & along shelf transp  on 150km of continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in SCA_grid_150:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    SCA_TLT_150.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    SCA_LS_150.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    SCA_CS_150.append(np.nanmean(CS_store)) 
    
    # WA top Temperature, cross-shelf transp & along shelf transp  on 250km of continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in WA_grid_250:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    WA_TLT_250.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    WA_LS_250.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    WA_CS_250.append(np.nanmean(CS_store))  
    
    # OR top Temperature, cross-shelf transp & along shelf transp  on 250km of continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in OR_grid_250:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    OR_TLT_250.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    OR_LS_250.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    OR_CS_250.append(np.nanmean(CS_store))   
    
    # NCA top Temperature, cross-shelf transp & along shelf transp  on 250km of continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in NCA_grid_250:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    NCA_TLT_250.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    NCA_LS_250.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    NCA_CS_250.append(np.nanmean(CS_store))  
    
    # SCA top Temperature, cross-shelf transp & along shelf transp  on 250km of continental shielf waters
    T_store =[]
    CS_store =[]
    LS_store =[]
    for ii in SCA_grid_250:
        temp = Temp[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        T_store+=temp.data.flatten().tolist()
        uemp = U_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        LS_store+=uemp.data.flatten().tolist()
        vemp = V_momntum[0:16,41,ii[0][0]:ii[1][0],ii[0][1]:ii[1][1]]
        CS_store+=vemp.data.flatten().tolist()
    T_store = np.array(T_store)
    T_store = T_store.astype('float')
    T_store[T_store>50] = np.nan  
    SCA_TLT_250.append(np.nanmean(T_store))
    LS_store = np.array(LS_store)
    LS_store = LS_store.astype('float')
    LS_store[LS_store>50] = np.nan  
    SCA_LS_250.append(np.nanmean(LS_store))
    CS_store = np.array(CS_store)
    CS_store = CS_store.astype('float')
    CS_store[CS_store>50] = np.nan  
    SCA_CS_250.append(np.nanmean(CS_store))   
    
cd '/Users/Documents/PostdocUW/data_roms'

np.savetxt("WA_BLT_LS.csv",WA_BLT_LS,delimiter=",");np.savetxt("OR_BLT_LS.csv",OR_BLT_LS,delimiter=",");np.savetxt("NCA_BLT_LS.csv",NCA_BLT_LS,delimiter=",");np.savetxt("SCA_BLT_LS.csv",SCA_BLT_LS,delimiter=",");
np.savetxt("WA_TLT_CS.csv",WA_TLT_CS,delimiter=",");np.savetxt("OR_TLT_CS.csv",OR_TLT_CS,delimiter=",");np.savetxt("NCA_TLT_CS.csv",NCA_TLT_CS,delimiter=",");np.savetxt("SCA_TLT_CS.csv",SCA_TLT_CS,delimiter=",");
np.savetxt("WA_TLT_150.csv",WA_TLT_150,delimiter=",");np.savetxt("OR_TLT_150.csv",OR_TLT_150,delimiter=",");np.savetxt("NCA_TLT_150.csv",NCA_TLT_150,delimiter=",");np.savetxt("SCA_TLT_150.csv",SCA_TLT_CS,delimiter=",");
np.savetxt("WA_TLT_250.csv",WA_TLT_250,delimiter=",");np.savetxt("OR_TLT_150.csv",OR_TLT_250,delimiter=",");np.savetxt("NCA_TLT_250.csv",NCA_TLT_250,delimiter=",");np.savetxt("SCA_TLT_250.csv",SCA_TLT_CS,delimiter=",");
np.savetxt("WA_LS_CS.csv",WA_LS_CS,delimiter=",");np.savetxt("OR_LS_CS.csv",OR_LS_CS,delimiter=",");np.savetxt("NCA_LS_CS.csv",NCA_LS_CS,delimiter=",");np.savetxt("SCA_LS_CS.csv",SCA_LS_CS,delimiter=",");
np.savetxt("WA_LS_150.csv",WA_LS_150,delimiter=",");np.savetxt("OR_LS_150.csv",OR_LS_150,delimiter=",");np.savetxt("NCA_LS_150.csv",NCA_LS_150,delimiter=",");np.savetxt("SCA_LS_150.csv",SCA_LS_150,delimiter=",");
np.savetxt("WA_LS_250.csv",WA_LS_250,delimiter=",");np.savetxt("OR_LS_250.csv",OR_LS_250,delimiter=",");np.savetxt("NCA_LS_250.csv",NCA_LS_250,delimiter=",");np.savetxt("SCA_LS_250.csv",SCA_LS_250,delimiter=",");
np.savetxt("WA_CS_CS.csv",WA_CS_CS,delimiter=",");np.savetxt("OR_CS_CS.csv",OR_CS_CS,delimiter=",");np.savetxt("NCA_CS_CS.csv",NCA_CS_CS,delimiter=",");np.savetxt("SCA_CS_CS.csv",SCA_CS_CS,delimiter=",");
np.savetxt("WA_CS_150.csv",WA_CS_150,delimiter=",");np.savetxt("OR_CS_150.csv",OR_CS_150,delimiter=",");np.savetxt("NCA_CS_150.csv",NCA_CS_150,delimiter=",");np.savetxt("SCA_CS_150.csv",SCA_CS_150,delimiter=",");
np.savetxt("WA_CS_250.csv",WA_CS_250,delimiter=",");np.savetxt("OR_CS_250.csv",OR_CS_250,delimiter=",");np.savetxt("NCA_CS_250.csv",NCA_CS_250,delimiter=",");np.savetxt("SCA_CS_250.csv",SCA_CS_250,delimiter=",");
cd '/Users/Documents/PostdocUW'
