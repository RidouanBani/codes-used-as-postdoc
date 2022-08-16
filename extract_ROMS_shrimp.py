#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:31:09 2021

@author: N
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 23:00:36 2021

@author: N
"""
import os
from pydap.client import open_url
import numpy as np
import multiprocessing as mp
import pandas as pd
import csv
import sys
sys.path.append('/home/rbani20/projects/def-guichard/rbani20/postdoc')

from BLT_LS_func import BLT_extract, temp_per_area, Temp_per_grid

# def main():
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
area_number = [32,30,29,28,26,24,22,21,20,19,18,12]
#os.chdir('/Users/Documents/PostdocUW/data_roms')
with open('datasets.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    datasets = list (csv_reader)
#os.chdir('/Users/Documents/PostdocUW/codes')
# pool = mp.Pool(12)
# [pool.apply(BLT_extract, args=(datasets, Grid, area, area_number)) for area in range(len(Grid))]
# pool.close()
l1 = [2118, 2294] #[0,706]
l2 = [2294, 2470]#[706,1412]
l3 = [2470, 2646]#[1412,2118]
l4 = [2646, 2827]#[2118, 2827]
l = l1

BLT_extract(l, datasets, Grid, area_number)

# if __name__ == '__main__':
#     # freeze_support() here if program needs to be frozen
#     main()  # execute this only when run directly, not when imported!

