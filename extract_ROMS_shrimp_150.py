#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 19:49:44 2021

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

from BLT_LS_func import TLT_func

# def main():
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
area_number = [32,30,29,28,26,24,22,21,20,19,18,12]
#os.chdir('/Users/Documents/PostdocUW/data_roms')
with open('datasets.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    datasets = list (csv_reader)
#os.chdir('/Users/Documents/PostdocUW/codes')
# pool = mp.Pool(12)
# [pool.apply(BLT_extract, args=(datasets, Grid, area, area_number)) for area in range(len(Grid))]
# pool.close()
l1 = [0,706]
l2 = [706,1412]
l3 = [1412,2118]
l4 = [2118, 2827]
l = l1

TLT_func(l, datasets, Grid, area_number)

# if __name__ == '__main__':
#     # freeze_support() here if program needs to be frozen
#     main()  # execute this only when run directly, not when imported!
