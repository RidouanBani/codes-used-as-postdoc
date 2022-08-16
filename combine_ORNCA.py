#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:40:50 2021

@author: N
"""

import numpy as np
import pandas as pd
from pandas import read_csv

X_OR= read_csv("/Users/Documents/PostdocUW/proc_data/X_OR.csv")
X_NCA= read_csv("/Users/Documents/PostdocUW/proc_data/X_NCA.csv")

X_NCA['meanLST_z1or'] = X_OR['meanLST_z1']
X_NCA['meanLST_z2or'] = X_OR['meanLST_z2']
X_NCA['meanLST_z3or'] = X_OR['meanLST_z3']
X_NCA['meanLST_z4or'] = X_OR['meanLST_z4']
X_NCA['meanLST_z5or'] = X_OR['meanLST_z5']
X_NCA['meanLST_mgor'] = X_OR['meanLST_mg']

X_NCA['meanCST_z1or'] = X_OR['meanCST_z1']
X_NCA['meanCST_z2or'] = X_OR['meanCST_z2']
X_NCA['meanCST_z3or'] = X_OR['meanCST_z3']
X_NCA['meanCST_z4or'] = X_OR['meanCST_z4']
X_NCA['meanCST_z5or'] = X_OR['meanCST_z5']
X_NCA['meanCST_mgor'] = X_OR['meanCST_mg']

X_WA = X_NCA
list1 = list(X_WA)
list1.remove("Y")
list1.remove("date")
list1.remove("Y_SD")

Xt1 = X_WA[list1]
Xt2 = X_WA[["Y","Y_SD"]]

Xt1 = Xt1[1:31].reset_index()
Xt2 = Xt2[5:35].reset_index()

Xt1 = Xt1.drop(columns = ['index'])
Xt2 = Xt2.drop(columns = ['index'])

X = Xt1.join(Xt2, how='left')
X['year']=list(range(1985,2015))

X.to_csv("/Users/Documents/PostdocUW/proc_data/X_NCAorlst.csv", )