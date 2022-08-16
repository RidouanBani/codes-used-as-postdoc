#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:21:26 2021

@author: N
"""
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

WA  = read_csv("/Users/Documents/PostdocUW/proc_data/WA_results_1000.csv")
OR  = read_csv("/Users/Documents/PostdocUW/proc_data/OR_results_512.csv")
NCA  = read_csv("/Users/Documents/PostdocUW/proc_data/NCA_results_512.csv")
SCA  = read_csv("/Users/Documents/PostdocUW/proc_data/SCA_results_512.csv")
WAq  = read_csv("/Users/Documents/PostdocUW/proc_data/WA_results_quad5000.csv")

WA = WA[list(WA)[2:23]]
WA = WA.fillna(0)
WA[WA!=0]=1
WA = WA.sum()
WA = WA.sort_values()

OR = OR[list(OR)[2:23]]
OR = OR.fillna(0)
OR[OR!=0]=1
OR = OR.sum()
OR = OR.sort_values()

NCA = NCA[list(NCA)[2:23]]
NCA = NCA.fillna(0)
NCA[NCA!=0]=1
NCA = NCA.sum()
NCA = NCA.sort_values()

SCA = SCA[list(SCA)[2:23]]
SCA = SCA.fillna(0)
SCA[SCA!=0]=1
SCA = SCA.sum()
SCA = SCA.sort_values()

WAq = WAq[list(WAq)[2:30]]
WAq = WAq.fillna(0)
WAq[WAq!=0]=1
WAq = WAq.sum()
WAq = WAq.sort_values()


fig = plt.figure()
ax2 = plt.subplot(3, 2, 5)
ax2 = WAq.plot.barh()
ax2.set_title('Washington', y=0.2, pad=-8)

ax1 = plt.subplot(3, 2, 1)
ax1 = WA.plot.barh()
ax1.set_title('Washington', y=0.2, pad=-8)

ax3 = plt.subplot(3, 2, 2)
ax3 = OR.plot.barh()
ax3.set_title('Oregon', y=0.2, pad=-8)

ax4 = plt.subplot(3, 2, 3)
ax4 = NCA.plot.barh()
ax4.set_title('Northern California', y=0.2, pad=-8)

ax5 = plt.subplot(3, 2, 4)
ax54 = SCA.plot.barh()
ax5.set_title('Central California', y=0.2, pad=-8)

fig.tight_layout()
fig.set_size_inches(10,16)
fig.savefig('/Users/Documents/PostdocUW/figure/SIFIF_count.pdf',bbox_inches='tight', format='pdf', dpi=1000)

