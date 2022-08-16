#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 06:44:00 2021

@author: N
"""

import matplotlib.pyplot as plt

fig1 = plt.figure()
ax = plt.subplot(221)
plt.plot(df_var_WA['date'], df_var_WA['meanBLT_precond'], 'k',label='WA')
plt.plot(df_var_WA['date'], df_var_OR['meanBLT_precond'], 'b',label='OR') 
plt.plot(df_var_WA['date'], df_var_NCA['meanBLT_precond'], 'r',label='N.CA') 
plt.plot(df_var_WA['date'], df_var_SCA['meanBLT_precond'], 'g',label='C.CA')  
plt.ylabel('meanBLT_precond')
plt.xlabel('year')
leg =  plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left')
plt.ylim(6,18)

ax = plt.subplot(222)
plt.plot(df_var_WA['date'], df_var_WA['meanBLT_spwn'], 'k',label='WA')
plt.plot(df_var_WA['date'], df_var_OR['meanBLT_spwn'], 'b',label='OR') 
plt.plot(df_var_WA['date'], df_var_NCA['meanBLT_spwn'], 'r',label='N.CA') 
plt.plot(df_var_WA['date'], df_var_SCA['meanBLT_spwn'], 'g',label='C.CA')  
plt.ylabel('meanBLT_spwn')
plt.xlabel('year')

plt.ylim(6,18)

ax = plt.subplot(223)
plt.plot(df_var_WA['date'], df_var_WA['meanBLT_jvnl0'], 'k',label='WA')
plt.plot(df_var_WA['date'], df_var_OR['meanBLT_jvnl0'], 'b',label='OR') 
plt.plot(df_var_WA['date'], df_var_NCA['meanBLT_jvnl0'], 'r',label='N.CA') 
plt.plot(df_var_WA['date'], df_var_SCA['meanBLT_jvnl0'], 'g',label='C.CA')  
plt.ylabel('meanBLT_jvnl0')
plt.xlabel('year')

plt.ylim(6,18)

ax = plt.subplot(224)
plt.plot(df_var_WA['date'], df_var_WA['maxTLT_z1'], 'k',label='WA')
plt.plot(df_var_WA['date'], df_var_OR['maxTLT_z1'], 'b',label='OR') 
plt.plot(df_var_WA['date'], df_var_NCA['maxTLT_z1'], 'r',label='N.CA') 
plt.plot(df_var_WA['date'], df_var_SCA['maxTLT_z1'], 'g',label='C.CA')  
plt.ylabel('maxTLT_z1')
plt.xlabel('year')

plt.ylim(6,18)

plt.show()
fig1.set_size_inches(14,7)
fig1.savefig('/Users/Documents/PostdocUW/figure/Bottom_layer_discontinuit.pdf',bbox_inches='tight', format='pdf', dpi=1000)

