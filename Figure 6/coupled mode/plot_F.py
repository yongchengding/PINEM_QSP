#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:55:48 2026

@author: jonzen
"""

import numpy as np
from scipy.linalg import expm
from coupled_evo import *

#Q, N vs 
Q_list = [30,10,2]
N_list = [2,3,4,5]
F = np.zeros([len(Q_list),len(N_list)])
buffer = 30
for i in range(len(Q_list)):
    for j in range(len(N_list)):
        print(i,j)
        target = np.ones(N_list[j],dtype=complex)
        target = normalize(target)
        thetas,phis =reverse_control_pulse(target, Q_list[i], buffer)
        c = forward_propagate(thetas, phis, Q_list[i], buffer)
        cf = c[buffer:buffer+N_list[j]]
        F[i,j] = fidelity(cf,target)
        #print(fid)
        
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap

darkgreen = "#344532"
darkpurple = "#6A2A5B"
cm = LinearSegmentedColormap.from_list('mycmap',["#f5f5f5",darkgreen])

import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['mathtext.fontset'] = 'stixsans'

fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1.3, 1.3)    
)



plt.scatter(N_list,-np.log10(1-F[0,:]),color=darkpurple,marker="o",label=r'$Q=30$')
plt.scatter(N_list,-np.log10(1-F[1,:]),color=darkgreen,marker="^",label=r'$Q=10$')
plt.scatter(N_list,-np.log10(1-F[2,:]),color='#868686',marker="x",label=r'$Q=2$')
plt.ylabel(r'$-\log_{10}(1-F)$',fontsize=16)
plt.xlabel(r'$N$',fontsize=16)
plt.xticks([2,3,4,5],[2,3,4,5],fontsize=14)
plt.yticks([2,4,6,8],[2,4,6,8],fontsize=14)
plt.xlim([1.5,5.5])
plt.ylim([1,8.5])
#plt.legend(fontsize=8,frameon=False,loc='top middle')
           
plt.savefig(
    'F.pdf',
    dpi=800,
    bbox_inches='tight',
    format='pdf'
)