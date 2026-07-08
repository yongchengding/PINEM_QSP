#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:29:54 2026

@author: jonzen
"""

import numpy as np
from scipy.linalg import expm
from coupled_evo import *
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



N=3
buffer= 10
target = np.ones(3,dtype=complex)
target = normalize(target)
width = 0.25
n_phys = np.array([-2,-1,0,1,2,3,4],dtype=float)

fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1.3, 1.3)    
)
thetas,phis =reverse_control_pulse(target, 30, buffer)
c = forward_propagate(thetas, phis, 30, buffer)
plt.bar(n_phys-width,abs(c[buffer-2:buffer+N+2])**2,width,color=darkpurple)
thetas,phis =reverse_control_pulse(target, 10, buffer)
c = forward_propagate(thetas, phis, 10, buffer)
plt.bar(n_phys,abs(c[buffer-2:buffer+N+2])**2,width,color=darkgreen)
thetas,phis =reverse_control_pulse(target, 2, buffer)
c = forward_propagate(thetas, phis, 2, buffer)
plt.bar(n_phys+width,abs(c[buffer-2:buffer+N+2])**2,width,color='#868686')
plt.plot(n_phys,0*n_phys+1/3,linestyle=':',color='k')
plt.ylabel(r'$P_n$',fontsize=16)
plt.xlabel(r'$n$',fontsize=16)
plt.yticks([0,0.1,0.2,0.3],[0,0.1,0.2,0.3],fontsize=14)
plt.xticks([-1,0,1,2,3],[-1,0,1,2,3],fontsize=14)
plt.xlim([-1.5,3.5])

plt.savefig(
    'P.pdf',
    dpi=800,
    bbox_inches='tight',
    format='pdf'
)