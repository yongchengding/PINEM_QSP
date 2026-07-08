#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:53:13 2026

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
thetas,phis =reverse_control_pulse(target,2, buffer)
c = forward_propagate(thetas, phis,2, buffer)
c0 = c[buffer]
c1 = c[buffer+1]
c2 = c[buffer+2]
angle10 = np.angle(c1/c0)
angle21 = np.angle(c2/c1)
angle = [angle10,angle21]
plt.scatter([0,1],angle,color='#868686',marker='x')
thetas,phis =reverse_control_pulse(target,10, buffer)
c = forward_propagate(thetas, phis, 10, buffer)
c0 = c[buffer]
c1 = c[buffer+1]
c2 = c[buffer+2]
angle10 = np.angle(c1/c0)
angle21 = np.angle(c2/c1)
angle = [angle10,angle21]
plt.scatter([0,1],angle,color=darkgreen,marker='^')

thetas,phis =reverse_control_pulse(target,30, buffer)
c = forward_propagate(thetas, phis, 30, buffer)
c0 = c[buffer]
c1 = c[buffer+1]
c2 = c[buffer+2]
angle10 = np.angle(c1/c0)
angle21 = np.angle(c2/c1)
angle = [angle10,angle21]
plt.scatter([0,1],angle,color=darkpurple,marker='o')


plt.plot([-1,2],[0,0],linestyle=':',color='k')

plt.xlim([-0.5,1.5])
plt.xticks([0,1],[0,1],fontsize=14)
plt.yticks([-0.2,0,0.2],[-0.2,0,0.2],fontsize=14)
plt.ylabel(r'arg$(c_{n+1}/c_n)$',fontsize=16)
plt.xlabel(r'$n$',fontsize=16)
plt.savefig(
    'angle.pdf',
    dpi=800,
    bbox_inches='tight',
    format='pdf'
)