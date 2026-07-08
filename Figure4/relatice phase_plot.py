#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:14:26 2026

@author: jonzen
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap

plt.plot([0,np.pi],[0,np.pi])
plt.xlim([-0.5,np.pi+0.5])
plt.ylim([-0.5,np.pi+0.5])

plt.scatter(np.pi,np.pi-0.025000000000000012)
plt.scatter(5*np.pi/6,5*np.pi/6+0.12800000000000003)
plt.scatter(4*np.pi/6,4*np.pi/6-0.032)
plt.scatter(3*np.pi/6,3*np.pi/6-0.23399999999999999)
plt.scatter(2*np.pi/6,2*np.pi/6-0.184)
plt.scatter(1*np.pi/6,1*np.pi/6-0.026999999999999986)


#deviation from Cf Delta theta in unit of rad
# Mean theta: -0.025000000000000012
# Std theta: 0.025787593916455256

# Mean theta: 0.12800000000000003
# Std theta: 0.03187475490101846

# Mean theta: -0.032
# Std theta: 0.04995998398718719

# Mean theta: -0.23399999999999999
# Std theta: 0.013564659966250529

# Mean theta: -0.184
# Std theta: 0.07130217387990355

# Mean theta: -0.026999999999999986
# Std theta: 0.030016662039607272

#mean = np.array([-0.026999999999999986, -0.184, -0.23399999999999999, -0.032, +0.12800000000000003, -0.025000000000000012])
std = np.array([0.030016662039607272, 0.07130217387990355, 0.013564659966250529, 0.04995998398718719,0.03187475490101846, 0.025787593916455256])

mean = np.array([0.15171555401080045,0.27477936778417533,0.4273414413168672,0.6621075169097621,0.8761098109101835,-1.0115114416780715+2])


#10 experiments before    
n = 10
# 95% confidence interval using t distribution
tval = 2.262 #student-t value
ci95 = tval * std / np.sqrt(n)


darkgreen = "#344532"
darkpurple = "#6A2A5B"
cm = LinearSegmentedColormap.from_list('mycmap',["#f5f5f5",darkgreen])

import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['mathtext.fontset'] = 'stixsans'

plt.subplots(
    sharex=True,
    figsize=(2, 2)    
)

plt.errorbar(
    np.linspace(1/6,1,6)*np.pi,
    mean*np.pi,
    yerr=ci95,
    fmt='o',
    capsize=10,
    ecolor=darkpurple,
    markerfacecolor=darkpurple,
    markeredgecolor='k',
)
plt.plot([0,np.pi],[0,np.pi],color='k',linestyle="--")
plt.xticks(np.linspace(0,1,7)*np.pi,[0,'','',r'$\pi/2$','','',r'$\pi$'],fontsize=14)
plt.yticks(np.linspace(0,1,7)*np.pi,[0,'','',r'$\pi/2$','','',r'$\pi$'],fontsize=14)
plt.ylabel(r'$\Delta\theta_{m}$',fontsize=16)
plt.xlabel(r'$\Delta\theta_{t}$',fontsize=16)
plt.tight_layout()
plt.savefig('relative_phase.png',dpi=800,bbox_inches='tight',format='png')