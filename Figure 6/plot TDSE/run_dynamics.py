#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:35:56 2026

@author: jonzen
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import solve_banded
from Zone import Zone

Q = 30 
thetas = np.load("thetas.npy")
phis = np.load("phis.npy")

# ---------------------------------------------------------
# 1. PHYSICAL CONSTANTS & TARGET PARAMETERS
# ---------------------------------------------------------

e = 1.602e-19       # C
hbar = 1.054e-34    # J*s
mass = 9.11e-31     # kg
c = 3e8             # m/s

# Target Tight-Binding Parameters
epsilon_dim = 1.0
s_dim = 1.25
t_f_dim = 4.0

# Free Electron / Laser Parameters
beta = 0.05
gamma = 1 / np.sqrt(1 - beta**2)

lam = 0.2           # micron
omegaL = 2 * np.pi * c / (lam * 1e-6)
deltak = 2 * np.pi / (beta * lam)
# ---------------------------------------------------------
# 3. SPACE AND TIME GRIDS
# ---------------------------------------------------------

Nsam = 12800*4
zmax = 0.4*4

dz = 2 * zmax / Nsam

zrange = (
    np.arange(-Nsam/2, Nsam/2) * dz
)

# FFT momentum grid
dp = 2 * np.pi / (2 * zmax)

prange = (
    np.arange(-Nsam/2, Nsam/2) * dp
)
sigmak0 = 0.01 * deltak

psi_input = np.exp(
    -(prange+0.5*deltak)**2 / (4 * sigmak0**2)
).astype(np.complex128)


psif, data0 = Zone(Q,thetas[0],phis[0],psi_input)
psif = np.roll(psif,-320)
psi_input = psif
psif, data1 = Zone(Q,thetas[1],phis[1],psi_input)


Nts0 = len(data0[0])
Nts1 = len(data1[0])
datap = np.zeros([len(data0),Nts0+Nts1],dtype=complex)
for i in range(Nts0):
    psi = np.roll(data0[:,i],160)
    datap[:,i] = psi
for i in range(Nts1):
    psi = np.roll(data1[:,i],160+320)
    datap[:,i+Nts0] = psi  

t0 = Nts0
t1 = Nts1
trange = np.linspace(0,1,t0+t1)
# Plotting in DIMENSIONLESS time to prove it maps to TB model
Time, P = np.meshgrid(trange, prange / deltak)


cm = LinearSegmentedColormap.from_list('mycmap',["#f5f5f5","#6A2A5B"])

import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']

mpl.rcParams['mathtext.fontset'] = 'stixsans'


plt.figure(figsize=(5, 2))
plt.pcolormesh(Time, P, np.abs(datap)**2, shading='nearest', cmap=cm)
plt.axvline(x=t0/(t0+t1),color='k',linestyle='dashed',linewidth=1)
plt.ylim([-0.5,2.5]) 
plt.xlabel(r'$t~/T$',fontsize=16)
plt.ylabel(r'$(k-k_0) / q$',fontsize=16)
plt.xticks([0,0.2,0.4,0.6,0.8,1],[0,0.2,0.4,0.6,0.8,1], fontsize=14)
plt.yticks([0,1,2],fontsize=14)
plt.tight_layout()
plt.savefig('tdse.png',dpi=800,bbox_inches='tight',format='png')





















