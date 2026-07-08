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
sigmak0 = 0.05 * deltak

psi_input = np.exp(
    -(prange+0.5*deltak)**2 / (4 * sigmak0**2)
).astype(np.complex128)

for i in range(len(thetas)):
    print('going for interaction zone:',i)
    psif = Zone(Q,thetas[i],phis[i],psi_input)
    #shift the momentum wavefunction
    psif = np.roll(psif,-320)
    psi_input = psif


#shift it back to n=0
psipf = np.roll(psif,320*len(thetas)+160)

c0 = psipf[25600-150:25600+150]
c1 = psipf[25600-150+320:25600+150+320]
c2 = psipf[25600-150+320*2:25600+150+320*2]
c3 = psipf[25600-150+320*3:25600+150+320*3]

plt.plot(abs(c0)**2)
plt.plot(abs(c1)**2)
plt.plot(abs(c2)**2)



P0 = np.sum(abs(c0)**2) * dp
P1 = np.sum(abs(c1)**2) * dp
P2 = np.sum(abs(c2)**2) * dp

print('population =',[P0,P1,P2])
print('relative phase=',np.angle(np.vdot(c1,c0)))
print('relative phase=',np.angle(np.vdot(c2,c1)))

phi01 = np.angle(np.vdot(c1,c0))
phi12 = np.angle(np.vdot(c2,c1))
phi_reconstruct = np.array([np.sqrt(P0), np.sqrt(P1) * np.exp(1j*phi01), np.sqrt(P2) *np.exp(1j*(phi01+phi12))],dtype=complex)
target = np.array([np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)],dtype=complex)
fidelity = abs(np.vdot(phi_reconstruct,target))**2
print('fidelity=',fidelity)




















