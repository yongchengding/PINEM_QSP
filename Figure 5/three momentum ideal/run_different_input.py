#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:44:10 2026

@author: jonzen
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import solve_banded
from Interferrometer import Evolve

Nsam = 12800*4
zmax = 0.4*4
dz = 2 * zmax / Nsam
zrange = np.arange(-Nsam/2, Nsam/2, 1) * dz
dp = 2 * np.pi / (2 * zmax)
prange = np.arange(-Nsam/2, Nsam/2, 1) * dp

# Free Electron / Laser Parameters
c = 3e8             # m/s
beta = 0.05
gamma = 1 / np.sqrt(1 - beta**2)
lam = 0.2           # micron (light wavelength)
omegaL = 2 * np.pi * c / (lam * 1e-6)   # Hz
deltak = 2 * np.pi / (beta * lam)       # 1/micron (Grating momentum q)



sigmak0 = 0.05 * deltak




Cf = np.load("Cf.npy")
N=10 #from PMP
# psip0 = np.zeros(len(prange),dtype=complex)
# for i in range(-4,5):
#     print(i)
#     print('|C_f|^2 = ',abs(Cf[N+i])**2)   
#     psip0 += Cf[N+i] * np.exp(-(prange-i*deltak)**2 / (2 * sigmak0**2)) # low to high index in p space
# psip0 /= np.sqrt(np.sum(np.abs(psip0)**2) * dp)                    

# plt.plot(abs(psip0)**2)
# plt.plot(abs(np.load("datap.npy")[:,-1])**2)
# datap = Evolve(psip0)
# np.save("datap_sim.npy",datap)



theta1_list = np.linspace(-0.5, 0.5, 51)
theta2_list = np.linspace(-0.5, 0.5, 51)

nt1 = len(theta1_list)
nt2 = len(theta2_list)

P0_sim  = np.zeros((nt1, nt2, 100))
Pp1_sim = np.zeros((nt1, nt2, 100))
Pm1_sim = np.zeros((nt1, nt2, 100))
Pp2_sim = np.zeros((nt1, nt2, 100))
Pm2_sim = np.zeros((nt1, nt2, 100))

for a, theta1 in enumerate(theta1_list):

    for b, theta2 in enumerate(theta2_list):

        print('theta1 =', theta1, 'theta2 =', theta2)

        psip0 = np.zeros(len(prange), dtype=complex)

        for i in range(-4, 5):

            amp = Cf[N+i]

            # perturb -2 sideband
            if i == -2:
                amp *= np.exp(1j * theta1)

            # perturb +2 sideband
            elif i == 2:
                amp *= np.exp(1j * theta2)

            psip0 += amp * np.exp(
                -(prange - i*deltak)**2 / (2 * sigmak0**2)
            )

        psip0 /= np.sqrt(np.sum(np.abs(psip0)**2) * dp)

        datap = Evolve(psip0)

        for t in range(100):

            psip = datap[:, t]

            P0_sim[a,b,t] = np.sum(
                abs(psip[25600-150 : 25600+150])**2
            ) * dp

            Pp1_sim[a,b,t] = np.sum(
                abs(psip[25600-150+320 : 25600+150+320])**2
            ) * dp

            Pm1_sim[a,b,t] = np.sum(
                abs(psip[25600-150-320 : 25600+150-320])**2
            ) * dp

            Pp2_sim[a,b,t] = np.sum(
                abs(psip[25600-150+2*320 : 25600+150+2*320])**2
            ) * dp

            Pm2_sim[a,b,t] = np.sum(
                abs(psip[25600-150-2*320 : 25600+150-2*320])**2
            ) * dp

np.save("P0_sim.npy",P0_sim)
np.save("Pp1_sim.npy",Pp1_sim)
np.save("Pm1_sim.npy",Pm1_sim)
np.save("Pp2_sim.npy",Pp2_sim)
np.save("Pm2_sim.npy",Pm2_sim)





















