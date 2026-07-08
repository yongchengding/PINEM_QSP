#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:15:31 2026

@author: jonzen
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import solve_banded
from Interferrometer import Evolve

np.random.seed(42)

P0_sim = np.load("P0_sim.npy")
Pp1_sim = np.load("Pp1_sim.npy")
Pm1_sim = np.load("Pm1_sim.npy")
Pp2_sim = np.load("Pp2_sim.npy")
Pm2_sim = np.load("Pm2_sim.npy")

dp = 0.625*np.pi

#plot P0
fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1.5, 1.5)    
)
for i in range(101):
    plt.plot(P0_sim[i,:])

data_inf = np.load("datap_after.npy")
P0_inf = np.zeros(100)
for i in range(100):
    psip = data_inf[:,i]
    P0_inf[i] = np.sum(abs(psip[25600-150:25600+150])**2)*dp
plt.plot(P0_inf,color = 'k')
plt.ylabel('P0')



#plot P+1
fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1.5, 1.5)    
)
for i in range(101):
    plt.plot(Pp1_sim[i,:])
Pp1_inf = np.zeros(100)
for i in range(100):
    psip = data_inf[:,i]
    Pp1_inf[i] = np.sum(abs(psip[25600-150+320:25600+150+320])**2)*dp
plt.plot(Pp1_inf,color = 'k')
plt.ylabel('P1')    

#plot P-1
fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1.5, 1.5)    
)
for i in range(101):
    plt.plot(Pm1_sim[i,:])
Pm1_inf = np.zeros(100)
for i in range(100):
    psip = data_inf[:,i]
    Pm1_inf[i] = np.sum(abs(psip[25600-150-320:25600+150-320])**2)*dp
plt.plot(Pm1_inf,color = 'k')
plt.ylabel('P-1') 

#plot P+2
fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1.5, 1.5)    
)
for i in range(101):
    plt.plot(Pp2_sim[i,:])
Pp2_inf = np.zeros(100)
for i in range(100):
    psip = data_inf[:,i]
    Pp2_inf[i] = np.sum(abs(psip[25600-150+2*320:25600+150+2*320])**2)*dp
plt.plot(Pp2_inf,color = 'k')
plt.ylabel('P2')    

#plot P-2
fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1.5, 1.5)    
)
for i in range(101):
    plt.plot(Pm2_sim[i,:])
Pm2_inf = np.zeros(100)
for i in range(100):
    psip = data_inf[:,i]
    Pm2_inf[i] = np.sum(abs(psip[25600-150-2*320:25600+150-2*320])**2)*dp
plt.plot(Pm2_inf,color = 'k')
plt.ylabel('P-2') 


#calculate LS
LS_list = np.zeros(101)
for i in range(101):
    LS_list[i] = np.sum((P0_inf - P0_sim[i,:])**2) 


# LS_random = np.zeros([101,10])
# bins = np.linspace(0, 100, 11)  # 10 intervals
# samples = [np.random.randint(int(bins[i]), int(bins[i+1])) for i in range(10)]
# print(samples)
# for k in range(101):
#     for i in range(10):
#         LS_random[k,i] = 0*(P0_inf[samples[i]]-P0_sim[k,samples[i]])**2
#         LS_random[k,i] += 0*(Pp1_inf[samples[i]]-Pp1_sim[k,samples[i]])**2
#         LS_random[k,i] += 0*(Pm1_inf[samples[i]]-Pm1_sim[k,samples[i]])**2
#         LS_random[k,i] += (Pp2_inf[samples[i]]-Pp2_sim[k,samples[i]])**2
#         LS_random[k,i] += (Pm2_inf[samples[i]]-Pm2_sim[k,samples[i]])**2
# LS_all = np.sum(LS_random,axis=1)
# print(LS_all)
# index = LS_all.argmin()
# print(index)




darkgreen = "#344532"
darkpurple = "#6A2A5B"
cm = LinearSegmentedColormap.from_list('mycmap',["#f5f5f5",darkgreen])

import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['mathtext.fontset'] = 'stixsans'




# number of independent randomizations
print('now we go 10 runs')
n_runs = 10

theta_list = np.linspace(-0.5, 0.5, 101)
theta_results = np.zeros(n_runs)

for run in range(n_runs):
    
    # stratified random sampling
    bins = np.linspace(0, 100, 11)
    samples = [np.random.randint(int(bins[i]), int(bins[i+1])) for i in range(10)]
    
    # compute LS for this sample set
    LS_random = np.zeros([101, 10])
    
    for k in range(101):
        for i in range(10):
            idx = samples[i]
            LS_random[k, i] = (P0_inf[idx] - P0_sim[k, idx])**2
            LS_random[k, i] += (Pp1_inf[idx] - Pp1_sim[k, idx])**2
            LS_random[k, i] += (Pm1_inf[idx] - Pm1_sim[k, idx])**2
            LS_random[k, i] += (Pp2_inf[idx] - Pp2_sim[k, idx])**2
            LS_random[k, i] += (Pm2_inf[idx] - Pm2_sim[k, idx])**2
    
    LS_all = np.sum(LS_random, axis=1)
    index = LS_all.argmin()
    
    theta_results[run] = theta_list[index]
    
    print(f"Run {run}: samples = {samples}, best index = {index}, theta = {theta_list[index]} rad")

# summary
print("\nTheta results (rad):", theta_results)
print("Mean theta:", np.mean(theta_results))
print("Std theta:", np.std(theta_results))

fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1, 1)    
)
trange=np.linspace(0,0.5,100)
plt.plot(trange,P0_sim[index,:],color = darkpurple)
plt.scatter(trange[samples],P0_inf[samples],c=darkgreen,s=10)
theta_list=np.linspace(-0.5,0.5,101)
# plt.ylabel(r'$P_0(t)$')
# plt.xlabel(r'$t~/T$')
plt.ylim([-0.05,1.05])
plt.yticks([0,0.5,1],['','',''],fontsize=14)
plt.xticks([0,0.25,0.5],['','',''],fontsize=14)
plt.savefig(
    'P0_fit.png',
    dpi=800,
    bbox_inches='tight',
    format='png'
)

fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1, 1)    
)
trange=np.linspace(0,0.5,100)
plt.plot(trange,Pp1_sim[index,:],color = darkpurple)
plt.scatter(trange[samples],Pp1_inf[samples],c=darkgreen,s=10)
theta_list=np.linspace(-0.5,0.5,101)
# plt.ylabel(r'$P_1(t)$')
# plt.xlabel(r'$t~/T$')
plt.ylim([-0.05,1.05])
plt.yticks([0,0.5,1],['','',''],fontsize=14)
plt.xticks([0,0.25,0.5],['','',''],fontsize=14)
plt.savefig(
    'Pp1_fit.png',
    dpi=800,
    bbox_inches='tight',
    format='png'
)

fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1, 1)    
)
trange=np.linspace(0,0.5,100)
plt.plot(trange,Pm1_sim[index,:],color = darkpurple)
plt.scatter(trange[samples],Pm1_inf[samples],c=darkgreen,s=10)
theta_list=np.linspace(-0.5,0.5,101)
# plt.ylabel(r'$P_{-1}(t)$')
# plt.xlabel(r'$t~/T$')
plt.ylim([-0.05,1.05])
plt.yticks([0,0.5,1],['','',''],fontsize=14)
plt.xticks([0,0.25,0.5],['','',''],fontsize=14)
plt.savefig(
    'Pm1_fit.png',
    dpi=800,
    bbox_inches='tight',
    format='png'
)

fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1, 1)    
)
trange=np.linspace(0,0.5,100)
plt.plot(trange,Pp2_sim[index,:],color = darkpurple)
plt.scatter(trange[samples],Pp2_inf[samples],c=darkgreen,s=10)
theta_list=np.linspace(-0.5,0.5,101)
#plt.ylabel(r'$P_2(t)$',fontsize=16)
#plt.xlabel(r'$t~/T$',fontsize=16)
plt.ylim([-0.05,1.05])
plt.yticks([0,0.5,1],['','',''],fontsize=14)
plt.xticks([0,0.25,0.5],['','',''],fontsize=14)
plt.savefig(
    'Pp2_fit.png',
    dpi=800,
    bbox_inches='tight',
    format='png'
)

fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(1, 1)    
)
trange=np.linspace(0,0.5,100)
plt.plot(trange,Pm2_sim[index,:],color = darkpurple)
plt.scatter(trange[samples],Pm2_inf[samples],c=darkgreen,s=10)
theta_list=np.linspace(-0.5,0.5,101)
# plt.ylabel(r'$P_{-2}(t)$')
# plt.xlabel(r'$t~/T$')
plt.ylim([-0.05,1.05])
plt.yticks([0,0.5,1],['','',''],fontsize=14)
plt.xticks([0,0.25,0.5],['','',''],fontsize=14)
plt.savefig(
    'Pm2_fit.png',
    dpi=800,
    bbox_inches='tight',
    format='png'
)

#plot the interference
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
lam = 0.2           # micron (light wavelength)
omegaL = 2 * np.pi * c / (lam * 1e-6)   # Hz
deltak = 2 * np.pi / (beta * lam)       # 1/micron (Grating momentum q)

# ---------------------------------------------------------
# 2. EXACT PARAMETER MAPPING
# ---------------------------------------------------------
# On-site energy (dispersion) in 1/fs. 
epsilon = 1e-3 * hbar * (deltak)**2 / (2 * gamma**3 * mass)

# Target hopping energy in 1/fs
kappa_target = s_dim * epsilon

# Reverse-engineer the required Electric Field (E0) to hit s = 1.25
# kappa = 1e-15 * e * E0 * beta * c / (2 * hbar * omegaL)
E0 = kappa_target / (1e-15 * e * beta * c / (2 * hbar * omegaL))
kappa = kappa_target

print(f"--- Mapped Physical Parameters ---")
print(f"Dispersion epsilon:  {epsilon:.4e} 1/fs")
print(f"Coupling kappa:      {kappa:.4e} 1/fs")
print(f"Required Field E0:   {E0:.2e} V/m")
print(f"Klein-Cook Q factor: {epsilon / (2*kappa):.2f} ")

# Total physical time to reach dimensionless t_f = 4
tmax = t_f_dim / epsilon # in fs
print(f"Total sim time:      {tmax:.2f} fs")

# Minimal Coupling Matrix Coefficients 
alpha2 = 1e-3 * hbar / (2 * gamma**3 * mass)                  # Kinetic term [um^2/fs]
v_A0 = 1e-9 * (e * E0) / (mass * gamma * omegaL)              # A*p term velocity [um/fs]
alpha0 = 1e-15 * (e**2 * E0**2) / (4 * mass * gamma * omegaL**2 * hbar) # A^2 term [1/fs]

# ---------------------------------------------------------
# 3. SPACE AND TIME GRIDS
# ---------------------------------------------------------


Nsam = 12800*4
zmax = 0.4*4


dz = 2 * zmax / Nsam
zrange = np.arange(-Nsam/2, Nsam/2, 1) * dz
dp = 2 * np.pi / (2 * zmax)
prange = np.arange(-Nsam/2, Nsam/2, 1) * dp

# FFT operators


Nt = 100 # Discretized to 100 steps
trange = np.linspace(0, tmax/4, Nt)
dt = trange[1] - trange[0]

# ---------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------
# Plotting in DIMENSIONLESS time to prove it maps to TB model



Time, P = np.meshgrid(trange/2 * epsilon, prange / deltak)


fig, ax1 = plt.subplots(
    sharex=True,
    figsize=(2.0, 1.5)    
)
pcm = ax1.pcolormesh(
    Time,
    P,
    np.abs(data_inf)**2,
    shading='nearest',
    cmap=cm
)

ax1.set_ylim([-4, 4])

ax1.set_ylabel(
    r'$(k-k_0)/q$',fontsize=16
)
ax1.set_xlabel(
    r'$t~/T$',fontsize=16
)

ax1.set_yticks([-3,-2,-1,0, 1,2,3],[-3,-2,-1,0, 1,2,3],fontsize=14)
ax1.set_xticks([0, 0.25, 0.5],[0, 0.25, 0.5],fontsize=14)
plt.savefig(
    'interference.png',
    dpi=800,
    bbox_inches='tight',
    format='png'
)

#calculate fidelity
psi0 = data_inf[:,0]
Pm = np.sum(abs(psi0[25600-150-320*2:25600+150-320*2])**2)*dp
Pp = np.sum(abs(psi0[25600-150+320*2:25600+150+320*2])**2)*dp
Cm = np.sqrt(Pm)
Cp = np.sqrt(Pp)

N=10
Cf = np.load("Cf.npy")
angle_ideal = np.angle(Cf[N+2])-np.angle(Cf[N-2])
print("angle from PMP is ", angle_ideal/np.pi, "in unit of pi")
relative_angle = angle_ideal + np.mean(theta_results)
print("angle from reconstruction is ", relative_angle/np.pi, "in unit of pi")
Cp *= np.exp(1j * relative_angle)
cat = np.array([Cm,Cp],dtype = complex)
target = np.array([1/np.sqrt(2),1/np.sqrt(2) * np.exp(1j * 2*np.pi/6)],dtype = complex)
print("Quantum fidelity = ", abs(np.vdot(cat,target))**2)
