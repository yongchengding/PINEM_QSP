# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap

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
Nsam = 3200
zmax = 0.1 # um #worked perfectly for that 12345

Nsam = 12800
zmax = 0.4


dz = 2 * zmax / Nsam
zrange = np.arange(-Nsam/2, Nsam/2, 1) * dz
dp = 2 * np.pi / (2 * zmax)
prange = np.arange(-Nsam/2, Nsam/2, 1) * dp

# FFT operators


Nt = 100 # Discretized to 100 steps
trange = np.linspace(0, tmax/4, Nt)
dt = trange[1] - trange[0]


datap = np.load("datap.npy")

# ---------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------
# Plotting in DIMENSIONLESS time to prove it maps to TB model

cm = LinearSegmentedColormap.from_list('mycmap',["#f5f5f5","#6A2A5B"])

import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['mathtext.fontset'] = 'stixsans'

Time, P = np.meshgrid(trange * epsilon, prange / deltak)

# plt.pcolormesh(Time, P, np.abs(datap)**2, shading='nearest', cmap=cm)

# plt.figure(figsize=(5, 2))
# plt.pcolormesh(Time, P, np.abs(datap)**2, shading='nearest', cmap=cm)
# plt.ylim([-2,3]) 
# plt.xlabel(r'$t~/T$',fontsize=16)
# plt.ylabel(r'$(k-k_0) / q$',fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks([-1,0,1,2],fontsize=14)
# plt.tight_layout()
# plt.savefig('tdse.png',dpi=800,bbox_inches='tight',format='png')




fig, (ax1, ax2) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(5, 2.0),
    gridspec_kw={
        'height_ratios': [4, 1],
        'hspace': 0.01
    }
)

# ---------------------------------------------------------
# TOP PANEL: TDSE MOMENTUM DYNAMICS
# ---------------------------------------------------------

pcm = ax1.pcolormesh(
    Time,
    P,
    np.abs(datap)**2,
    shading='nearest',
    cmap=cm
)

ax1.set_ylim([-3, 3])

ax1.set_ylabel(
    r'$(k-k_0)/q$',fontsize=16
)

ax1.set_yticks([-2,-1, 0, 1, 2],[-2,-1,0,1,2],fontsize=14)


# remove upper x tick labels
ax1.tick_params(
    labelbottom=False,
    direction='in'
)

# ---------------------------------------------------------
# BOTTOM PANEL: OPTIMAL PHASE
# ---------------------------------------------------------
phi = np.load("phi_opt.npy")

darkgreen = "#344532"
ax2.plot(
    np.linspace(0, 1, 100),
    phi / np.pi,
    color=darkgreen,
    linewidth=2
)

ax2.set_xlabel(
    r'$t~/T$',fontsize=16
)

ax2.set_ylabel(
    r'$\phi^{*}(t)$',fontsize=16
)

ax2.set_ylim([-1.3,1.3])
ax2.set_xticks([0,0.2,0.4,0.6,0.8,1],[0,0.2,0.4,0.6,0.8,1],fontsize=14)
ax2.set_yticks([-1,1],[r'-$\pi$',r'$\pi$'],fontsize=14)

ax2.tick_params(
    direction='out'
)

# ---------------------------------------------------------
# TIGHT LAYOUT
# ---------------------------------------------------------

fig.subplots_adjust(
    hspace=0.02
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------


plt.savefig(
    'combined_figure.png',
    dpi=800,
    bbox_inches='tight',
    format='png'
)

plt.show()
