# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import solve_banded

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
# 2. EXACT PARAMETER MAPPING
# ---------------------------------------------------------

epsilon = 1e-3 * hbar * (deltak)**2 / (2 * gamma**3 * mass)

kappa_target = s_dim * epsilon

E0 = kappa_target / (
    1e-15 * e * beta * c / (2 * hbar * omegaL)
)

kappa = kappa_target

print(f"--- Mapped Physical Parameters ---")
print(f"Dispersion epsilon:  {epsilon:.4e} 1/fs")
print(f"Coupling kappa:      {kappa:.4e} 1/fs")
print(f"Required Field E0:   {E0:.2e} V/m")
print(f"Klein-Cook Q factor: {epsilon / (2*kappa):.2f}")

tmax = t_f_dim / epsilon

print(f"Total sim time:      {tmax:.2f} fs")

# Minimal Coupling Matrix Coefficients
alpha2 = 1e-3 * hbar / (2 * gamma**3 * mass)

v_A0 = 1e-9 * (
    e * E0
) / (mass * gamma * omegaL)

alpha0 = 1e-15 * (
    e**2 * E0**2
) / (
    4 * mass * gamma * omegaL**2 * hbar
)

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

Nt = 100

trange = np.linspace(0, tmax, Nt)

dt = trange[1] - trange[0]

# ---------------------------------------------------------
# 4. CONTROL PHASE
# ---------------------------------------------------------

phi_control = np.load("phi_opt.npy")

# ---------------------------------------------------------
# 5. INITIAL STATE
# ---------------------------------------------------------

sigmak0 = 0.05 * deltak

psip0 = np.exp(
    -(prange)**2 / (4 * sigmak0**2)
).astype(np.complex128)

# normalize in momentum space
psip0 /= np.sqrt(
    np.sum(np.abs(psip0)**2) * dp
)

# inverse FFT -> position space
chiz0 = np.fft.fftshift(
    np.fft.fft(
        np.fft.ifftshift(psip0)
    )
)

# normalize
chiz0 /= np.sqrt(
    np.sum(np.abs(chiz0)**2) * dz
)

# ---------------------------------------------------------
# 6. STORAGE
# ---------------------------------------------------------

dataz = np.zeros(
    (Nsam, Nt),
    dtype=np.complex128
)

datap = np.zeros(
    (Nsam, Nt),
    dtype=np.complex128
)

dataz[:, 0] = chiz0
datap[:, 0] = psip0

# ---------------------------------------------------------
# 7. TDSE DYNAMICS LOOP (OPTIMIZED)
# ---------------------------------------------------------

start = time.time()

identity = np.ones(Nsam, dtype=np.complex128)

for n in range(1, Nt):

    print(f"Step {n}/{Nt-1}")

    # -----------------------------------------------------
    # Phase
    # -----------------------------------------------------

    phit = (
        deltak * zrange
        + 3*np.pi/2
        + phi_control[n]
    )

    sin_phit = np.sin(phit)
    cos_phit = np.cos(phit)

    # -----------------------------------------------------
    # Potentials
    # -----------------------------------------------------

    V_int = -2 * kappa * sin_phit

    V_A2 = alpha0 * sin_phit**2

    V_divA = (
        1j
        * (v_A0 * deltak / 2)
        * cos_phit
    )

    # -----------------------------------------------------
    # Hamiltonian diagonals
    # -----------------------------------------------------

    diag = (
        2 * alpha2 / dz**2
        + V_int
        + V_A2
        + V_divA
    )

    vA_array = v_A0 * sin_phit

    upper = (
        -alpha2 / dz**2
        + 1j * vA_array[:-1] / (2 * dz)
    )

    lower = (
        -alpha2 / dz**2
        - 1j * vA_array[1:] / (2 * dz)
    )

    # -----------------------------------------------------
    # Crank-Nicolson matrices
    # -----------------------------------------------------

    factor = 1j * dt / 2

    A_diag = identity + factor * diag
    A_upper = factor * upper
    A_lower = factor * lower

    B_diag = identity - factor * diag
    B_upper = -factor * upper
    B_lower = -factor * lower

    # -----------------------------------------------------
    # RHS = B psi_old
    # -----------------------------------------------------

    rhs = B_diag * chiz0

    rhs[:-1] += B_upper * chiz0[1:]
    rhs[1:] += B_lower * chiz0[:-1]

    # -----------------------------------------------------
    # Banded matrix format for scipy
    # -----------------------------------------------------

    ab = np.zeros(
        (3, Nsam),
        dtype=np.complex128
    )

    # upper diagonal
    ab[0, 1:] = A_upper

    # main diagonal
    ab[1, :] = A_diag

    # lower diagonal
    ab[2, :-1] = A_lower

    # -----------------------------------------------------
    # Solve tridiagonal system
    # -----------------------------------------------------

    chi = solve_banded(
        (1, 1),
        ab,
        rhs
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    chiz0 = chi / np.sqrt(
        np.sum(np.abs(chi)**2) * dz
    )

    dataz[:, n] = chiz0

    # -----------------------------------------------------
    # FFT to momentum space
    # -----------------------------------------------------

    psip = np.fft.fftshift(
        np.fft.ifft(
            np.fft.ifftshift(chiz0)
        )
    )

    psip /= np.sqrt(
        np.sum(np.abs(psip)**2) * dp
    )

    datap[:, n] = psip

# ---------------------------------------------------------
# 8. TIMING
# ---------------------------------------------------------

print(
    f"\nSimulation completed in "
    f"{time.time() - start:.2f} seconds"
)

# ---------------------------------------------------------
# 9. VISUALIZATION
# ---------------------------------------------------------

Time, P = np.meshgrid(
    trange * epsilon,
    prange / deltak
)

darkred = '#8B0000'

cm = LinearSegmentedColormap.from_list(
    'mycmap',
    ['w', darkred]
)

plt.figure(figsize=(5, 5))

plt.pcolormesh(
    Time,
    P,
    np.abs(datap)**2,
    shading='nearest',
    cmap=cm
)

plt.ylim([-6, 6])

plt.xlabel(
    r'$\tau = \epsilon t$ (Dimensionless)'
)

plt.ylabel(
    r'$p / \hbar q$'
)

plt.title(
    r'Momentum Dynamics with $\phi(t)$ control'
)

plt.tight_layout()

plt.show()

# ---------------------------------------------------------
# 10. SAVE
# ---------------------------------------------------------

np.save("datap.npy", datap)