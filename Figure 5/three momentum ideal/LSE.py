import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

np.random.seed(42)
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['mathtext.fontset'] = 'stixsans'



# =========================================================
# LOAD DATA
# =========================================================

P0_sim  = np.load("P0_sim.npy")   # shape = (101,101,100)
Pp1_sim = np.load("Pp1_sim.npy")
Pm1_sim = np.load("Pm1_sim.npy")
Pp2_sim = np.load("Pp2_sim.npy")
Pm2_sim = np.load("Pm2_sim.npy")

dp = 0.625*np.pi

# =========================================================
# LOAD EXPERIMENTAL / INFERENCE DATA
# =========================================================

data_inf = np.load("datap_after.npy")

P0_inf  = np.zeros(100)
Pp1_inf = np.zeros(100)
Pm1_inf = np.zeros(100)
Pp2_inf = np.zeros(100)
Pm2_inf = np.zeros(100)

for i in range(100):

    psip = data_inf[:, i]

    P0_inf[i] = np.sum(
        abs(psip[25600-150:25600+150])**2
    ) * dp

    Pp1_inf[i] = np.sum(
        abs(psip[25600-150+320:25600+150+320])**2
    ) * dp

    Pm1_inf[i] = np.sum(
        abs(psip[25600-150-320:25600+150-320])**2
    ) * dp

    Pp2_inf[i] = np.sum(
        abs(psip[25600-150+2*320:25600+150+2*320])**2
    ) * dp

    Pm2_inf[i] = np.sum(
        abs(psip[25600-150-2*320:25600+150-2*320])**2
    ) * dp

# =========================================================
# STYLE
# =========================================================

darkgreen = "#344532"
darkpurple = "#6A2A5B"

cm = LinearSegmentedColormap.from_list(
    'mycmap',
    ["#f5f5f5", darkgreen]
)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['mathtext.fontset'] = 'stixsans'

# =========================================================
# PARAMETER GRID
# =========================================================

theta1_list = np.linspace(-0.5, 0.5, 51)
theta2_list = np.linspace(-0.5, 0.5, 51)

n1 = len(theta1_list)
n2 = len(theta2_list)

# =========================================================
# FITTING
# =========================================================

print('now we go 10 runs')

n_runs = 10

theta1_results = np.zeros(n_runs)
theta2_results = np.zeros(n_runs)

n1 = len(theta1_list)
n2 = len(theta2_list)

for run in range(n_runs):

    # stratified random sampling
    bins = np.linspace(0, 100, 11)

    samples = [
        np.random.randint(int(bins[i]), int(bins[i+1]))
        for i in range(10)
    ]

    # LS landscape
    LS_all = np.zeros((n1, n2))

    for a in range(n1):

        for b in range(n2):

            LS = 0

            for idx in samples:

                LS += (P0_inf[idx]  - P0_sim[a,b,idx])**2
                LS += (Pp1_inf[idx] - Pp1_sim[a,b,idx])**2
                LS += (Pm1_inf[idx] - Pm1_sim[a,b,idx])**2
                LS += (Pp2_inf[idx] - Pp2_sim[a,b,idx])**2
                LS += (Pm2_inf[idx] - Pm2_sim[a,b,idx])**2

            LS_all[a,b] = LS
            print(LS,a,b)

    # best fit
    best_index = np.unravel_index(
        np.argmin(LS_all),
        LS_all.shape
    )

    best_a = best_index[0]
    best_b = best_index[1]

    theta1_results[run] = theta1_list[best_a]
    theta2_results[run] = theta2_list[best_b]

    print(
        f"Run {run}: "
        f"theta1 = {theta1_list[best_a]:.4f}, "
        f"theta2 = {theta2_list[best_b]:.4f}"
    )

# =========================================================
# SUMMARY
# =========================================================

print("\nTheta1 results:")
print(theta1_results)

print("\nTheta2 results:")
print(theta2_results)

print("\nTheta1 mean =", np.mean(theta1_results))
print("Theta1 std  =", np.std(theta1_results, ddof=1))

print("\nTheta2 mean =", np.mean(theta2_results))
print("Theta2 std  =", np.std(theta2_results, ddof=1))

# =========================================================
# BEST FIT PARAMETERS
# =========================================================

theta1_best = np.mean(theta1_results)
theta2_best = np.mean(theta2_results)

best_a = np.argmin(np.abs(theta1_list - theta1_best))
best_b = np.argmin(np.abs(theta2_list - theta2_best))

# =========================================================
# TIME AXIS
# =========================================================

trange = np.linspace(0, 0.5, 100)

# =========================================================
# GENERIC PLOTTING FUNCTION
# =========================================================

def make_plot(sim_data, inf_data, ylabel, filename):

    fig, ax = plt.subplots(
        sharex=True,
        figsize=(1.5, 1.5)
    )

    plt.plot(
        trange,
        sim_data[best_a, best_b, :],
        color=darkpurple
    )

    plt.scatter(
        trange[samples],
        inf_data[samples],
        c=darkgreen,
        s=10
    )

    # plt.ylabel(ylabel)
    # plt.xlabel(r'$t~/T$')

    plt.ylim([0,1])

    plt.yticks([0,0.5,1],['','',''])
    plt.xticks([0,0.25,0.5],['','',''])

    plt.savefig(
        filename,
        dpi=800,
        bbox_inches='tight',
        format='png'
    )

# =========================================================
# MAKE ALL PLOTS
# =========================================================

make_plot(
    P0_sim,
    P0_inf,
    r'$P_0(t)$',
    'P0_fit.png'
)

make_plot(
    Pp1_sim,
    Pp1_inf,
    r'$P_1(t)$',
    'Pp1_fit.png'
)

make_plot(
    Pm1_sim,
    Pm1_inf,
    r'$P_{-1}(t)$',
    'Pm1_fit.png'
)

make_plot(
    Pp2_sim,
    Pp2_inf,
    r'$P_2(t)$',
    'Pp2_fit.png'
)

make_plot(
    Pm2_sim,
    Pm2_inf,
    r'$P_{-2}(t)$',
    'Pm2_fit.png'
)

# =========================================================
# 95% CONFIDENCE INTERVALS
# =========================================================

tval = 2.262
n_runs = len(theta1_results)

theta1_mean = np.mean(theta1_results)
theta2_mean = np.mean(theta2_results)

theta1_std = np.std(theta1_results, ddof=1)
theta2_std = np.std(theta2_results, ddof=1)

theta1_ci95 = tval * theta1_std / np.sqrt(n_runs)
theta2_ci95 = tval * theta2_std / np.sqrt(n_runs)

print("theta1 = %.4f ± %.4f rad (95%% CI)" %
      (theta1_mean, theta1_ci95))

print("theta2 = %.4f ± %.4f rad (95%% CI)" %
      (theta2_mean, theta2_ci95))

# =========================================================
# LS LANDSCAPE WITH 95% CI
# =========================================================

fig, ax = plt.subplots(figsize=(2,2))

pcm = plt.pcolormesh(
    theta1_list,
    theta2_list,
    LS_all.T,
    shading='nearest',
    cmap=cm
)

# plt.colorbar(label='LS')

# best-fit point + confidence interval
plt.errorbar(
    theta1_mean,
    theta2_mean,

    xerr=theta1_ci95,
    yerr=theta2_ci95,

    fmt='o',

    color=darkpurple,
    ecolor=darkpurple,

    capsize=3,
    markersize=4,

    markerfacecolor=darkpurple,
    markeredgecolor='k'
)


#detuned from other folder
plt.errorbar(
    0.26399999999999996,
    -0.398,

    xerr=0.0232,
    yerr=0.0274,

    fmt='^',

    color=darkgreen,
    ecolor=darkgreen,

    capsize=3,
    markersize=4,

    markerfacecolor=darkgreen,
    markeredgecolor=darkgreen
)

#wider
plt.errorbar(
    0.45200000000000007,
    -0.2879999999999999,

    xerr=0.0422,
    yerr=0.1060,

    fmt='x',

    color='#868686',
    ecolor='#868686',

    capsize=3,
    markersize=4,

    markerfacecolor='#868686',
    markeredgecolor='#868686'
)




plt.xlabel(r'$\delta\theta_{-2}$',fontsize=16)
plt.ylabel(r'$\delta\theta_{2}$',fontsize=16)

plt.xticks([-0.5,0,0.5],fontsize=14)
plt.yticks([-0.5,0,0.5],fontsize=14)

plt.savefig(
    'LS_landscape.png',
    dpi=800,
    bbox_inches='tight'
)

plt.show()


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
Pi = np.sum(abs(psi0[25600-150-320*0:25600+150-320*0])**2)*dp
Pp = np.sum(abs(psi0[25600-150+320*2:25600+150+320*2])**2)*dp
Cm = np.sqrt(Pm)
Ci = np.sqrt(Pi)
Cp = np.sqrt(Pp)


N=10
Cf = np.load("Cf.npy")
PMP_mi = np.angle(Cf[N-2]) - np.angle(Cf[N]) 
PMP_ip = np.angle(Cf[N+2]) - np.angle(Cf[N]) + np.pi*2

PMP_mi += np.mean(theta1_results)
PMP_ip += np.mean(theta2_results)

Cm *= np.exp(1j*PMP_mi)
Cp *= np.exp(1j*PMP_ip)

three = np.array([Cm,Ci,Cp],dtype=complex)
target = np.array([(np.exp(-1j*np.pi/2))*1/np.sqrt(3), 1/np.sqrt(3), (np.exp(1j*np.pi/2))*1/np.sqrt(3)],dtype=complex)
print("Quantum fidelity = ", abs(np.vdot(three,target))**2)