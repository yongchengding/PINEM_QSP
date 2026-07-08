import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


############################################
# Hamiltonian
############################################

def build_H(phi,kappa,epsilon,N):

    dim = 2*N+1
    H = np.zeros((dim,dim),dtype=complex)

    nvals = np.arange(-N,N+1)

    for i,n in enumerate(nvals):

        H[i,i] = epsilon*n**2

        if i < dim-1:
            H[i,i+1] = kappa*np.exp(1j*phi)

        if i > 0:
            H[i,i-1] = kappa*np.exp(-1j*phi)

    return H


############################################
# derivative dH/dphi
############################################

def dH_dphi(phi,kappa,N):

    dim = 2*N+1
    dH = np.zeros((dim,dim),dtype=complex)

    for i in range(dim-1):

        dH[i,i+1] = 1j*kappa*np.exp(1j*phi)
        dH[i+1,i] = -1j*kappa*np.exp(-1j*phi)

    return dH


############################################
# GRAPE optimizer
############################################

def optimize_phi_grape(N,kappa,epsilon,tf,nsteps,P_target,
                       n_iter):

    dim = 2*N+1
    dt = tf/nsteps

    # random initial guess
    #phi = 2*np.pi*np.random.rand(nsteps)
    #phi = 2*np.pi*np.random.rand(nsteps)
    #phi =  0.5 * np.linspace(0,2*np.pi,nsteps)
    #phi = np.linspace(-1.5*np.pi,1.5*np.pi,nsteps)
    
    
    #phi = (1*np.pi)*np.ones(nsteps) * 0.01 * np.linspace(-0.1,0.1,nsteps)
    
    #phi = (1*np.pi)*np.ones(nsteps) + 0.1 * np.sin(5*np.linspace(0,tf,nsteps))
    
    #phi = (1*np.pi)*np.ones(nsteps) + 0 * np.linspace(0,tf,nsteps)**2
    
    phi = 2 * np.linspace(0,tf,nsteps)**2

    # initial state
    C0 = np.zeros(dim,dtype=complex)
    C0[N] = 1
    
    F_opt = 0

    for iteration in range(n_iter):

        # ---------- forward propagation ----------
        C = np.zeros((nsteps+1,dim),dtype=complex)
        C[0] = C0

        U_list = []

        for t in range(nsteps):

            H = build_H(phi[t],kappa,epsilon,N)
            U = expm(-1j*H*dt)

            U_list.append(U)
            C[t+1] = U@C[t]

        Cf = C[-1]
        pop = np.abs(Cf)**2

        #F = 1-0.5*np.sum((pop-P_target)**2)
        F = np.sum(np.sqrt(pop)*np.sqrt(P_target))**2
        
        if F > F_opt:
            F_opt = F
            phi_opt = np.copy(phi)
        

        # ---------- adjoint ----------
        D = np.zeros((nsteps+1,dim),dtype=complex)
        
        eps = 1e-12
        sqrt_pop = np.sqrt(pop + eps)
        sqrt_target = np.sqrt(P_target)

        S = np.sum(sqrt_pop * sqrt_target)

        D[-1] = S * (sqrt_target / sqrt_pop) * Cf

        #D[-1] = -Cf*(pop-P_target)

        for t in reversed(range(nsteps)):

            U = U_list[t]
            D[t] = U.conj().T@D[t+1]

        # ---------- gradient ----------
        grad = np.zeros(nsteps)

        for t in range(nsteps):

            dH = dH_dphi(phi[t],kappa,N)

            grad[t] = 2*np.imag(
                np.vdot(D[t+1], dH@C[t])
            )*dt

        # adaptive step
        #step = 0.01/(np.max(np.abs(grad))+1e-8)
        step = 0.1

        phi += step*grad
        
        #phi = ((phi + 2*np.pi) % (4*np.pi)) - 2*np.pi
        phi = ((phi + 1*np.pi) % (2*np.pi)) - 1*np.pi

        if iteration%20==0:
            print("iter",iteration,"F=",F)

    return phi_opt,F_opt


############################################
# dynamics propagation
############################################

def evolve_dynamics(N,kappa,epsilon,tf,phi):

    nsteps=len(phi)
    dt=tf/nsteps

    dim=2*N+1

    C=np.zeros((nsteps+1,dim),dtype=complex)
    C[0,N]=1

    for t in range(nsteps):

        H = build_H(phi[t],kappa,epsilon,N)
        U = expm(-1j*H*dt)

        C[t+1] = U@C[t]

    return C


############################################
# plotting
############################################

def plot_results(C_history,phi):

    pop=np.abs(C_history)**2

    plt.figure(figsize=(10,4))

    plt.subplot(121)
    plt.imshow(pop.T,aspect='auto',origin='lower')
    plt.colorbar()
    plt.xlabel("time step")
    plt.ylabel("momentum index")
    plt.title("Population dynamics")

    plt.subplot(122)
    plt.plot(phi)
    plt.xlabel("time step")
    plt.ylabel("phi(t)")
    plt.title("Optimal phase")

    plt.tight_layout()
    plt.show()


############################################
# example
############################################

if __name__=="__main__":

    N=10

    kappa = 1.25
    #kappa = 4
    epsilon=1
    tf=4
    

    
    nsteps=100

    dim=2*N+1

    P_target=np.zeros(dim)
    P_target[N+2]=1
    
    # P_target[N-2]=1/15
    # P_target[N-1]=2/15
    # P_target[N]=3/15
    # P_target[N+1]=4/15
    # P_target[N+2]=5/15
    
    # P_target[N-2] = 1/3
    # P_target[N] = 1/3
    # P_target[N+2] = 1/3
    
    # P_target[N-3] = 0.5
    # P_target[N+2] = 0.5

    # phi_opt,F_opt=optimize_phi_grape(
    #    N,kappa,epsilon,tf,nsteps,P_target,n_iter=200000
    # )

    # print("best F =",F_opt)
    phi_opt = np.load("phi_optP2.npy")
    
    
    #phi_opt = 1.0 * (np.linspace(0,tf,nsteps))**2 
    #phi_opt = 1.5 * np.linspace(0,tf,nsteps)
    
    #phi_opt= (1*np.pi)*np.ones(nsteps) + 0.1 * np.sin(5*np.linspace(0,tf,nsteps))

    C_hist=evolve_dynamics(
        N,kappa,epsilon,tf,phi_opt
    )
    
    Cf = C_hist[-1]
    pop = np.abs(Cf)**2
    print(np.sum(np.sqrt(pop)*np.sqrt(P_target))**2)
    

    plot_results(C_hist,phi_opt)
    
    np.save("phi_opt",phi_opt)