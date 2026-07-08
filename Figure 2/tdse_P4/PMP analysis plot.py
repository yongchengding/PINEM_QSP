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
    
    N=10
    tf=4
    nsteps=100
    
    Time, P = np.meshgrid(np.linspace(0,tf,nsteps+1),np.linspace(-N,N,2*N+1))

    plt.figure(figsize = (5,1.5))
    
    plt.pcolormesh(Time, P, pop.T, shading='nearest')
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("sidebamd index")
    plt.ylim([-5,5])
    plt.title("Population dynamics")
    plt.tight_layout()
    #plt.show()
    #plt.savefig('PMP.pdf',dpi=800,bbox_inches='tight',format='pdf')



    import matplotlib as mpl

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica']
    #mpl.rcParams['mathtext.fontset'] = 'stixsans'
    
    darkgreen = "#344532"
    
    plt.figure(figsize = (5,1.3))
    plt.plot(np.linspace(0,tf/4,nsteps),phi/(np.pi),color=darkgreen)
    plt.xlabel(r"$t~/T$")
    plt.ylabel(r"$\phi^{*}(t)~[\pi]$")
    #plt.title("Optimal phase")

    plt.tight_layout()
    #plt.show()
    plt.savefig('phi_opt.pdf',dpi=800,bbox_inches='tight',format='pdf')
    
    


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
    P_target[N+1]=1
    
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

    phi_opt = np.load("phi_opt.npy")

    C_hist=evolve_dynamics(
        N,kappa,epsilon,tf,phi_opt
    )
    
    Cf = C_hist[-1]
    pop = np.abs(Cf)**2
    print(np.sum(np.sqrt(pop)*np.sqrt(P_target))**2)
    

    plot_results(C_hist,phi_opt)
    