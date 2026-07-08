#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:16:44 2026

@author: jonzen
"""

import numpy as np
from scipy.linalg import expm

def normalize(v):
    v = np.asarray(v, dtype=complex)
    return v / np.linalg.norm(v)


def fidelity(a, b):
    a = normalize(a)
    b = normalize(b)
    return np.abs(np.vdot(a, b))**2

def build_H(N, buffer, k, phi, Q):
    '''
    N is the maximum sideband from 0, 1, 2, ... , N in the control
    N buffer is the extra sideband for leakage
    theta is the SU2 pulse area
    phi is the SU2 pulse phase
    Q is the klein cook parameter
    k is the kth zone that aims at exchanging k and k+1 sideband
    '''
    H = np.zeros([N+2*buffer+1,N+2*buffer+1],dtype=complex)
    #on-site energy
    n_physical = np.linspace(-buffer,N+buffer,N+2*buffer+1)
    #equivalent detuning
    center = k+0.5
    Delta = -2 * center
    #on-site energy
    epsilon = 1 #match zone.py
    kappa = epsilon/(2*Q)
    for i in range(len(n_physical)):
        H[i,i] = (n_physical[i]**2+Delta*n_physical[i]) * epsilon       
    #hopping energy
    for i in range(len(n_physical)-1):
        coup = kappa * np.exp(-1j * phi)
        H[i, i+1] = coup
        H[i+1, i] = np.conj(coup)    
    return H

def forward_propagate(thetas,phis,Q,buffer):
    N = len(thetas)
    c = np.zeros(N+2*buffer+1,dtype=complex)
    c[buffer] = 1 #c_0 = 1 as initial
    for k in range(len(thetas)):
        theta = thetas[k]
        phi = phis[k]
        H = build_H(N, buffer, k, phi, Q)
        time = theta*Q
        U = expm(-1j*H*time)
        c = U @ c
        #print(abs(c)**2)
    return c

def back_propagate(thetas,phis,Q,buffer):
    N = len(thetas)
    c = np.zeros(N+2*buffer+1,dtype=complex)
    c[buffer] = 1 #c_0 = 1 as initial
    for k in range(len(thetas)):
        theta = thetas[k]
        phi = phis[k]
        H = build_H(N, buffer, k, phi, Q)
        time = theta*Q
        U = expm(1j*H*time)
        c = U @ c
        #print(abs(c)**2)
    return c

def calculate_pulse(c, N, buffer, k,Q):
    c_k1 = c[buffer+k+1]
    c_k = c[buffer+k]
    center = k+0.5
    Delta = -2 * center
    if np.abs(c_k) < 1e-14:
        theta = np.pi
        time = theta * Q
        phase = np.exp(1j * 1 * (2*k + 1 + Delta) * time)
        print('compen',phase)
        phi = np.angle(phase * 1j * c_k1)
    else:
        ratio = 1j * c_k1/c_k
        theta = 2 * np.arctan(np.abs(ratio))
        phi = np.angle(ratio)
    return theta, phi

def reverse_control_pulse(target,Q,buffer):
    target = normalize(target)
    buffer_state = np.zeros(buffer,dtype=complex)
    target_c = np.concatenate([buffer_state,target,buffer_state])
    N = len(target)-1
    c = target_c.copy()
    print(c)
    thetas = np.zeros(N)
    phis = np.zeros(N)
    print(thetas,phis)
    
    for k in range(N-1,-1,-1):
        print('we are calculatin the pulse for ',k,'zone, exchange', k, 'and',k+1,' sideband')
        theta, phi = calculate_pulse(c,N,buffer,k,Q)
        thetas[k] = theta
        phis[k] = phi
        # evolve the pulse to pop k+1 to k
        H = build_H(N, buffer, k, phi, Q)
        time = theta*Q
        U = expm(1j * H*time)
        c = U @ c
        print(abs(c)**2)
        print(thetas,phis)    
    return thetas, phis

if __name__ == "__main__":
    Q =10
    buffer = 50

    
    # thetas = np.array([np.pi/2+0.3,np.pi/2-0.2])
    # phis = np.array([np.pi/3+0.1,5*np.pi/6])
    # phis = np.array([np.pi/2+0.3,np.pi/2-0.2])
    
    # thetas = np.array([np.pi/3-0.1])
    # phis = np.array([5*np.pi/3])
    
    # c = forward_propagate(thetas, phis, Q, buffer)
    # print('Whole pulse')
    # print('amp=',abs(c[buffer:buffer+len(thetas)+1])**2)
    # for i in range(len(thetas)):
    #     print('relative phase=',np.angle(c[buffer+i+1]/c[buffer+i]))
    
    # np.save("thetas",thetas)
    # np.save("phis",phis)
    # print(thetas)
    # print(phis)
    
    target = np.array([np.sqrt(0.7), np.exp(1j * 0.4) * np.sqrt(0.2), np.sqrt(0.1)],dtype=complex)
    target = np.array([np.sqrt(1/3), np.exp(1j * np.pi/2*0) * np.sqrt(1/3), np.sqrt(1/3)],dtype=complex)
    #target = np.array([np.sqrt(1/4), np.exp(1j * np.pi/2*0.8) * np.sqrt(1/4), np.sqrt(1/4),np.exp(1j * np.pi/2*1.2) * np.sqrt(1/4)],dtype=complex)
    thetas,phis =reverse_control_pulse(target, Q, buffer)
    #result = forward_propagate(thetas, phis, Q, buffer)
    c = forward_propagate(thetas, phis, Q, buffer)
    print('amp=',abs(c[buffer:buffer+len(thetas)+1])**2)
    for i in range(len(thetas)):
        print('relative phase=',np.angle(c[buffer+i+1]/c[buffer+i]))
    np.save("thetas",thetas)
    np.save("phis",phis)
    print(phis)

    
    
    
    
    
    
    
    
    
    