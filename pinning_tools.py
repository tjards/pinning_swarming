#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module implements some useful tools for pinning control 

Things to do:
    
- how to properly selected a pinned agent (leader or pin selection)

Preliminaries:
    - Let us consider V nodes (vertices, agents)
    - Define E is a set of edges (links) as the set of ordered pairs
    from the Cartesian Product V x V, E = {(a,b) | a /in V and b /in V}
    - Then we consider Graph, G = {V,E} (nodes and edges)
    - G is simple: (a,a) not \in E \forall a \in V 
    - G is undirected: (a,b) \in E <=> (b,a) \in E
    - Nodes i,j are neighbours if they share an edge, (i,j) /in E
    - d1=|N_1| is the degree of Node 1, or, the number of neighbours

Created on Tue Dec 20 13:32:11 2022

References:
    http://kth.diva-portal.org/smash/get/diva2:681041/FULLTEXT01.pdf

@author: tjards
    
"""

import numpy as np

# data set for testing 
data = np.load('Data/state_25.npy') # 25 nodes, 6 states [x,y,z,vx,vy,vz]

#%% Kronrcker product (demo)
# ------------------------
#A1 = np.eye(2)          #   A = m x n matrix
#B1 = data               #   B = p x q matrix
#Kron = np.kron(A1,B1)   #   Kron = pm x qn matrix

#%% Compute the Adjacency Matrix
# ------------------------------
# A = {a_ij} s.t. 1 if i,j are neighbours, 0 if not
def compute_adj_matrix(data,r):
    # initialize
    nNodes  = data.shape[1]             # number of agents (nodes)
    A       = np.zeros((nNodes,nNodes)) # initialize adjacency matrix as zeros
    # for each node
    for i in range(0,nNodes):  
        # search through neighbours
        for j in range(0,nNodes):
            # skip self
            if i != j:
                # compute distance
                dist = np.linalg.norm(data[0:3,j]-data[0:3,i])
                # if close enough
                if dist < r:
                    # mark as neighbour
                    A[i,j] = 1
    # ensure A = A^T
    assert (A == A.transpose()).all()
    # return the matrix
    return A
               
            
#%% Compute the Degree Matrix
# ------------------------------
# D = diag{d1,d2,...dN}
def compute_deg_matrix(data,r):
    # initialize
    nNodes  = data.shape[1]             # number of agents (nodes)
    D       = np.zeros((nNodes,nNodes)) # initialize degree matrix as zeros
    # for each node
    for i in range(0,nNodes):
        # search through neighbours
        for j in range(0,nNodes):
            # skip self
            if i != j:
                # compute distance
                dist = np.linalg.norm(data[0:3,j]-data[0:3,i])
                # if close enough
                if dist < r:
                    # mark as neighbour
                    D[i,i] += 1
    # return the matrix
    return D

#%% Compute the graph Laplacian
# -----------------------------
def compute_lap_matrix(A,D):
    L = D-A
    eigs = np.linalg.eigvals(L)         # eigen values 
    # ensure L = L^T
    assert (L == L.transpose()).all()
    # ensure has zero row sum
    assert L.sum() == 0
    # ensure Positive Semi-Definite (all eigen values are >= 0)
    assert (eigs >= 0).all()
    # return the matrix
    return L
    
#%% Compute Components: Component set of a graph is a set with no 
#   neighbours outside itself. The number of null eigen values gives 
#   the number of components.
# ------------------------------------------------------------------
#   
def compute_comp(L):
    eigs = np.linalg.eigvals(L)         # eigen values 
    # how many components (how many zero eig values)
    nComp = np.count_nonzero(eigs==0)
    #print('the graph has ', nComp, ' component(s)')
    return nComp

#%% Compute Augmented Laplacian: The number of null eigen values is 
#   the number of components in the graph that do not contain pins.
#   generally, the larger the aug connectivity, the better.
# ----------------------------------------------------------------- 
def compute_aug_lap_matrix(L,P,gamma,rho):
    L_aug = np.multiply(gamma, L) + np.multiply(rho, P)
    eigs = np.linalg.eigvals(L_aug)         # eigen values
    # ensure Positive Semi-Definite (all eigen values are >= 0)
    assert (eigs >= 0).all()
    # tell me if not fully pinned (i.e. there are null eigen values)
    if np.count_nonzero(eigs==0) > 0:
        print('note: graph is not fully pinned')
    # compute the augmented connectivity (smallest eig value)
    aug_connectivity = np.amin(eigs)
    # and the index
    aug_connectivity_i = np.argmin(eigs)
    # return the matrix, augmented connectivity, and index
    return L_aug, aug_connectivity, aug_connectivity_i


# %%try it
# ---------
r = 2         # range to be considered a neighbour 
gamma   = 1   # coupling strength
rho     = 1   # pinning strength
A = compute_adj_matrix(data, r)  
D = compute_deg_matrix(data, r)   
L = compute_lap_matrix(A,D)   
nComp = compute_comp(L) 


# let us set pin (manually)
# -------------------------
P = np.zeros((data.shape[1],data.shape[1])) # initialize Pin Matrix
P[0,0] = 1
P[5,5] = 1

# compute augmented connectivity 
L_aug, aug_connectivity, aug_connectivity_i = compute_aug_lap_matrix(L,P,gamma,rho)



    
    