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
    - Adjacency Matrix, A = {a_ij} s.t. 1 if i,j are neighbours, 0 if not
    - Degree Matrix, D = diag{d1,d2,...dN}
    - A Component set of a graph is a set with no neighbours outside itself

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

# try it
r = 2 # range to be considered a neighbour 
A = compute_adj_matrix(data, r)                
            
#%% Compute the Degree Matrix
# ------------------------------
def compute_deg_matrix(data,r):
    # initialize
    nNodes  = data.shape[1]             # number of agents (nodes)
    D       = np.zeros((nNodes,nNodes)) # initialize degree matrix as zeros
    # for each node
    for i in range(0,nNodes):
        # initialize the degree
        degree = 0
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
    # how many components
    nComp = np.count_nonzero(eigs==0)
    print('the graph has ', nComp, ' component(s)')
    # return the matrix
    return L, nComp
    


# %%try it
# -------
r = 2 # range to be considered a neighbour 
A = compute_adj_matrix(data, r)  
D = compute_deg_matrix(data, r)   
L, nComp = compute_lap_matrix(A,D)   
    
    