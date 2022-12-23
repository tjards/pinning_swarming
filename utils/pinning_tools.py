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
import random

# data set for testing 
#data = np.load('state_21.npy') # 21 nodes, 6 states [x,y,z,vx,vy,vz]
#np.random.seed(2)

#%% Hyperparameters
# -----------------

# key ranges 
d       = 5             # lattice scale 
r       = 1.2*d    #20*d           # range at which neighbours can be sensed 
d_prime = 0.6*d     #0.5 #0.6*d    # desired separation 
r_prime = 1.2*d_prime     #2*2*d_prime   # range at which obstacles can be sensed

# gains
c1_a = 1
c2_a = 2*np.sqrt(1)
c1_b = 3
c2_b = 2*np.sqrt(3)
c1_g = 2
c2_g = 2*np.sqrt(2)

#%% Kronrcker product (demo)
# ------------------------
#A1 = np.eye(2)          #   A = m x n matrix
#B1 = data               #   B = p x q matrix
#Kron = np.kron(A1,B1)   #   Kron = pm x qn matrix

#%% Useful functions
# -----------------

# constants for later functions
a   = 5
b   = 5
c   = np.divide(np.abs(a-b),np.sqrt(4*a*b)) 
eps = 0.1
#eps = 0.5
h   = 0.2
pi  = 3.141592653589793


def sigma_norm(z):    
    norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
    return norm_sig

def rho_h(z):    
    if 0 <= z < h:
        rho_h = 1        
    elif h <= z < 1:
        rho_h = 0.5*(1+np.cos(pi*np.divide(z-h,1-h)))    
    else:
        rho_h = 0  
    return rho_h
 
def phi_a(q_i, q_j, r_a, d_a): 
    z = sigma_norm(q_j-q_i)        
    phi_a = rho_h(z/r_a) * phi(z-d_a)    
    return phi_a
    
def phi(z):    
    phi = 0.5*((a+b)*sigma_1(z+c)+(a-b))    
    return phi 

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

def n_ij(q_i, q_j):
    n_ij = np.divide(q_j-q_i,np.sqrt(1+eps*np.linalg.norm(q_j-q_i)**2))    
    return n_ij

def a_ij(q_i, q_j, r_a):        
    a_ij = rho_h(sigma_norm(q_j-q_i)/r_a)
    return a_ij

def b_ik(q_i, q_ik, d_b):        
    b_ik = rho_h(sigma_norm(q_ik-q_i)/d_b)
    return b_ik

def phi_b(q_i, q_ik, d_b): 
    z = sigma_norm(q_ik-q_i)        
    phi_b = rho_h(z/d_b) * (sigma_1(z-d_b)-1)    
    return phi_b

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

# use saber flocking to produce lattice
# --------------------------------------
def compute_cmd_a(states_q, states_p, targets, targets_v, k_node):
    
    # initialize 
    r_a = sigma_norm(r)                         # lattice separation (sensor range)
    d_a = sigma_norm(d)                         # lattice separation (goal)   
    u_int = np.zeros((3,states_q.shape[1]))     # interactions
         
    # search through each neighbour
    for k_neigh in range(states_q.shape[1]):
        # except for itself (duh):
        if k_node != k_neigh:
            # compute the euc distance between them
            dist = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
            # if it is within the interaction range
            if dist < r:
                # compute the interaction command
                u_int[:,k_node] += c1_a*phi_a(states_q[:,k_node],states_q[:,k_neigh],r_a, d_a)*n_ij(states_q[:,k_node],states_q[:,k_neigh]) + c2_a*a_ij(states_q[:,k_node],states_q[:,k_neigh],r_a)*(states_p[:,k_neigh]-states_p[:,k_node]) 

    return u_int[:,k_node] 


# use saber flocking obstacle avoidance command
# ---------------------------------------------
def compute_cmd_b(states_q, states_p, obstacles, walls, k_node):
      
    # initialize 
    d_b = sigma_norm(d_prime)                   # obstacle separation (goal range)
    u_obs = np.zeros((3,states_q.shape[1]))     # obstacles 
    
    # Obstacle Avoidance term (phi_beta)
    # ---------------------------------   
    # search through each obstacle 
    for k_obstacle in range(obstacles.shape[1]):

        # compute norm between this node and this obstacle
        normo = np.linalg.norm(states_q[:,k_node]-obstacles[0:3,k_obstacle])
        
        # ignore if overlapping
        if normo < 0.2:
            continue 
        
        # compute mu
        mu = np.divide(obstacles[3, k_obstacle],normo)
        # compute bold_a_k (for the projection matrix)
        bold_a_k = np.divide(states_q[:,k_node]-obstacles[0:3,k_obstacle],normo)
        bold_a_k = np.array(bold_a_k, ndmin = 2)
        # compute projection matrix
        P = np.identity(states_p.shape[0]) - np.dot(bold_a_k,bold_a_k.transpose())
        # compute beta-agent position and velocity
        q_ik = mu*states_q[:,k_node]+(1-mu)*obstacles[0:3,k_obstacle]
        # compute distance to beta-agent
        dist_b = np.linalg.norm(q_ik-states_q[:,k_node])
        # if it is with the beta range
        if dist_b < r_prime:
            # compute the beta command
            p_ik = mu*np.dot(P,states_p[:,k_node])    
            u_obs[:,k_node] += c1_b*phi_b(states_q[:,k_node], q_ik, d_b)*n_ij(states_q[:,k_node], q_ik) + c2_b*b_ik(states_q[:,k_node], q_ik, d_b)*(p_ik - states_p[:,k_node])
           
    # search through each wall (a planar obstacle)
    for k_wall in range(walls.shape[1]):
        
        # define the wall
        bold_a_k = np.array(np.divide(walls[0:3,k_wall],np.linalg.norm(walls[0:3,k_wall])), ndmin=2).transpose()    # normal vector
        y_k = walls[3:6,k_wall]         # point on plane
        # compute the projection matrix
        P = np.identity(y_k.shape[0]) - np.dot(bold_a_k,bold_a_k.transpose())
        # compute the beta_agent 
        q_ik = np.dot(P,states_q[:,k_node]) + np.dot((np.identity(y_k.shape[0])-P),y_k)
        # compute distance to beta-agent
        dist_b = np.linalg.norm(q_ik-states_q[:,k_node])
        # if it is with the beta range
        maxAlt = 10 # TRAVIS: maxAlt is for testing, only enforces walls below this altitude
        if dist_b < r_prime and states_q[2,k_node] < maxAlt:
            p_ik = np.dot(P,states_p[:,k_node])
            u_obs[:,k_node] += c1_b*phi_b(states_q[:,k_node], q_ik, d_b)*n_ij(states_q[:,k_node], q_ik) + c2_b*b_ik(states_q[:,k_node], q_ik, d_b)*(p_ik - states_p[:,k_node])

        return u_obs[:,k_node] 
    
    
# navigation command
# ------------------
def compute_cmd_g(states_q, states_p, targets, targets_v, k_node, pin_matrix):

    # initialize 
    u_nav = np.zeros((3,states_q.shape[1]))     # navigation

    # select pins
    #pin_matrix = select_pins(states_q)

    # Navigation term 
    # ---------------------------
    u_nav[:,k_node] = - pin_matrix[k_node,k_node]*c1_g*sigma_1(states_q[:,k_node]-targets[:,k_node])- pin_matrix[k_node,k_node]*c2_g*(states_p[:,k_node] - targets_v[:,k_node])
  
    return u_nav[:,k_node]


# select pins
# -----------
def select_pins(states_q):
    pin_matrix = np.zeros((states_q.shape[1],states_q.shape[1]))
    index = random.randint(0,states_q.shape[1])-1
    #index = 1
    pin_matrix[index,index]=1
    index = random.randint(0,states_q.shape[1])-1
    pin_matrix[index,index]=1
    index = random.randint(0,states_q.shape[1])-1
    pin_matrix[index,index]=1

    return pin_matrix

# consolidated control signals
# ----------------------------
def compute_cmd(centroid, states_q, states_p, obstacles, walls, targets, targets_v, k_node, pin_matrix):
    
    # initialize 
    cmd_i = np.zeros((3,states_q.shape[1]))
    
    u_int = compute_cmd_a(states_q, states_p, targets, targets_v, k_node)
    u_obs = compute_cmd_b(states_q, states_p, obstacles, walls, k_node)
    u_nav = compute_cmd_g(states_q, states_p, targets, targets_v, k_node, pin_matrix)
    
    # temp: just nudge all towards centroid, for stability, until I figure this out
    #u_nav += 0.01*c1_g*sigma_1(states_q[:,k_node]-centroid[:,0])
    
    cmd_i[:,k_node] = u_int + u_obs + u_nav
    
    return cmd_i[:,k_node]

#%% compute the controlability matrix
# ---------------------------------
def func_ctrlb(Ai,Bi):
    A = np.mat(Ai)
    B = np.mat(Bi).transpose()
    n = A.shape[0]
    ctrlb = B
    for i in range(1,n):
        ctrlb = np.hstack((ctrlb,A**i*B))
    return ctrlb
        
#%% build Graph (as dictionary)
# ----------------------------
def build_graph(data, r):
    G = {}
    nNodes  = data.shape[1]     # number of agents (nodes)
    # for each node
    for i in range(0,nNodes):
        # create a set of edges
        set_i = set()
        # search through neighbours (will add itself)
        for j in range(0,nNodes):
            # compute distance
            dist = np.linalg.norm(data[0:3,j]-data[0:3,i])
            # if close enough
            if dist < r:
                # add to set_i
                set_i.add(j)
        G[i] = set_i
    return G

#%% find connected components
# --------------------------

def find_connected_components(G):
    # this will record all components 
    all_components = []
    # initialize set of visited nodes 
    visited = set()
    # search through each node in the graph
    for node in G:
        # if it hasn't already been visited
        if node not in visited:
            # find the (sub)components
            component, visited = find_connected_subcomponents(G, node, visited)
            all_components.append(component)
    return all_components


def find_connected_subcomponents(G, node, visited):
        component = []
        nodes = set([node])
        while nodes:
            # pop() pulls out the node and removes it from the list
            node = nodes.pop()
            # updated the listed of visited nodes
            visited.add(node)
            # the graph tells us what edges to add to this (sub)component
            nodes = nodes or G[node] - visited
            # add the nodes to this (sub)component 
            component.append(node)
        return component, visited

#%% 

def select_pins_components(states_q):
    # initialize the pins
    pin_matrix = np.zeros((states_q.shape[1],states_q.shape[1]))
    # build a graph
    G = build_graph(states_q, r)
    # fund the components of the graph
    components = find_connected_components(G)
    
    for i in range(0,len(components)):
        # just take the first in the component for now
        index = components[i][0]
        # note: later, optimize this selection (i.e. instead of [0], use Grammian)
        pin_matrix[index,index]=1

    return pin_matrix
    


         
# %%try it
# ---------
# import numpy as np
# #data = np.load('state_21.npy') # 21 nodes, 6 states [x,y,z,vx,vy,vz]
#data = states_all[0,:,:]
#r = 6         # range to be considered a neighbour 

#P, c = select_pins_components(data,r)

# G = build_graph(data, r)
# components = find_connected_components(G)
# print(components)



#%%%
# gamma   = 1   # coupling strength
# rho     = 1   # pinning strength
# A = compute_adj_matrix(data, r)  
# D = compute_deg_matrix(data, r)   
# L = compute_lap_matrix(A,D)   
# nComp = compute_comp(L) 

# B = np.zeros((A.shape[0]))
# B[1] = 1
# # compute controlability matrix
# Ctrlb = func_ctrlb(A,B)
# # find rank 
# rank = np.linalg.matrix_rank(Ctrlb)
# trace= np.matrix.trace(Ctrlb)

# # let us set pin (manually)
# # -------------------------
# P = np.zeros((data.shape[1],data.shape[1])) # initialize Pin Matrix
# P[0,0] = 1
# P[5,5] = 1


    