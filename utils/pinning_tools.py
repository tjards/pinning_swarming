#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module implements some useful tools for pinning control 

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

Some related work:
    
    https://arxiv.org/pdf/1611.06485.pdf
    http://kth.diva-portal.org/smash/get/diva2:681041/FULLTEXT01.pdf
    https://ieeexplore-ieee-org.proxy.queensu.ca/stamp/stamp.jsp?tp=&arnumber=6762966
    https://ieeexplore-ieee-org.proxy.queensu.ca/document/9275901

@author: tjards
    
"""

#%% Import stuff
# --------------
import numpy as np
import random

#%% Hyperparameters
# -----------------

# key ranges 
d       = 5             # lattice scale 
r       = 1.5*d         # range at which neighbours can be sensed 
d_prime = 0.6*d         # desired separation 
r_prime = 1.2*d_prime   # range at which obstacles can be sensed

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
def compute_adj_matrix(data):
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
def compute_deg_matrix(data):
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

    # Navigation term 
    # ---------------------------
    u_nav[:,k_node] = - pin_matrix[k_node,k_node]*c1_g*sigma_1(states_q[:,k_node]-targets[:,k_node])- pin_matrix[k_node,k_node]*c2_g*(states_p[:,k_node] - targets_v[:,k_node])
  
    return u_nav[:,k_node]

# select pins randomly
# --------------------
def select_pins_random(states_q):
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
def func_ctrlb(Ai,Bi,horizon):
    A = np.mat(Ai)
    B = np.mat(Bi).transpose()
    n = horizon
    #n = A.shape[0]
    ctrlb = B
    for i in range(1,n):
        #ctrlb = np.hstack((ctrlb,A**i*B))
        ctrlb = np.hstack((ctrlb,np.dot(A**i,B)))
    return ctrlb
        
#%% build Graph (as dictionary)
# ----------------------------
def build_graph(data):
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
            #else:
            #    print("debug: ", i," is ", dist, "from ", j)
        G[i] = set_i
    return G

#%% find connected components
# -------------------------
def find_connected_components_A(A):
    all_components = []                                     # stores all connected components
    visited = []                                            # stores all visisted nodes
    for node in range(0,A.shape[1]):                        # search all nodes (breadth)
        if node not in visited:                             # exclude nodes already visited
            component       = []                            # stores component nodes
            candidates = np.nonzero(A[node,:].ravel()==1)[0].tolist()    # create a set of candidates from neighbours 
            component.append(node)
            visited.append(node)
            candidates = list(set(candidates)-set(visited))
            while candidates:                               # now search depth
                candidate = candidates.pop(0)               # grab a candidate 
                visited.append(candidate)                   # it has how been visited 
                subcandidates = np.nonzero(A[:,candidate].ravel()==1)[0].tolist()
                component.append(candidate)
                #component.sort()
                candidates.extend(list(set(subcandidates)-set(candidates)-set(visited))) # add the unique nodes          
            all_components.append(component)
    return all_components

#%% select pins for each component 
# --------------------------------
def select_pins_components(states_q, method):
    
    # initialize the pins
    pin_matrix = np.zeros((states_q.shape[1],states_q.shape[1]))
    
    # compute adjacency matrix
    A = compute_adj_matrix(states_q)
    
    # find the components of the graph
    components = find_connected_components_A(A)
    
    if method == 'gramian':
        
        # for each component
        for i in range(0,len(components)):
            
            # find the adjacency matrix+ of this component 
            states_i = states_q[:,components[i]]
            A = compute_adj_matrix(states_i)
            D = compute_deg_matrix(states_i)
            
            # find gramian trace (i.e. energy demand) of first component
            index_i = components[i][0]
            
            # if this is a lone agent
            if len(components[i])==1:
                # pin it
                pin_matrix[index_i,index_i]=1
                
            else:
                    
                #ctrlable, trace_i = compute_gram_trace(A,D,index_i,A.shape[1])
                ctrlable, trace_i = compute_gram_trace(A,D,0,A.shape[1])
                # set a default pin
                pin_matrix[index_i,index_i]=1
                # note: add a test for controlability here
    
                # cycle through the remaining agents in the component
                for j in range(1,len(components[i])): 
                    
                    #print('component: ',i, 'item: ',j,'agent: ',components[i][j])
                    
                    # find trace (set horizon to num of agents, for now)
                    #ctrlable, trace = compute_gram_trace(A,D,components[i][j],A.shape[1])
                    ctrlable, trace = compute_gram_trace(A,D,j,A.shape[1])
                    
                    # take the smallest energy value
                    if trace < trace_i:
                        # make this the new benchmark
                        trace_i = trace
                        # de-pin the previous
                        pin_matrix[index_i,index_i]=0
                        index_i = components[i][j]
                        # pin this one
                        pin_matrix[index_i,index_i]=1
                                 
    elif method == 'random':
        
        for i in range(0,len(components)):
            # just take the first in the component for now
            index = components[i][0]
            # note: later, optimize this selection (i.e. instead of [0], use Grammian)
            pin_matrix[index,index]=1
    else:
        
        print('Warning: no pin selected.')

    return pin_matrix
    
#%% compute the controlability Gram trace
# -------------------------------------
def compute_gram_trace(A,D,node,horizon):
    
    # define B
    B = np.zeros((A.shape[0]))
    #B = np.ones((A.shape[0]))
    B[node] = 1
    
    # discretize (zero order hold)
    #Ad = np.eye(A.shape[0],A.shape[0])+A*dt
    #Bd = B*dt
    
    # IAW with "transmission" from Appendix of Nozari et al. (2018)
    #D_c_in = compute_deg_matrix(A) # inmport this in
    A_dyn = np.dot(A,np.linalg.inv(D))
    
    #alpha = 1
    #A_dyn = np.exp(alpha*(-np.eye(A.shape[0],A.shape[0])+A))
    
    # compute
    C = func_ctrlb(A_dyn,B, horizon)
    W = np.dot(C,C.transpose())
    
    #test controlability
    rank = np.linalg.matrix_rank(C)
    if rank == C.shape[1]:
        ctrlable = True
    else:
        ctrlable = False
        
    # the trace is inversely prop to the energy required to control network
    trace = np.matrix.trace(W)
    
    return ctrlable, trace
          
 



    