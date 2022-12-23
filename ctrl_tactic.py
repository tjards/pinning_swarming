#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module computes the commands for various swarming strategies 

Note: we have created this separate module to permit mixing and/or sharing between tactic types

Created on Mon Jan  4 12:45:55 2021

@author: tjards

"""

import numpy as np
# import reynolds_tools
# import saber_tools
# import encirclement_tools as encircle_tools
# import lemni_tools
# import staticShapes_tools as statics
# import starling_tools
#import pinning_tools

from utils import pinning_tools, reynolds_tools, saber_tools, lemni_tools, starling_tools  
from utils import encirclement_tools as encircle_tools
from utils import staticShapes_tools as statics

#%% Tactic Command Equations 
# ------------------------  
def commands(states_q, states_p, obstacles, walls, targets, targets_v, targets_enc, targets_v_enc, swarm_prox, tactic_type, centroid, params):   
     
    # initialize 
    u_int = np.zeros((3,states_q.shape[1]))     # interactions
    u_obs = np.zeros((3,states_q.shape[1]))     # obstacles 
    u_nav = np.zeros((3,states_q.shape[1]))     # navigation
    u_enc = np.zeros((3,states_q.shape[1]))     # encirclement 
    u_statics = np.zeros((3,states_q.shape[1])) # statics
    cmd_i = np.zeros((3,states_q.shape[1]))     # store the commands
    pin_matrix = np.zeros((states_q.shape[1],states_q.shape[1])) # store pins 
        
    # if doing Reynolds, reorder the agents 
    if tactic_type == 'reynolds':
        distances = reynolds_tools.order(states_q)
        
    # if doing pinning control, select pins
    if tactic_type == 'pinning':
        #pin_matrix = pinning_tools.select_pins(states_q) 
        pin_matrix = pinning_tools.select_pins_components(states_q) 
        
    # for each vehicle/node in the network
    for k_node in range(states_q.shape[1]): 
                 
        # Reynolds Flocking
        # ------------------
        if tactic_type == 'reynolds':
           
           cmd_i[:,k_node] = reynolds_tools.compute_cmd(targets, centroid, states_q, states_p, k_node, distances)
           
           # steal obstacle avoidance term from saber
           # ----------------------------------------
           u_obs[:,k_node] = saber_tools.compute_cmd_b(states_q, states_p, obstacles, walls, k_node)
        
        
        # Saber Flocking
        # ---------------                                
        if tactic_type == 'saber':
               
            # Lattice Flocking term (phi_alpha)
            # ---------------------------------  
            u_int[:,k_node] = saber_tools.compute_cmd_a(states_q, states_p, targets, targets_v, k_node)    
        
            # Navigation term (phi_gamma)
            # ---------------------------
            u_nav[:,k_node] = saber_tools.compute_cmd_g(states_q, states_p, targets, targets_v, k_node)
                          
            # Obstacle Avoidance term (phi_beta)
            # ---------------------------------   
            u_obs[:,k_node] = saber_tools.compute_cmd_b(states_q, states_p, obstacles, walls, k_node)

        # Encirclement term (phi_delta)
        # ---------------------------- 
        if tactic_type == 'circle':       
            
            u_enc[:,k_node] = encircle_tools.compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node)
            
            # steal obstacle avoidance term from saber
            # ----------------------------------------
            u_obs[:,k_node] = saber_tools.compute_cmd_b(states_q, states_p, obstacles, walls, k_node)
                 
        # Lemniscatic term (phi_lima)
        # ---------------------------- 
        if tactic_type == 'lemni':    
            
            u_enc[:,k_node] = lemni_tools.compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node)
            
            # steal obstacle avoidance term from saber
            # ----------------------------------------
            u_obs[:,k_node] = saber_tools.compute_cmd_b(states_q, states_p, obstacles, walls, k_node)
        
        
        if tactic_type == 'statics':
            u_statics[:,k_node] = statics.compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node)
            
            # steal obstacle avoidance term from saber
            # ----------------------------------------
            u_obs[:,k_node] = saber_tools.compute_cmd_b(states_q, states_p, obstacles, walls, k_node)
                  
        # Starling
        # --------
        if tactic_type == 'starling':
           
            # compute command 
            cmd_i[:,k_node], params = starling_tools.compute_cmd(targets, centroid, states_q, states_p, k_node, params, 0.02)
        
        
        # Pinning
        # --------
        if tactic_type == 'pinning':
            
            cmd_i[:,k_node] = pinning_tools.compute_cmd(centroid, states_q, states_p, obstacles, walls, targets, targets_v, k_node, pin_matrix)
            
            
            
        # Mixer
        # -----         
        if tactic_type == 'saber':
            cmd_i[:,k_node] = u_int[:,k_node] + u_obs[:,k_node] + u_nav[:,k_node] 
        elif tactic_type == 'reynolds':
            cmd_i[:,k_node] = cmd_i[:,k_node] + u_obs[:,k_node] # adds the saber obstacle avoidance 
        elif tactic_type == 'circle':
            cmd_i[:,k_node] = u_obs[:,k_node] + u_enc[:,k_node] 
        elif tactic_type == 'lemni':
            cmd_i[:,k_node] = u_obs[:,k_node] + u_enc[:,k_node]
        elif tactic_type == 'statics':
            cmd_i[:,k_node] = u_obs[:,k_node] + u_statics[:,k_node]
        elif tactic_type == 'starling':
            cmd_i[:,k_node] = cmd_i[:,k_node]
        elif tactic_type == 'pinning':
            cmd_i[:,k_node] = cmd_i[:,k_node]
            
        # if using pinning control
        # pin (agent 0) just does the u_nav part
        # --------------------------------------
        #cmd_i[:,0] = u_nav[:,0] 
        

    cmd = cmd_i    
    
    return cmd, params, pin_matrix




