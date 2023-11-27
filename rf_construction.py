#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:53:39 2023

@author: ir337
"""
import numpy as np
import random as rnd
import time
from numpy import array, zeros, diag, diagflat, dot
from math import sqrt
import pprint
import scipy
import scipy.linalg

def create_rf_vector(adj_lists, weight_lists, base_vertex,
                     p_halt, nb_random_walks):
  "create RF vector when we're using iid coupling"
  nb_vertices = len(adj_lists)
  rf_vector = np.zeros(nb_vertices)   #initialise container for RF vector
  for _ in range(nb_random_walks):  
    load = 1.0    #initialise load
    current_vertex = base_vertex    #set walker vertex to starting vertex
    rf_vector[current_vertex] += load   
    if adj_lists[current_vertex] == []:
      return rf_vector
    while (rnd.uniform(0,1) > p_halt):   #draw termination random variables, see if terminates     
        rnd_index = int(rnd.uniform(0,1) * len(adj_lists[current_vertex]))   #choose a neighbouring vertex at random
        multiplier = len(adj_lists[current_vertex])    #get degree of current vertex
        load *= weight_lists[current_vertex][rnd_index]   #update load
        load *= multiplier / (1.0 - p_halt)
        current_vertex = adj_lists[current_vertex][rnd_index]   #update vertex
        rf_vector[current_vertex] += load   #update RF vector
  rf_vector /= nb_random_walks    #normalise by no. random walks
  return rf_vector


def create_antithetic_rf_vector(adj_lists, weight_lists, base_vertex,
                     p_halt, nb_random_walks):
  "creates RF vector when we're using antithetic termination"
  nb_vertices = len(adj_lists) 
  rf_vector = np.zeros(nb_vertices)
  if nb_random_walks%2 != 0:
      raise Exception('Need an even number of walks for antithetic coupling')  #check that possible to pair up walkers
  for _ in range(int(nb_random_walks/2)):
    term_indicator1 = 0   #initialise two termination indicators
    term_indicator2 = 0
    load1 = 1.0   #and two loads
    load2 = 1.0
    current_vertex1 = base_vertex   #and two vertex holder variables
    current_vertex2 = base_vertex
    rf_vector[current_vertex1] += load1
    rf_vector[current_vertex2] += load2
    if adj_lists[current_vertex1] == []:
      return rf_vector
    while term_indicator1 == 0 or term_indicator2 == 0:
        rand_draw1 = rnd.uniform(0,1)
        rand_draw2 = np.mod(rand_draw1+0.5,1)   #second termination RV is offset from first by 1/2 modulo 1 (see Eq. 11)
       
        if term_indicator1 ==0:
            term_indicator1 = rand_draw1 < p_halt
        if term_indicator2 == 0:
            term_indicator2 = rand_draw2 < p_halt
           
        if term_indicator1 ==0:     #do updates to load, vertex position and RF vector for first walker
            rnd_index1 = int(rnd.uniform(0,1) * len(adj_lists[current_vertex1]))
            multiplier1 = len(adj_lists[current_vertex1])
            load1 *= weight_lists[current_vertex1][rnd_index1]
            load1 *= multiplier1 / (1.0 - p_halt)
            current_vertex1 = adj_lists[current_vertex1][rnd_index1]
            rf_vector[current_vertex1] += load1
           
        if term_indicator2 ==0:   #likewise for second walker
            rnd_index2 = int(rnd.uniform(0,1) * len(adj_lists[current_vertex2]))
            multiplier2 = len(adj_lists[current_vertex2])
            load2 *= weight_lists[current_vertex2][rnd_index2]
            load2 *= multiplier2 / (1.0 - p_halt)
            current_vertex2 = adj_lists[current_vertex2][rnd_index2]
            rf_vector[current_vertex2] += load2
       
  rf_vector /= nb_random_walks    #normalise by no. walkers
  return rf_vector

def create_lr_fact_of_squared_inv_id_min_w(adj_lists, weight_lists,
                                           p_halt,
                                           nb_random_walks,term_coupling = None):
  "create lr factorisation of matrix using specified coupling. Estimating (1 - U)^{-2}"  
  rf_vectors = []
  for vertex in range(len(adj_lists)):
    if term_coupling == None:   #if no coupling specified, use i.i.d. walkers
        rf_vector = create_rf_vector(adj_lists, weight_lists,
                                 vertex, p_halt, nb_random_walks)
    
    elif term_coupling == 'antithetic':   #else use antithetic termination
        rf_vector = create_antithetic_rf_vector(adj_lists, weight_lists,
                                 vertex, p_halt, nb_random_walks)
      
    else:
       raise Exception('Coupling scheme not recognised. Either set to None or \'antithetic\'')

    rf_vectors.append(rf_vector)

  A_matrix = np.array(rf_vectors)   #convert stacked random feature vectors to a matrix
  B_matrix = np.transpose(A_matrix)   #return this matrix and its transpose
  
  return A_matrix, B_matrix


