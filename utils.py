#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:24:18 2023

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



def get_laplacian_matrix(W):
    "to get the graph Laplacian from adj matrix"
    nb_vertices = len(W)
    L = 0.5 * np.eye(nb_vertices)
    degrees = np.sum(W,axis=1)
    for i in range(nb_vertices):
        for j in range(i):
            L[i,j] = - W[i,j]/np.sqrt(degrees[i] * degrees[j]) 
    L += L.T
    return L

def get_unnormalised_laplacian_matrix(W):
    "to get the graph Laplacian from adj matrix"
    nb_vertices = len(W)
    degrees = np.sum(W,axis=1)
    return np.diag(degrees) - W
   
def get_exact_evaluation(A):
    "to calculate th exact form of the matrix"
    nb_vertices = len(A)
    M = np.eye(nb_vertices) - A
    M2 = np.matmul(M,M)
    M2I = np.linalg.inv(M2)   
    return M2I

def adj_matrix_to_lists(A):
  "takes a matrix and returns lists of neighbours and edge weights"
  adj_lists = []
  weight_lists = []

  for i in range(len(A)):
    neighbors = []
    weights = []
    for j in range(len(A[i])):
      if A[i][j] != 0.0:
        neighbors.append(j)
        weights.append(A[i][j])
    adj_lists.append(neighbors)
    weight_lists.append(weights)
    
  return adj_lists, weight_lists

def compute_fro(first_matrix, second_matrix, relative=True):
  "calculates the Frobenius norm"  
  diff_norm = np.linalg.norm(first_matrix - second_matrix)
  if not relative:
    return diff_norm * diff_norm
  else:
    return diff_norm / np.linalg.norm(first_matrix)
