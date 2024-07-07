#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.06.23 Created by T.Ishigaki

import numpy as np
import sympy as sp

import math

def iszero(x):
  tolerance = 1e-8  # 許容範囲
  return math.isclose(x, 0, abs_tol=tolerance)

def sin(theta, LIB = 'numpy'):
  if LIB == 'numpy':
    return np.sin(theta)
  elif LIB == 'sympy':
    return sp.sin(theta)
  else:
    raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")
  
def cos(theta, LIB = 'numpy'):
  if LIB == 'numpy':
    return np.cos(theta)
  elif LIB == 'sympy':
    return sp.cos(theta)
  else:
    raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

def zeros(shape, LIB = 'numpy'):
  if LIB == 'numpy':
    return np.zeros(shape)
  elif LIB == 'sympy':
    if len(shape) == 2:
      return sp.zeros(shape[0],shape[1])
    elif len(shape) == 1:
      return sp.vector.zeros(shape)
  else:
    raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")
  
def identity(size, LIB = 'numpy'):
  if LIB == 'numpy':
    return np.identity(size)
  elif LIB == 'sympy':
    return sp.identity(size)
  else:
    raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")  

def norm(vec, LIB = 'numpy'):
    if LIB == 'numpy':
      return np.linalg.norm(vec)
    elif LIB == 'sympy':
      return sp.sqrt(vec.dot(vec))
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

def jac_lie_wrt_scaler(lie, vec, a, dvec):
  m = lie.mat(vec, a)
  integ_m = -lie.adj_integ_mat(vec, -a)

  return m @ lie.hat(integ_m @ dvec)

def jac_adj_lie_wrt_scaler(lie, vec, a, dvec):
  m = lie.adj_mat(vec, a)
  integ_m = -lie.adj_integ_mat(vec, -a)

  return m @ lie.adj_hat(integ_m @ dvec)

def jac_lie_v_wrt_vector(lie, vec, a, v):
  m = lie.mat(vec, a)
  integ_m = -lie.adj_integ_mat(vec, -a)

  return m @ lie.hat_commute(v) @ integ_m