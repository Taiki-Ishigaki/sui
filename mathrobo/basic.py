#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.06.23 Created by T.Ishigaki

import numpy as np
import sympy as sp
import jax.numpy as jnp

import math

LIB_EPPOR = "Unsupported library. Choose 'numpy', 'sympy' or 'jax."

def iszero(x):
  tolerance = 1e-8  # 許容範囲
  return math.isclose(x, 0, abs_tol=tolerance)

def sin(theta, LIB = 'numpy'):
  if LIB == 'numpy':
    return np.sin(theta)
  elif LIB == 'sympy':
    return sp.sin(theta)
  elif LIB == 'jax':
    return jnp.sin(theta)
  else:
    raise ValueError(LIB_EPPOR)
  
def cos(theta, LIB = 'numpy'):
  if LIB == 'numpy':
    return np.cos(theta)
  elif LIB == 'sympy':
    return sp.cos(theta)
  elif LIB == 'jax':
    return jnp.cos(theta)
  else:
    raise ValueError(LIB_EPPOR)

def zeros(shape, LIB = 'numpy'):
  if LIB == 'numpy':
    return np.zeros(shape)
  elif LIB == 'sympy':
    if len(shape) == 2:
      return sp.zeros(shape[0],shape[1])
    elif len(shape) == 1:
      return sp.vector.zeros(shape)
  elif LIB == 'jax':
    return jnp.zeros(shape)
  else:
    raise ValueError(LIB_EPPOR)
  
def identity(size, LIB = 'numpy'):
  if LIB == 'numpy':
    return np.identity(size)
  elif LIB == 'sympy':
    return sp.identity(size)
  elif LIB == 'jax':
    return jnp.identity(size)
  else:
    raise ValueError(LIB_EPPOR)

def norm(vec, LIB = 'numpy'):
  if LIB == 'numpy':
    return np.linalg.norm(vec)
  elif LIB == 'sympy':
    return sp.sqrt(vec.dot(vec))
  elif LIB == 'jax':
    return jnp.linalg.norm(vec)
  else:
    raise ValueError(LIB_EPPOR)

def jac_lie_wrt_scaler(lie, vec, a, dvec, LIB = 'numpy'):
  m = lie.mat(vec, a, LIB)
  integ_m = -lie.adj_integ_mat(vec, -a, LIB)

  return m @ lie.hat(integ_m @ dvec, LIB)

def jac_adj_lie_wrt_scaler(lie, vec, a, dvec, LIB = 'numpy'):
  m = lie.adj_mat(vec, a, LIB)
  integ_m = -lie.adj_integ_mat(vec, -a, LIB)

  return m @ lie.adj_hat(integ_m @ dvec, LIB)

def jac_lie_v_wrt_vector(lie, vec, a, v, LIB = 'numpy'):
  m = lie.mat(vec, a, LIB)
  integ_m = -lie.adj_integ_mat(vec, -a, LIB)

  return m @ lie.hat_commute(v, LIB) @ integ_m