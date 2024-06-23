#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.06.23 Created by T.Ishigaki
import numpy as np
from scipy.linalg import expm
from scipy import integrate

def cross3(vec):
  mat = np.zeros((3,3))
  mat[1,2] = -vec[0]
  mat[2,1] =  vec[0]
  mat[2,0] = -vec[1]
  mat[0,2] =  vec[1]
  mat[0,1] = -vec[2]
  mat[1,0] =  vec[2]

  return mat

def cross4(vec):
  mat = np.zeros((4,4))
  mat[0:3,0:3] = cross3(vec[0:3])
  mat[0:3,3] = vec[3:6]

  return mat

def cross6(vec):
  mat = np.zeros((6,6))
  mat[0:3,0:3] = cross3(vec[0:3])
  mat[3:6,3:6] = cross3(vec[0:3])
  mat[3:6,0:3] = cross3(vec[3:6])

  return mat

def cross6dual(vec):
  mat = np.zeros((6,6))
  mat[0:3,0:3] = cross3(vec[0:3])
  mat[3:6,3:6] = cross3(vec[0:3])
  mat[0:3,3:6] = cross3(vec[3:6])

  return mat

def cross6dual_(vec):
  mat = np.zeros((6,6))
  mat[0:3,0:3] = cross3(vec[0:3])
  mat[0:3,3:6] = cross3(vec[3:6])
  mat[3:6,0:3] = cross3(vec[3:6])

  return -mat

def pX_pt(t, x, px_pt, cross_func):
  m = cross_func(x)

  def integrad(t_):
    return expm(-t_*m)
  result, _ = integrate.quad_vec(integrad, 0, t)
  C = result

  return expm(t*m) @ cross_func(C@px_pt)

def pAdualxv_pa(a, v, t):
  m = cross6dual(a)
  A = expm(t*m)

  cross_v = cross6dual_(v)

  m = cross6(a)

  def integrad(t_):
    return expm(-t_*m)
  result, _ = integrate.quad_vec(integrad, 0, t)
  C = result

  return A @ cross_v @ C