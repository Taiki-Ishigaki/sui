import sympy as sp

import numpy as np
from scipy import integrate

from mathrobo.se3 import *

def test_se3_hat():
  x = sp.symbols("x_{0:6}", Integer=True)
  v = sp.Matrix(x)
  
  m = sp.Matrix([ \
    [0., -v[2], v[1], v[3]], \
    [v[2], 0., -v[0], v[4]], \
    [-v[1], v[0], 0., v[5]],
    [0.,     0.,      0.,     0.]])
  
  res = SE3.hat(v, 'sympy')
  
  assert res == m
  
def test_se3_hat_commute():
  x = sp.symbols("x_{0:6}", Integer=True)
  y = sp.symbols("y_{0:6}", Integer=True)
  v = sp.Matrix(x)
  w = sp.Matrix(y)
  
  w_ = zeros(4, 'sympy')
  w_[0:3,0] = w[0:3,0]
  
  res1 = SE3.hat(v, 'sympy') @ w_
  res2 = SE3.hat_commute(w, 'sympy') @ v
  
  assert res1 == res2
  
def test_se3_vee():
  x = sp.symbols("x_{0:6}", Integer=True)
  v = sp.Matrix(x)

  hat = SE3.hat(v, 'sympy')
  res = SE3.vee(hat, 'sympy')

  assert res == v

def test_se3_mat():
  a = sp.symbols('a')
  x = sp.symbols("x_{0:6}", Integer=True)
  v = sp.Matrix(x)
  
  r = SE3.mat(v, a, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(6)
  
  vec[0:3] = vec[0:3] / np.linalg.norm(vec[0:3] )

  res = sympy_subs_mat(r, x, vec)
  res = res.subs([(a, angle[0])]) 
  
  m = SE3.mat(vec, angle)

  np.testing.assert_allclose(m, sympy_to_numpy(res))
  
def test_se3_integ_mat():
  a = sp.symbols('a')
  x = sp.symbols("x_{0:6}", Integer=True)
  v = sp.Matrix(x)
  
  r = SE3.integ_mat(v, a, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(6)
  
  vec[0:3] = vec[0:3] / np.linalg.norm(vec[0:3] )

  res = sympy_subs_mat(r, x, vec)
  res = res.subs([(a, angle[0])]) 
  
  m = SE3.integ_mat(vec, angle)
  
  np.testing.assert_allclose(m, sympy_to_numpy(res))
  
def test_se3_jac_lie_wrt_scaler():
  a = sp.symbols('a')
  x = sp.symbols("x_{0:6}", Integer=True)
  v = sp.Matrix(x)
  dx = sp.symbols("dx_{0:6}", Integer=True)
  dv = sp.Matrix(dx)
  
  r = jac_lie_wrt_scaler(SE3, v, a, dv, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(6)
  dvec = np.random.rand(6)
  
  vec[0:3] = vec[0:3] / np.linalg.norm(vec[0:3] )
  
  res = sympy_subs_mat(r, x, vec)
  res = sympy_subs_mat(res, dx, dvec)
  res = res.subs([(a, angle[0])]) 
  
  m = jac_lie_wrt_scaler(SE3, vec, angle, dvec)
  
  np.testing.assert_allclose(m, sympy_to_numpy(res))

# def test_so3_jac_lie_wrt_scaler_integ():
#   a_ = sp.symbols('a_')
#   a = sp.symbols('a')
#   x = sp.symbols("x_{0:6}", Integer=True)
#   v = sp.Matrix(x)
#   dx = sp.symbols("dx_{0:6}", Integer=True)
#   dv = sp.Matrix(dx)
  
#   r_ = jac_lie_wrt_scaler(SE3, v, a_, dv, 'sympy')
#   r = sp.integrate(r_, [a_, 0, a])
  
#   angle = np.random.rand(1)
#   vec = np.random.rand(6)
#   dvec = np.random.rand(6)
  
#   vec[0:3] = vec[0:3] / np.linalg.norm(vec[0:3] )
  
#   def integrad(s):
#     return jac_lie_wrt_scaler(SE3, vec, s, dvec)
  
#   m, _ = integrate.quad_vec(integrad, 0, angle)
  
#   res = sympy_subs_mat(r, x, vec)
#   res = sympy_subs_mat(res, dx, dvec)
#   res = res.subs([(a, angle[0])]) 
  
#   np.testing.assert_allclose(m, sympy_to_numpy(res))