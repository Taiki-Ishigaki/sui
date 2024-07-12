import sympy as sp

import numpy as np
from scipy import integrate

from mathrobo.so3 import *

def test_so3_hat():
  x, y, z = sp.symbols('x y z')
  v = sp.Matrix([x, y, z])
  
  m = sp.Matrix([[0., -v[2], v[1]],[v[2], 0., -v[0]],[-v[1], v[0], 0.]])
  
  res = SO3.hat(v, 'sympy')
  
  assert res == m
  
def test_so3_hat_commute():
  x, y, z = sp.symbols('x y z')
  a, b, c = sp.symbols('a b c')
  v = sp.Matrix([x, y, z])
  w = sp.Matrix([a, b, c])
  
  res1 = SO3.hat(v, 'sympy') @ w
  res2 = SO3.hat_commute(w, 'sympy') @ v
  
  assert res1 == res2
  
def test_so3_vee():
  x, y, z = sp.symbols('x y z')
  v = sp.Matrix([x, y, z])
  
  hat = SO3.hat(v, 'sympy')
  res = SO3.vee(hat, 'sympy')
  
  assert res == v

def test_so3_mat():
  a = sp.symbols('a')
  x, y, z = sp.symbols('x y z')
  v = sp.Matrix([x, y, z])
  
  r = SO3.mat(v, a, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(3)
  
  vec = vec / np.linalg.norm(vec)

  res = r.subs([(x, vec[0]), (y, vec[1]), (z, vec[2]), (a, angle[0])])
  
  m = SO3.mat(vec, angle)
  
  np.testing.assert_allclose(m, sympy_to_numpy(res))
  
def test_so3_integ_mat():
  a = sp.symbols('a')
  x, y, z = sp.symbols('x y z')
  v = sp.Matrix([x, y, z])
  
  r = SO3.integ_mat(v, a, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(3)
  
  vec = vec / np.linalg.norm(vec)

  res = r.subs([(x, vec[0]), (y, vec[1]), (z, vec[2]), (a, angle[0])])
  
  m = SO3.integ_mat(vec, angle)
  
  np.testing.assert_allclose(m, sympy_to_numpy(res))
  
def test_so3_integ2nd_mat():
  a = sp.symbols('a')
  x, y, z = sp.symbols('x y z')
  v = sp.Matrix([x, y, z])
  
  r = SO3.integ2nd_mat(v, a, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(3)
  
  vec = vec / np.linalg.norm(vec)

  res = r.subs([(x, vec[0]), (y, vec[1]), (z, vec[2]), (a, angle[0])])
  
  m = SO3.integ2nd_mat(vec, angle)
  
  np.testing.assert_allclose(m, sympy_to_numpy(res))
  
def test_so3_ac_lie_wrt_scaler():
  a = sp.symbols('a')
  x, y, z = sp.symbols('x y z')
  dx, dy, dz = sp.symbols('dx dy dz')
  v = sp.Matrix([x, y, z])
  dv = sp.Matrix([dx, dy, dz])
  
  r = jac_lie_wrt_scaler(SO3, v, a, dv, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(3)
  dvec = np.random.rand(3)
  
  vec = vec / np.linalg.norm(vec)

  res = r.subs([(x, vec[0]), (y, vec[1]), (z, vec[2]), (dx, dvec[0]), (dy, dvec[1]), (dz, dvec[2]), (a, angle[0])])
  
  m = jac_lie_wrt_scaler(SO3, vec, angle, dvec)
  
  np.testing.assert_allclose(m, sympy_to_numpy(res))

def test_so3_jac_lie_wrt_scaler_integ():
  a_ = sp.symbols('a_')
  a = sp.symbols('a')
  x, y, z = sp.symbols('x y z')
  dx, dy, dz = sp.symbols('dx dy dz')
  v = sp.Matrix([x, y, z])
  dv = sp.Matrix([dx, dy, dz])
  
  r_ = jac_lie_wrt_scaler(SO3, v, a_, dv, 'sympy')
  res_ = sp.integrate(r_, [a_, 0, a])
  
  angle = np.random.rand(1)
  vec = np.random.rand(3)
  dvec = np.random.rand(3)
  
  vec = vec / np.linalg.norm(vec)
  
  def integrad(s):
    return jac_lie_wrt_scaler(SO3, vec, s, dvec)
  
  m, _ = integrate.quad_vec(integrad, 0, angle)
  
  res = res_.subs([(x, vec[0]), (y, vec[1]), (z, vec[2]), (dx, dvec[0]), (dy, dvec[1]), (dz, dvec[2]), (a, angle[0])])
  
  np.testing.assert_allclose(m, sympy_to_numpy(res))