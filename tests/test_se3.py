import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy import integrate
from sui.se3 import *

def test_se3():
  res = SE3()
  e = np.identity(4)

  np.testing.assert_array_equal(res.matrix(), e)
  
def test_se3_inv():
  v = np.random.rand(6) 
  rot = SE3(v)
  
  res = rot.matrix() @ rot.inverse()
  
  e = np.identity(4)
  
  np.testing.assert_allclose(res, e, rtol=1e-15, atol=1e-15)
  
def test_se3_adj():
  v = np.random.rand(6) 
  res = SE3(v)
  
  print(res.adjoint())
  print(expm(SE3.adj_hat(v)))
  
  np.testing.assert_allclose(res.adjoint(), expm(SE3.adj_hat(v)))

def test_se3_hat():
  v = np.random.rand(6)  
  m = np.array([[0., -v[2], v[1], v[3]],
                [v[2], 0., -v[0], v[4]],
                [-v[1], v[0], 0., v[5]],
                [  0.,  0.,  0.,  0.]])
  
  res = SE3.hat(v)

  np.testing.assert_array_equal(res, m)
  
def test_se3_hat_commute():
  v1 = np.random.rand(6)
  v2 = np.append(np.random.rand(3), 0.)

  res1 = SE3.hat(v1) @ v2
  res2 = SE3.hat_commute(v2) @ v1
  
  np.testing.assert_array_equal(res1, res2)
  
def test_se3_vee():
  v = np.random.rand(6)
  
  hat = SE3.hat(v)
  res = SE3.vee(hat)
  
  np.testing.assert_array_equal(v, res)
  
def test_se3_mat():
  v = np.random.rand(6)
  a = np.random.rand(1)
  res = SE3.mat(v, a)

  m = expm(a*SE3.hat(v))
  
  np.testing.assert_allclose(res, m)
  
def test_se3_integ_mat():
  v = np.random.rand(6)
  a = np.random.rand(1)
  res = SE3.integ_mat(v, a)

  def integrad(s):
    return expm(s*SE3.hat(v))
  
  m, _ = integrate.quad_vec(integrad, 0, a)
  
  m[3,3] = m[3,3] / a
  
  np.testing.assert_allclose(res, m)
  
def test_se3_adj_hat():
  v = np.random.rand(6)  
  m = np.array([[0., -v[2], v[1], 0., 0., 0.],
                [v[2], 0., -v[0], 0., 0., 0.],
                [-v[1], v[0], 0., 0., 0., 0.],
                [0., -v[5], v[4], 0., -v[2], v[1]],
                [v[5], 0., -v[3], v[2], 0., -v[0]],
                [-v[4], v[3], 0., -v[1], v[0], 0.]])
  
  res = SE3.adj_hat(v)

  np.testing.assert_array_equal(res, m)
  
def test_se3_adj_hat_commute():
  v1 = np.random.rand(6)
  v2 = np.random.rand(6)
  
  res1 = SE3.adj_hat(v1) @ v2
  res2 = SE3.adj_hat_commute(v2) @ v1
  
  np.testing.assert_allclose(res1, res2)
  
def test_se3_adj_mat():
  v = np.random.rand(6)
  a = np.random.rand(1)
  res = SE3.adj_mat(v, a)

  m = expm(a*SE3.adj_hat(v))
  
  np.testing.assert_allclose(res, m)
  
def test_se3_adj_integ_mat():
  vec = np.random.rand(6)
  angle = np.random.rand()

  res = SE3.adj_integ_mat(vec, angle)

  def integrad(s):
    return expm(s*SE3.adj_hat(vec))
  
  m, _ = integrate.quad_vec(integrad, 0, angle)
  
  print(res)
  print(m)
    
  np.testing.assert_allclose(res, m)
