import numpy as np
from scipy.linalg import expm
from sui.so3 import *

def test_so3():
  res = SO3()
  e = np.identity(3)

  np.testing.assert_array_equal(res.matrix(), e)
  
def test_so3_inv():
  v = np.random.rand(3) 
  rot = SO3(v)
  
  res = rot.matrix() @ rot.inverse()
  
  e = np.identity(3)
  
  np.testing.assert_allclose(res, e, rtol=1e-15, atol=1e-15)
  
def test_so3_adj():
  v = np.random.rand(3) 
  res = SO3(v)
  
  np.testing.assert_array_equal(res.adjoint(), res.matrix())

def test_so3_hat():
  v = np.random.rand(3)  
  m = np.array([[0., -v[2], v[1]],[v[2], 0., -v[0]],[-v[1], v[0], 0.]])
  
  res = SO3.hat(v)

  np.testing.assert_array_equal(res, m)
  
def test_so3_hat_commute():
  v1 = np.random.rand(3)
  v2 = np.random.rand(3)
  
  res1 = SO3.hat(v1) @ v2
  res2 = SO3.hat_commute(v2) @ v1
  
  np.testing.assert_array_equal(res1, res2)
  
def test_so3_vee():
  v = np.random.rand(3)
  
  hat = SO3.hat(v)
  res = SO3.vee(hat)
  
  np.testing.assert_array_equal(v, res)

def test_so3_mat():
  v = np.random.rand(3)
  a = np.random.rand(1)
  res = SO3.mat(v, a)

  m = expm(a*SO3.hat(v))
  
  np.testing.assert_allclose(res, m)