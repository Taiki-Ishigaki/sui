import numpy as np
from sui.so3 import *

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
