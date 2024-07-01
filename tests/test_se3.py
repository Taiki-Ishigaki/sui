import numpy as np
from sui.se3 import *

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
  
  np.testing.assert_array_equal(res1, res2)
