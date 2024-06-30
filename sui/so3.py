import numpy as np
from sui.lie_abst import *
from sui.basic import *

class SO3(LieAbstract):
  def __init__(self):
    self.mat = xp.Identity()  
    
  def adjoint(self):
    return self.mat

  @staticmethod
  def hat(vec):
    mat = zeros(3,3)
    mat[1,2] = -vec[0]
    mat[2,1] =  vec[0]
    mat[2,0] = -vec[1]
    mat[0,2] =  vec[1]
    mat[0,1] = -vec[2]
    mat[1,0] =  vec[2]

    return mat
  
  @staticmethod
  def hat_commute(self, vec):
    return -self.hat(vec)

  @staticmethod
  def vee(vec_hat):
    vec = zeros(3,1)
    vec[0] = (-vec_hat[1,2] + vec_hat[2,1]) / 2
    vec[1] = (-vec_hat[2,0] + vec_hat[0,2]) / 2
    vec[2] = (-vec_hat[0,1] + vec_hat[1,0]) / 2
    return vec 

  @staticmethod
  def mat(vec, a):
    """
      回転行列の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIBRARY == 'numpy':
      theta = norm(vec)
      if theta != 1.0:
        a_ = a*theta
      else:
        a_ = a

      x = vec[0]/theta
      y = vec[1]/theta
      z = vec[2]/theta
    elif LIBRARY == 'sympy':
      a_ = a
      x = vec[0]
      y = vec[1]
      z = vec[2]
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    sa = xp.sin(a_)
    ca = xp.cos(a_)

    mat = zeros(3,3)

    mat[0,0] = ca + (1-ca)*x*x
    mat[0,1] = (1-ca)*x*y - sa*z
    mat[0,2] = (1-ca)*x*z + sa*y
    mat[1,0] = (1-ca)*y*x + sa*z
    mat[1,1] = ca + (1-ca)*y*y
    mat[1,2] = (1-ca)*y*z - sa*x
    mat[2,0] = (1-ca)*z*x - sa*y
    mat[2,1] = (1-ca)*z*y + sa*x
    mat[2,2] = ca + (1-ca)*z*z

    return mat
  
  @staticmethod
  def integ_mat(vec, a):
    """
      回転行列の積分の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIBRARY == 'numpy':
      theta = norm(vec)
      if theta != 1.0:
        a_ = a*theta
      else:
        a_ = a

      x = vec[0]/theta
      y = vec[1]/theta
      z = vec[2]/theta
    elif LIBRARY == 'sympy':
      a_ = a
      x = vec[0]
      y = vec[1]
      z = vec[2]
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    sa = xp.sin(a_)
    ca = xp.cos(a_)

    mat = zeros(3,3)

    mat[0,0] = sa + (a_-sa)*x*x
    mat[0,1] = (a_-sa)*x*y - (1-ca)*z
    mat[0,2] = (a_-sa)*x*z + (1-ca)*y
    mat[1,0] = (a_-sa)*y*x + (1-ca)*z
    mat[1,1] = sa + (a_-sa)*y*y
    mat[1,2] = (a_-sa)*y*z - (1-ca)*x
    mat[2,0] = (a_-sa)*z*x - (1-ca)*y
    mat[2,1] = (a_-sa)*z*y + (1-ca)*x
    mat[2,2] = sa + (a_-sa)*z*z

    return mat
  
  def inverse(self):
    return self.mat.transpose()
  
  def adj_hat(self, vec):
    return self.hat(vec)
  
  def adj_hat_commute(self, vec):
    return self.hat(vec)
  
  def adj_mat(self, vec, a):
    return self.mat(vec, a)
  
  def adj_integ_mat(self, vec, a):
    return self.integ_mat(vec, a)