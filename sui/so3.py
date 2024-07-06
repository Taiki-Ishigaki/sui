from sui.lie_abst import *
from sui.basic import *

class SO3(LieAbstract):
  def __init__(self, vec = zeros(3), LIB = 'numpy'):
    '''
    Constructor
    '''
    self._matrix = SO3.mat(vec)
    self._lib = LIB
    
  def matrix(self):
    return self._matrix
    
  def inverse(self):
    return self._matrix.transpose()

  def adjoint(self):
    return self._matrix

  @staticmethod
  def hat(vec, LIB = 'numpy'):
    mat = zeros((3,3), LIB)
    mat[1,2] = -vec[0]
    mat[2,1] =  vec[0]
    mat[2,0] = -vec[1]
    mat[0,2] =  vec[1]
    mat[0,1] = -vec[2]
    mat[1,0] =  vec[2]

    return mat
  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    return -SO3.hat(vec, LIB)

  @staticmethod
  def vee(vec_hat, LIB = 'numpy'):
    vec = zeros(3, LIB)
    vec[0] = (-vec_hat[1,2] + vec_hat[2,1]) / 2
    vec[1] = (-vec_hat[2,0] + vec_hat[0,2]) / 2
    vec[2] = (-vec_hat[0,1] + vec_hat[1,0]) / 2
    return vec 

  @staticmethod
  def mat(vec, a = 1., LIB = 'numpy'):
    """
      回転行列の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIB == 'numpy':
      theta = norm(vec, LIB)
      if theta != 1.0:
        a_ = a*theta
      else:
        a_ = a
        
      if iszero(theta):
        return identity(3)
      else:
        x = vec[0]/theta
        y = vec[1]/theta
        z = vec[2]/theta           

    elif LIB == 'sympy':
      a_ = a
      x = vec[0]
      y = vec[1]
      z = vec[2]
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    sa = sin(a_, LIB)
    ca = cos(a_, LIB)

    mat = zeros((3,3), LIB)

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
  def integ_mat(vec, a = 1., LIB = 'numpy'):
    """
      回転行列の積分の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIB == 'numpy':
      theta = norm(vec, LIB)
      if theta != 1.0:
        a_ = a*theta
      else:
        a_ = a

      if iszero(theta):
        return a*identity(3)
      else:
        x = vec[0]/theta
        y = vec[1]/theta
        z = vec[2]/theta  
        k = 1./theta
        
    elif LIB == 'sympy':
      a_ = a
      x = vec[0]
      y = vec[1]
      z = vec[2]
      k = 1.
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    sa = sin(a_, LIB)
    ca = cos(a_, LIB)

    mat = zeros((3,3), LIB)

    mat[0,0] = k*sa + k*(a_-sa)*x*x
    mat[0,1] = k*(a_-sa)*x*y - k*(1-ca)*z
    mat[0,2] = k*(a_-sa)*z*x + k*(1-ca)*y
    mat[1,0] = k*(a_-sa)*x*y + k*(1-ca)*z
    mat[1,1] = k*sa + k*(a_-sa)*y*y
    mat[1,2] = k*(a_-sa)*y*z - k*(1-ca)*x
    mat[2,0] = k*(a_-sa)*z*x - k*(1-ca)*y
    mat[2,1] = k*(a_-sa)*y*z + k*(1-ca)*x
    mat[2,2] = k*sa + k*(a_-sa)*z*z

    return mat
  
  @staticmethod
  def integ2nd_mat(vec, a = 1., LIB = 'numpy'):
    """
      回転行列の積分の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIB == 'numpy':
      theta = norm(vec, LIB)
      if theta != 1.0:
        a_ = a*theta
      else:
        a_ = a

      if iszero(theta):
        return a*identity(3)
      else:
        x = vec[0]/theta
        y = vec[1]/theta
        z = vec[2]/theta  
        k = 1./(theta*theta)
        
    elif LIB == 'sympy':
      a_ = a
      x = vec[0]
      y = vec[1]
      z = vec[2]
      k = 1.
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    sa = sin(a_, LIB)
    ca = cos(a_, LIB)

    mat = zeros((3,3), LIB)

    mat[0,0] = k*(1-ca) + k*(0.5*a_**2-1+ca)*x*x
    mat[0,1] = k*(0.5*a_**2-1+ca)*x*y - k*(a_-sa)*z
    mat[0,2] = k*(0.5*a_**2-1+ca)*z*x + k*(a_-sa)*y
    mat[1,0] = k*(0.5*a_**2-1+ca)*x*y + k*(a_-sa)*z
    mat[1,1] = k*(1-ca) + k*(0.5*a_**2-1+ca)*y*y
    mat[1,2] = k*(0.5*a_**2-1+ca)*y*z - k*(a_-sa)*x
    mat[2,0] = k*(0.5*a_**2-1+ca)*z*x - k*(a_-sa)*y
    mat[2,1] = k*(0.5*a_**2-1+ca)*y*z + k*(a_-sa)*x
    mat[2,2] = k*(1-ca) + k*(0.5*a_**2-1+ca)*z*z
    
    return mat
  
  @staticmethod
  def adj_hat(self, vec, LIB = 'numpy'):
    return self.hat(vec, LIB)
  
  @staticmethod
  def adj_hat_commute(self, vec, LIB = 'numpy'):
    return self.hat(vec, LIB)
  
  @staticmethod
  def adj_mat(self, vec, a, LIB = 'numpy'):
    return self.mat(vec, a, LIB)
  
  @staticmethod
  def adj_integ_mat(self, vec, a, LIB = 'numpy'):
    return self.integ_mat(vec, a, LIB)