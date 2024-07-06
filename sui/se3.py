from sui.basic import *
from sui.lie_abst import *
from sui.so3 import *

class SE3(LieAbstract):

  def __init__(self, vec = zeros(6), LIB = 'numpy'): 
    '''
    Constructor
    '''
    self._rot = SO3.mat(vec[0:3])
    self._pos = SO3.integ_mat(vec[0:3])@vec[3:6]

    self.lib = LIB
    
  def matrix(self):
    mat = identity(4)
    mat[0:3,0:3] = self._rot
    mat[0:3,3] = self._pos
    return mat
  
  def pos(self):
    return self._pos
  
  def rot(self):
    return self._rot
    
  def inverse(self):
    self._rot = self._rot.transpose()
    self._pos = -self._rot@self._pos
    return self.matrix()
  
  def adjoint(self):
    mat = zeros((6,6), self.lib)
    
    mat[0:3,0:3] = self._rot
    mat[3:6,0:3] = SO3.hat(self._pos, self.lib)@self._rot
    mat[3:6,3:6] = self._rot
    
    return mat

  @staticmethod
  def hat(vec, LIB = 'numpy'):
    '''
    hat operator on the tanget space vector
    '''
    mat = zeros((4,4), LIB)

    mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
    mat[0:3,3] = vec[3:6]

    return mat
  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    '''
    hat commute operator on the tanget space vector
    hat(a) @ b = hat_commute(b) @ a 
    '''
    mat = zeros((4,6), LIB)

    mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
    
    return -mat

  @staticmethod
  def vee(vec_hat, LIB = 'numpy'):
    '''
    a = vee(hat(a))
    '''
    vec = zeros(6, LIB)
    vec[0:3] = SO3.vee(vec_hat[0:3,0:3], LIB)
    vec[3:6] = vec_hat[0:3,3]

    return vec
  
  @staticmethod
  def mat(vec, a = 1., LIB = 'numpy'):
    '''
    同次変換行列の計算
    sympyの場合,vec[0:3]の大きさは1を想定
    '''
    if LIB == 'numpy':
      rot = vec[0:3]
      pos = vec[3:6]
    elif LIB == 'sympy':
      rot = sp.Matrix(vec[0:3])
      pos = sp.Matrix(vec[3:6])
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    mat = zeros((4,4), LIB)
    mat[0:3,0:3] = SO3.mat(rot, a, LIB)
    V = SO3.integ_mat(rot, a, LIB)

    mat[0:3,3] = V @ pos
    mat[3,3] = 1

    return mat

  @staticmethod
  def integ_mat(vec, a = 1., LIB = 'numpy'):
    pass
  
  @staticmethod
  def adj_hat(vec, LIB = 'numpy'):
    mat = zeros((6,6), LIB)

    mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
    mat[3:6,3:6] = SO3.hat(vec[0:3], LIB)
    mat[3:6,0:3] = SO3.hat(vec[3:6], LIB)

    return mat
  
  @staticmethod
  def adj_hat_commute(vec, LIB = 'numpy'):
    return -SE3.adj_hat(vec, LIB)

  @staticmethod
  def adj_vee(vec_hat, LIB = 'numpy'):
    vec = zeros((6,1), LIB)
    
    vec[0,3] = 0.5*(SO3.vee(vec_hat[0:3,0:3], LIB)+SO3.vee(vec_hat[3:6,3:6]), LIB)
    vec[3,6] = SO3.vee(vec_hat[3:6,0:3], LIB)

    return vec
  
  @staticmethod
  def adj_mat(vec, a = 1., LIB = 'numpy'):
    '''
    空間変換行列の計算
    sympyの場合,vec[0:3]の大きさは1を想定
    '''

    h = SE3.mat(vec, a, LIB = 'numpy')

    mat = zeros((6,6))
    mat[0:3,0:3] = h[0:3,0:3]
    mat[3:6,0:3] = SO3.hat(h[0:3,3], LIB)@h[0:3,0:3]
    mat[3:6,3:6] = h[0:3,0:3]

    return mat
  
  @staticmethod
  def adj_integ_mat(vec, a, LIB = 'numpy'):
    
    c = SE3.integ_mat(vec, a, LIB)

    mat = zeros((6,6), LIB)
    mat[0:3,0:3] = c[0:3,0:3]
    mat[3:6,0:3] = SO3.hat(c[0:3,3])@c[0:3,0:3]
    mat[3:6,3:6] = c[0:3,0:3]

    return mat
  
class SE3wrench(SE3):
  
  @staticmethod
  def hat(vec, LIB = 'numpy'):
    mat = zeros((6,6), LIB)
    mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
    mat[3:6,3:6] = SO3.hat(vec[0:3], LIB)
    mat[0:3,3:6] = SO3.hat(vec[3:6], LIB)

    return mat
  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    mat = zeros((6,6), LIB)
    mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
    mat[0:3,3:6] = SO3.hat(vec[3:6], LIB)
    mat[3:6,0:3] = SO3.hat(vec[3:6], LIB)

    return -mat
  
  @staticmethod
  def mat(vec, a, LIB = 'numpy'):
    return SE3.mat(vec, -a, LIB).Transpose()
  
  @staticmethod
  def integ_mat(vec, a, LIB = 'numpy'):
    return SE3.integ_mat(vec, -a, LIB).Transpose()
  
  @staticmethod
  def adj_mat(vec, a, LIB = 'numpy'):
    return SE3.adj_mat(vec, -a, LIB).Transpose()
  
  @staticmethod
  def adj_integ_mat(vec, a, LIB = 'numpy'):
    return SE3.integ_mat(vec, -a, LIB).Transpose()