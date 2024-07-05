from sui.basic import *
from sui.lie_abst import *
from sui.so3 import *

class SE3(LieAbstract):

  def __init__(self): 
    '''
    Constructor
    '''
    pass

  @staticmethod
  def hat(vec):
    '''
    hat operator on the tanget space vector
    '''
    mat = zeros((4,4))

    mat[0:3,0:3] = SO3.hat(vec[0:3])
    mat[0:3,3] = vec[3:6]

    return mat
  
  @staticmethod
  def hat_commute(vec):
    '''
    hat commute operator on the tanget space vector
    hat(a) @ b = hat_commute(b) @ a 
    '''
    mat = zeros((4,6))

    mat[0:3,0:3] = SO3.hat(vec[0:3])
    
    return -mat

  @staticmethod
  def vee(vec_hat):
    '''
    a = vee(hat(a))
    '''
    vec = zeros(6)
    vec[0:3] = SO3.vee(vec_hat[0:3,0:3])
    vec[3:6] = vec_hat[0:3,3]

    return vec
  
  @staticmethod
  def mat(vec, a):
    '''
    同次変換行列の計算
    sympyの場合,vec[0:3]の大きさは1を想定
    '''
    if LIBRARY == 'numpy':
      rot = vec[0:3]
      pos = vec[3:6]
    elif LIBRARY == 'sympy':
      rot = xp.Matrix(vec[0:3])
      pos = xp.Matrix(vec[3:6])
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    mat = zeros((4,4))
    mat[0:3,0:3] = SO3.mat(rot, a)
    V = SO3.integ_mat(rot, a)

    mat[0:3,3] = V @ pos
    mat[3,3] = 1

    return mat

  @staticmethod
  def integ_mat(vec, a):
    pass
  
  def inverse(self):
    self.mat[0:3,0:3] = self.mat[0:3,0:3].Transpose()
    self.mat[0:3,3] = self.mat[0:3,0:3]@self.mat[0:3,0:3]
    return self.mat
  
  def adjoint(self):
    mat = zeros(6,6)
    
    mat[0:3,0:3] = self.mat[0:3,0:3]
    mat[3:6,3:6] = SO3.hat(self.mat[3,0:3])@self.mat[0:3,0:3]
    mat[3:6,0:3] = self.mat[0:3,0:3]
    
    return mat
  
  @staticmethod
  def adj_hat(vec):
    mat = zeros((6,6))

    mat[0:3,0:3] = SO3.hat(vec[0:3])
    mat[3:6,3:6] = SO3.hat(vec[0:3])
    mat[3:6,0:3] = SO3.hat(vec[3:6])

    return mat
  
  @staticmethod
  def adj_hat_commute(vec):
    return -SE3.adj_hat(vec)

  @staticmethod
  def adj_vee(vec_hat):
    vec = zeros((6,1))
    
    vec[0,3] = 0.5*(SO3.vee(vec_hat[0:3,0:3])+SO3.vee(vec_hat[3:6,3:6]))
    vec[3,6] = SO3.vee(vec_hat[3:6,0:3])

    return vec
  
  @staticmethod
  def adj_mat(vec, a):
    '''
    空間変換行列の計算
    sympyの場合,vec[0:2]の大きさは1を想定
    '''

    h = SE3.mat(vec, a)

    mat = zeros((6,6))
    mat[0:3,0:3] = h[0:3,0:3]
    mat[3:6,0:3] = SO3.hat(h[0:3,3])@h[0:3,0:3]
    mat[3:6,3:6] = h[0:3,0:3]

    return mat
  
  @staticmethod
  def adj_integ_mat(self, vec, a):
    pass
  
class SE3wrench(SE3):
  
  @staticmethod
  def hat(vec):
    mat = zeros((6,6))
    mat[0:3,0:3] = SO3.hat(vec[0:3])
    mat[3:6,3:6] = SO3.hat(vec[0:3])
    mat[0:3,3:6] = SO3.hat(vec[3:6])

    return mat
  
  @staticmethod
  def hat_commute(vec):
    mat = zeros((6,6))
    mat[0:3,0:3] = SO3.hat(vec[0:3])
    mat[0:3,3:6] = SO3.hat(vec[3:6])
    mat[3:6,0:3] = SO3.hat(vec[3:6])

    return -mat
  
  @staticmethod
  def mat(vec, a):
    return SE3.mat(vec, -a).Transpose()
  
  @staticmethod
  def integ_mat(vec, a):
    return SE3.integ_mat(vec, -a).Transpose()