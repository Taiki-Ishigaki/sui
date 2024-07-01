import numpy as np
from sui.basic import *

class LieAbstract:

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
    pass
  
  @staticmethod
  def hat_commute(vec):
    '''
    hat commute operator on the tanget space vector
    hat(a) @ b = hat_commute(b) @ a 
    '''
    pass

  @staticmethod
  def vee(vec_hat):
    '''
    a = vee(hat(a))
    '''
    pass  
  
  @staticmethod
  def mat(vec, a):
    pass

  @staticmethod
  def integ_mat(vec, a):
    pass
  
  def inverse(self):
    pass
  
  def adjoint(self):
    '''
    adjoint expresion of Lie group
    '''
    pass
  
  def adj_hat(self, vec):
    pass
  
  def adj_hat_commute(self, vec):
    pass
  
  def adj_mat(self, vec, a):
    pass
  
  def adj_integ_mat(self, vec, a):
    pass