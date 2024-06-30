from sui.basic import *
from sui.geo import *

def vec_to_rot_mat(vec, a):
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

def vec_to_integ_rot_mat(vec, a):
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

def vec_to_hom_mat(vec, t):
  """
    同次変換行列の計算
    sympyの場合,vec[0:2]の大きさは1を想定
  """
  if LIBRARY == 'numpy':
    rot = vec[0:3]
    pos = vec[3:6]
  elif LIBRARY == 'sympy':
    rot = xp.Matrix(vec[0:3])
    pos = xp.Matrix(vec[3:6])
  else:
    raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

  mat = zeros(4,4)
  mat[0:3,0:3] = vec_to_rot_mat(rot, t)
  V = vec_to_integ_rot_mat(rot, t)
  print(pos)
  mat[0:3,3] = V @ pos
  mat[3,3] = 1

  return mat

def vec_to_spa_mat(vec, t):
  """
    空間変換行列の計算
    sympyの場合,vec[0:2]の大きさは1を想定
  """

  h = vec_to_hom_mat(vec, t)

  mat = zeros(6,6)
  mat[0:3,0:3] = h[0:3,0:3]
  mat[3:6,0:3] = cross3(h[0:3,3])@h[0:3,0:3]
  mat[3:6,3:6] = h[0:3,0:3]

  return mat