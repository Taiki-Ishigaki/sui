from sui.basic import *

def cross3(vec):
  mat = zeros(3,3)
  mat[1,2] = -vec[0]
  mat[2,1] =  vec[0]
  mat[2,0] = -vec[1]
  mat[0,2] =  vec[1]
  mat[0,1] = -vec[2]
  mat[1,0] =  vec[2]

  return mat

def cross4(vec):
  mat = zeros(4,4)

  mat[0:3,0:3] = cross3(vec[0:3])
  mat[0:3,3] = vec[3:6]

  return mat

def cross6(vec):
  mat = zeros(6,6)

  mat[0:3,0:3] = cross3(vec[0:3])
  mat[3:6,3:6] = cross3(vec[0:3])
  mat[3:6,0:3] = cross3(vec[3:6])

  return mat

def cross6_(vec):
  return -cross6(vec)

def cross6dual(vec):
  mat = zeros(6,6)
  mat[0:3,0:3] = cross3(vec[0:3])
  mat[3:6,3:6] = cross3(vec[0:3])
  mat[0:3,3:6] = cross3(vec[3:6])

  return mat

def cross6dual_(vec):
  mat = zeros(6,6)
  mat[0:3,0:3] = cross3(vec[0:3])
  mat[0:3,3:6] = cross3(vec[3:6])
  mat[3:6,0:3] = cross3(vec[3:6])

  return -mat