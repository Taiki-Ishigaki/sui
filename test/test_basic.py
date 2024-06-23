import numpy as np
import sui.basic as sui

def test_cross3():
  v = np.array([1, 2, 3])
  m = np.array([[0, -3, 2],[3, 0, -1],[-2, 1, 0]])
  res = sui.cross3(v)

  np.testing.assert_array_equal(res, m)