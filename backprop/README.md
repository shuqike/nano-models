# Back-propagation from scratch

## Implementation notes for data flows in neural architecture

For `NumPy` arrays, always add the additional batch dimension even if you are doing singleton batch. Otherwise you get into trouble when doing outer products. For example,
```Python
>>> import numpy as np
>>> x1 = np.array([1,2,3])
>>> x2 = np.ones(3)
>>> x1 @ x2.T
6.0
>>> x3 = np.array([[1,2,3]])
>>> x3.shape
(1, 3)
>>> x4 = np.ones((1,3))
>>> x4.shape
(1, 3)
>>> x3.T @ x4
array([[1., 1., 1.],
       [2., 2., 2.],
       [3., 3., 3.]])
>>> x1.T @ x2
6.0
```
