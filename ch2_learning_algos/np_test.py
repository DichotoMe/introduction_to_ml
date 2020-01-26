import numpy as np

arr = np.array([[0, 1, 3],
                [3, 5, 0],
                [2, 1, 1]])

ind = np.argsort(arr)

print(np.take_along_axis(arr, ind, axis=1))