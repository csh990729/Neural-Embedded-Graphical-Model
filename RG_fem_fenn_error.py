import numpy as np

# arr is a 1D array with length 1002001
arr0 = np.loadtxt("fem.txt")

# Reshape to 1001 x 1001
arr2d = arr0.reshape(1001, 1001)

# Uniformly downsample to 11 x 11
arr11 = arr2d[::100, ::100]

print(arr11.shape)
# (11, 11)

arrx = np.loadtxt("fenn.txt")
arrx11 = arrx.reshape(11, 11)

print(np.mean(np.abs(arr11-arrx11)/arr11))

np.savetxt("error_fem_fenn.txt", np.abs(arr11-arrx11))
