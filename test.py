from plotting import mean_and_std
import numpy as np

m, s = mean_and_std([np.array([3.0, 4.0]), np.array([5.0, 6.0]), np.array([10,10])])

print(m)
print(s)