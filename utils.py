import math

import matplotlib.pyplot as plt
import numpy as np

def gaussian_density(x, mu, sigma):
    result = math.exp(-0.5 * ( (x-mu)/sigma )** 2)
    result /= sigma * (2 * math.pi) ** 0.5
    return result


class RightTailGaussianFunc:
    def __init__(self, capacity, in_capacity_value, sd):
        self.capacity = capacity
        self.in_capacity_value = in_capacity_value
        self.sd = sd

    def __call__(self, x):
        if x <= self.capacity:
            return self.in_capacity_value
        else:
            return gaussian_density(x, self.capacity, self.sd)

if __name__ == "__main__":
    #visualise value functions
    import matplotlib.pyplot as plt
    COLOURS = ["blue", "red", "green"]
    NAMES = ["Car", "Bus", "Walk"]

    value_functions = [RightTailGaussianFunc(0.0, 1, 0.4), RightTailGaussianFunc(0.4, 1.14, 0.35), lambda x: 1]
    for i,f in enumerate(value_functions):
        x = np.linspace(0,1,100)
        y = [f(j) for j in x]
        plt.plot(x, y, color=COLOURS[i], label=NAMES[i])
    plt.xlabel("ω: fraction of agents selecting the action")
    plt.ylabel("V(ω): value of the action")
    plt.legend()
    plt.savefig("value_functions.png")