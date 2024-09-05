def gaussian_density(x, mu, sigma):
    result = math.exp(-0.5 * ( (x-mu)/sigma )** 2)
    result /= sigma * (2 * math.pi) ** 0.5
    return result


class RightTailGaussianFunc:
    def __init__(self, capacity, at_capacity_value, sd):
        self.capacity = capacity
        self.at_capacity_value = at_capacity_value
        self.sd = sd

    def __call__(self, x):
        if x <= capacity:
            return x