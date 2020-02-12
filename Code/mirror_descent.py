import numpy as np

'''
subgradient: oracle returning a subgradient at a given point
projection: oracle returning the euclidean projection of a given point
to the feasible regien
num_steps: Number iterations to perform
start: starting point, assumed to lie in the feasible region
R: upper bound on radius of feasible region
'''
def mirror_descent(subgradient, projection, num_steps, start, R, L):
    x = start
    step_length = R/(L*np.sqrt(num_steps))
    x_sum = x
    for i in range(num_steps):
        y = x - step_length * subgradient(x)
        x = projection(y)
        x_sum += y
    result = x_sum/num_steps
    return result
