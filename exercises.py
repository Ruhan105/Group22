import numpy as np
import matplotlib.pyplot as plt

# Excersise 9.2.1/2
a = 1.5
b = 1.0
X0 = 1.0
fine_resolution = 2**10
t0 = 0
tn = 1


def actualSolution(a, b, X0, n):
    """Generates a sample path using the exact solution of the SDE in 9.2.1"""

    n = 2**10

    dt = 1/n

    dW_fine = [np.random.normal(loc=0, scale=np.sqrt(dt)) for _ in range(n)]
    W_fine = np.cumsum(dW_fine)
    sample_points = []

    for t, w in enumerate(W_fine):
        sample_points.append(X0*np.exp((a - (b/2)**2)*(t/n) + b*w))

    return sample_points, W_fine

# Compute Euler Approximation

dt = 2**-6
N = int(1/dt)

points, W_fine = actualSolution(a, b, X0, fine_resolution)

# Using the same sample paths, skipping enough terms for fine case
W_coarse = W_fine[::int((dt*fine_resolution))]
dW_coarse = np.diff(W_coarse)

X_approx = [1.0]

for dW in dW_coarse:
    Yn = X_approx[-1]
    X_approx.append(Yn + a*Yn*dt + b*Yn*dW)


fig, ax = plt.subplots()
ax.plot(np.linspace(start=t0, stop=tn, num=fine_resolution), points)

plt.plot(np.linspace(t0, tn, N), X_approx)

plt.show()

# Exercise 9.3.1


n = 25

