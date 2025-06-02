import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def simulateOrnstein(b, s, X0, p):
    """ Simulates 2-D Ornstein"""
    n = 2**10
    dt = 1/n
    dW_fine = np.random.multivariate_normal(mean=[0, 0],
                                            cov=[[dt, dt*p],
                                                 [dt*p, dt]], size=n)
    
    W_fine = np.cumsum(dW_fine, axis=0) 
    sample_points = []
    X0_col = np.array(X0).reshape(-1, 1) 

    for t in range(n): 
        current_time = (t + 1) * dt 
        integral_approx = np.zeros((2, 1)) 

        for k in range(t + 1): 
            dW_k_col = dW_fine[k].reshape(-1, 1) 
            integral_approx += expm(-b * (current_time - (k)*dt)) @ s @ dW_k_col

        sample_points.append((expm(-b * current_time) @ X0_col + integral_approx ).flatten()) 

    return np.array(sample_points), W_fine 

b = np.array([[1, 0], [0, 1]])
s = np.array([[1, 0], [0, 1]])

t_fine_plot = np.linspace(0, 1, 2**10) 

fig = plt.figure()


dt = 2**-6
N = int(1/dt)

sample_points, W_fine = simulateOrnstein(b, s, [1, 1], 1)

W_coarse_points = W_fine[::int(dt*2**10)]

W_coarse = np.array(W_coarse_points)
dW_coarse = np.diff(W_coarse, axis=0)


X_approx = [np.array([1, 1])]

for dW in dW_coarse:
    Yn = X_approx[-1]
    Yn_col = Yn.reshape(-1, 1) 
    dW_col = dW.reshape(-1, 1) 
    X_approx.append((Yn_col - b@Yn_col*dt + s@dW_col).flatten()) 

x_approx = [i[0] for i in X_approx]
y_approx = [i[1] for i in X_approx]

x = [i[0] for i in sample_points]
y = [i[1] for i in sample_points]


ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x, y, t_fine_plot, label='Exact Solution') 

ax.plot3D(x_approx, y_approx, np.linspace(0, 1, N), label="Euler Approximation") 

plt.show()