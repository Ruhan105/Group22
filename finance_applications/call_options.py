# A Closed-Form Solution for
# Options with Stochastic
# Volatility with Applications
# to Bond and Currency
# Options
# Steven L. Heston
import numpy as np
import matplotlib.pyplot as plt


dt = 2**-8
N = int(1/dt)
k = 2
theta = 0.01
p = 0
sigma = 0.1
r = 0 
K = 100

dW = [np.random.multivariate_normal(mean=[0,0], cov=[[dt, dt*p], [dt*p, dt]]) for _ in range(N-1)]

## now just apply euler approximation as before but with vector components

S_approx = [100]
v_approx = [0.01]

for i in dW:
    Sn = S_approx[-1]
    vn = v_approx[-1]

    Sn = Sn + np.sqrt(vn * Sn)*i[0]
    vn = vn + k*(theta - vn)*dt + sigma*np.sqrt(vn)*i[1]

    S_approx.append(Sn)
    v_approx.append(vn)

t = np.linspace(0, 1, N)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot3D(S_approx, v_approx, t, label='/')

plt.show()