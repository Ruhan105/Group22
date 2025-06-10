# Euler schemes and large deviations for stochastic
# Volterra equations with singular kernels 

# FIRST-ORDER EULER SCHEME FOR SDES DRIVEN BY
# FRACTIONAL BROWNIAN MOTIONS: THE ROUGH CASE

# Efﬁcient option pricing in the rough Heston model
# using weak simulation schemes (2024 paper)

## start with naive Euler scheme then apply 2024 paper and compare results.



import numpy as np
import matplotlib.pyplot as plt
from fractional_brownian import FractionalBrownian


dt = 2**-10
N = int(1/dt)
k = 2
theta = 0.01
p = 0
sigma = 0.1
r = 0
strike_price = 100

H = 0.2

# fig, ax = plt.subplots()

# for _ in range(5):

#     fBm = FractionalBrownian(N, H)

#     dW = np.diff(fBm.generate_sample())

#     dB = np.diff(fBm.generate_sample())
#     S_approx = [100]
#     v_approx = [0.01]

#     for i in range(len(dW)):
#         Sn = S_approx[-1]
#         vn = v_approx[-1]

#         # Using default values as defined in the paper

#         Sn = Sn + np.sqrt((max(0, vn)))*Sn*dW[i]   # set drift coefficient to 0 for testing purposes
#         vn = vn + k*(theta - vn)*dt + sigma*vn*dB[i]

#         S_approx.append(Sn)
#         v_approx.append(vn)



#     t = np.linspace(0, 1, N)


#     plt.plot(t, S_approx, label="Options Price over time")
#     plt.xlabel("Time")
#     plt.ylabel("Stock Price (Euros)")
#     plt.title("Heston Model")


# plt.show()

## Weak simulation scheme using lifted heston  ( as described in 2022 paper)



fig, ax = plt.subplots()

def K(t):
    return t**(H - 1/2)

for _ in range(5):

    dW = [np.random.multivariate_normal(mean=[0, 0], cov=[[dt, 0], [0, dt]]) for _ in range(N-1)]

    Y_approx = [np.log(strike_price)]  # S^n = exp(Y^n)
    v_approx = [0.01]

    for dw in dW:

        vn = 0.01
        Yn = Y_approx[-1]
        

        m = len(Y_approx)

        for i in range(m):
            v_i = v_approx[i]
            print(v_i)
            vn += K(dt*m - dt*i)*((theta - k*max(v_i, 0))*dt + sigma*np.sqrt(max(v_i, 0))*dW[i][0])

        Yn = Yn  -(1/2)*(max(vn, 0)*dt) + p*np.sqrt(max(vn, 0))*dw[0] + np.sqrt(1-p**2)*np.sqrt(max(vn, 0))*dw[1]
        
        Y_approx.append(Yn)
        v_approx.append(vn)

    t = np.linspace(0, 1, N)

    S_approx = [np.exp(i) for i in Y_approx]

    plt.plot(t, S_approx, label="Options Price over time")
    plt.xlabel("Time")
    plt.ylabel("Stock Price (Euros)")
    plt.title("Rough Heston Model (2022 paper)")

plt.show()

