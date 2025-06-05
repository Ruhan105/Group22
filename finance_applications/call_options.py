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
k = 2 # mean reversion
theta = 0.01 # long run variance
p = 0 # correlation between weiner process
sigma = 0.1 # volatility of volatility
r = 0 #interest rate
K = 100 # strike price

fig, ax = plt.subplots()

for _ in range(5):
    dW = [np.random.multivariate_normal(mean=[0,0], cov=[[dt, dt*p], [dt*p, dt]]) for _ in range(N-1)]


    S_approx = [100]
    v_approx = [0.01]

    for i in dW:
        Sn = S_approx[-1]
        vn = v_approx[-1]

        # Using default values as defined in the paper

        Sn = Sn + np.sqrt(vn * Sn)*i[0]   # set drift coefficient to 0 for testing purposes
        vn = vn + k*(theta - vn)*dt + sigma*np.sqrt(vn)*i[1]

        S_approx.append(Sn)
        v_approx.append(vn)

    t = np.linspace(0, 1, N)


    plt.plot(t, v_approx, label="Volatility over time")
    plt.xlabel("Time")
    plt.ylabel("")
    plt.title("Heston Model")


fig, ax = plt.subplots()

for _ in range(5):
    dW = [np.random.multivariate_normal(mean=[0,0], cov=[[dt, dt*p], [dt*p, dt]]) for _ in range(N-1)]


    S_approx = [100]
    v_approx = [0.01]

    for i in dW:
        Sn = S_approx[-1]
        vn = v_approx[-1]

        # Using default values as defined in the paper

        Sn = Sn + np.sqrt(vn)*Sn*i[0]   # set drift coefficient to 0 for testing purposes
        vn = vn + k*(theta - vn)*dt + sigma*vn*i[1]

        S_approx.append(Sn)
        v_approx.append(vn)

    t = np.linspace(0, 1, N)


    plt.plot(t, S_approx, label="Stock Price over time")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("GARCH diffusion model")

plt.show()

#TODO- look at Runge Kutta scheme and Milstein model + compare?
#TODO- Analyse the effectiveness of the Chen model to model real world interest rates?
## apply MLE to estimate parameters 
# look at merton jump diffusion model
# fractional brownian motion (s,t) |-> 1/2( abs(s)^2H + abs(t)^2H + abs(t-s))