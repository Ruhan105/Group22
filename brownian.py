import numpy as np
import matplotlib.pyplot as plt

from sympy import symbols, sympify
from sympy.core.sympify import SympifyError


def geometric_brownian(a,b,N):
    dt = 1.0 / N
    t = np.linspace(0,1.0,N+1)
    dW = np.random.normal(0,np.sqrt(dt),size=N)
    W = np.cumsum(dW)
    X = np.zeros(N+1)
    X[0] = 1.0
    for i in range(N):
        X[i+1] = X[i] + a*X[i]*dt + b*X[i]*dW[i]
    plt.plot(t, X)
    plt.title("Geometric Brownian Motion")
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.grid(True)
    plt.show()


const_a = input("Enter constant of a: ") 
const_b = input("Enter constant of b: ")
try:
    a = sympify(const_a)
except SympifyError:
    print("invalid function")
try:
    b = sympify(const_b)
except SympifyError:
    print("invalid function")
geometric_brownian(a, b, 1000)
