# Excersise 9.2.1
a = 1.5
b = 1.0
X0 = 1.0


def actualSolution():
    """Generates a sample path using the exact solution of the SDE in 9.2.1"""
    n = 2**9

    dt = 1/n

    increments = [np.random.normal(loc=0, scale=np.sqrt(dt)) for _ in range(n)]
    total_increments = [increments[0]]
    for i in increments[1:]:
        total_increments.append(total_increments[-1] + i)

    points = []

    for t, w in enumerate(total_increments):
        points.append(1*np.exp((a - (b/2)**2)*(t/n) + b*w))
    
    return points


dt = 2**-2
N = 4

X_approx = [1.0]

# Euler approximation of SDE
for _ in range(N-1):
    dW = np.random.normal(loc=0, scale=dt)
    Yn = X_approx[-1]
    X_approx.append(Yn + a*Yn*dt + b*Yn*dW)


fig, ax = plt.subplots()
ax.plot(np.linspace(start=0, stop=1, num=2**9), actualSolution())

plt.plot(np.linspace(0,1,N), X_approx)

plt.show()
