import numpy as np
import matplotlib.pyplot as plt

N = 1000
dt = 1/N

steps = [np.random.normal(loc=0, scale=np.sqrt(dt)) for _ in range(N)]

path = [0] + [0 for _ in steps]

for i in range(1,len(steps)+1):
    path[i] = steps[i-1] + path[i-1]
    
fig, ax = plt.subplots()
ax.plot(np.linspace(0, 1, N+1), path)
plt.show()
