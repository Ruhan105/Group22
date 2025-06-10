import numpy as np
import matplotlib.pyplot as plt


"""Generate fractional brownian motion sample paths """


class FractionalBrownian:
    """
        class for generating fBm sample paths.

        params:
            N - sample path resolution
            H - Hurst parameter
    """
    def __init__(self, N=2**10, H=1/2):
        self.N = N
        self.H = H
        self.__dt = 1/N

    
    def autocov(self, k):
        return (1/2)*(abs(k+1)**(2*self.H) + abs(k-1)**(2*self.H) - 2*abs(k)**(2*self.H)) 

    @property
    def covariance(self):
        return np.array([[self.autocov((i + j) % self.N) for i in range(self.N)] for j in range(self.N)])


    def generate_sample(self):

        samples = np.random.multivariate_normal(mean=[0 for i in range(self.N)], cov=self.covariance)

        return np.cumsum(samples*(self.__dt ** self.H))

    def plot_samples(self, n, t0, tn):
       
        fig, ax = plt.subplots()

        t = np.linspace(t0, tn, self.N)

        for _ in range(n):
            samples = self.generate_sample()

            plt.plot(t, samples)
        

        plt.xlabel("Time")
        plt.title("Fractional Brownian motion")

        plt.show()