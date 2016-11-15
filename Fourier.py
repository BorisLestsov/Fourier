import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Fourier:
    def __init__(self, f, x1, x2, n, maxk):
        self.x1 = x1
        self.x2 = x2
        self.n = n
        self.f = f
        self.maxk = maxk

        self.x = np.linspace(x1, x2, n, endpoint=True)
        self._cx = np.linspace(-np.math.pi, np.math.pi, n, endpoint=True)
        self.vec_f = np.vectorize(f)

    def run(self):
        def _graph_animate(t):
            an = 1.0/np.math.pi * simps(self._cf(self.x) * np.cos(t*self._cx), self._cx)
            bn = 1.0/np.math.pi * simps(self._cf(self.x) * np.sin(t*self._cx), self._cx)
            self.sum += an * np.cos(t * self.x) + bn* np.sin(t * self.x)

            line.set_ydata(self.sum)
            return line,

        a0 = 1.0/np.math.pi * simps(self._cf(self.x), self._cx)
        self.sum = np.full(len(self.x), a0/2)

        fig, ax = plt.subplots()
        line, = ax.plot(self.x, self.vec_f(self.x)*np.cos(self.x), c='b')
        ax.plot(self.x, self.f(self.x), c='r')
        ani = animation.FuncAnimation(fig, _graph_animate, frames=self.maxk, interval=500,
                                      repeat=False)
        plt.xlim(self.x1, self.x2)
        plt.ylim(self.f(self.x1))
        plt.show()

    def _cf(self, x):
        return self.vec_f(np.math.pi*(2*(x - self.x1)/(self.x2 - self.x1) - 1))
        #return self.vec_f((self.x2 - self.x1)/(2*np.math.pi) * (x + np.math.pi) + self.x1)

def f(x):
    return x


def main():
    x1 = 1
    x2 = 5
    n = 100
    maxk = 100

    a = Fourier(f, x1, x2, n, maxk)
    a.run()





if __name__ == "__main__":
    main()