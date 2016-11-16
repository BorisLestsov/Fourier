import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Fourier:
    def __init__(self, f, x1, x2, n, maxk):
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.L = (self.x2 - self.x1)/2
        self.n = n
        self.f = f
        self.maxk = maxk

        self.x = np.linspace(x1, x2, n, endpoint=True)
        self._cx = np.linspace(-self.L, self.L, n, endpoint=True)
        self.vec_f = np.vectorize(f)

    def run(self):
        def _init_f():
            pass

        def _graph_animate(t):
            t=t+1
            an = 1.0/self.L * simps(self.vec_f(self.x) * np.cos((t * np.math.pi * self.x) / self.L), self.x)
            bn = 1.0/self.L * simps(self.vec_f(self.x) * np.sin((t * np.math.pi * self.x) / self.L), self.x)
            self.sum += an*np.cos((t * np.math.pi * self.x) / self.L) + bn*np.sin((t * np.math.pi * self.x) / self.L)
            line.set_ydata(self.sum)
            return line,

        a0 = 1.0 / self.L * simps(self.vec_f(self.x), self.x)
        self.sum = np.full(len(self.x), a0 / 2.0)
        fig, ax = plt.subplots()
        ax.plot(self.x, self.vec_f(self.x), c='r')
        line, = ax.plot(self.x, self.sum, c='b')
        ani = animation.FuncAnimation(fig, _graph_animate, frames=self.maxk, interval=500,
                                      repeat=False, init_func=_init_f)
        plt.xlim(self.x1-1, self.x2+1)
        plt.ylim(np.min(self.vec_f(self.x)) - 1, np.max(self.f(self.x)) + 1)
        plt.show()


def f(x):
    return np.sin(x**2)


def main():
    x1 = 1
    x2 = 10
    n = 100
    maxk = 100

    a = Fourier(f, x1, x2, n, maxk)
    a.run()





if __name__ == "__main__":
    main()