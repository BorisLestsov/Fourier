import numpy as np
from scipy.integrate import simps
from scipy.misc import derivative
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Approx:
    def __init__(self, f, x1, x2, n, maxk, a):
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.L = (self.x2 - self.x1)/2
        self.n = n
        self.f = f
        self.maxk = maxk
        self.a = a

        self.x = np.linspace(x1, x2, n, endpoint=True)
        self.vec_f = np.vectorize(f)

    def run(self, type='both'):
        def _init_f():
            pass

        def _graph_animate(t):
            t += 1

            if type == 'both' or type == 'fourier':
                an = 1.0/self.L * simps(self.vec_f(self.x) * np.cos((t * np.math.pi * self.x) / self.L), self.x)
                bn = 1.0/self.L * simps(self.vec_f(self.x) * np.sin((t * np.math.pi * self.x) / self.L), self.x)
                self.sum_f += an * np.cos((t * np.math.pi * self.x) / self.L) + bn * np.sin((t * np.math.pi * self.x) / self.L)
                line.set_ydata(self.sum_f)

            if type == 'both' or type == 'taylor':
                der = derivative(self.vec_f, self.a, dx=0.1, n=t, order=2*t+1)
                #der = fs(self.a)
                self.sum_t += der / np.math.factorial(t) * (self.x - self.a) ** t
                line1.set_ydata(self.sum_t)

        if type != 'both' and type != 'taylor' and type != 'fourier':
            raise Exception("Unknown type of graph")

        if type == 'both' or type == 'fourier':
            a0 = 1.0 / self.L * simps(self.vec_f(self.x), self.x)
            self.sum_f = np.full(len(self.x), a0 / 2.0)

        if type == 'both' or type == 'taylor':
            a01 = self.vec_f(self.a)
            self.sum_t = np.full(len(self.x), a01)

        fig, ax = plt.subplots()
        ax.plot(self.x, self.vec_f(self.x), c='r', linewidth=2)

        if type == 'both' or type == 'fourier':
            ax.plot(self.x1, self.vec_f(self.x1), 'o', c='b')
            ax.plot(self.x2, self.vec_f(self.x2), 'o', c='b')
            line, = ax.plot(self.x, self.sum_f, c='b')

        if type == 'both' or type == 'taylor':
            ax.plot(self.a, self.vec_f(self.a), 'o', c='g')
            line1, = ax.plot(self.x, self.sum_t, c='g')
        ani = animation.FuncAnimation(fig, _graph_animate, frames=self.maxk, interval=500,
                                      repeat=False, init_func=_init_f)
        plt.xlim(self.x1-1, self.x2+1)
        plt.ylim(np.min(self.vec_f(self.x)) - 1, np.max(self.vec_f(self.x)) + 1)
        plt.show()


def f(x):
    return np.math.sin(x)


def main():
    x1 = -10
    x2 = 10
    n = 100
    maxk = 100
    a = 0

    a = Approx(f, x1, x2, n, maxk, a)
    a.run('both')





if __name__ == "__main__":
    main()