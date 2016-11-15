import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Fourier:
    def __init__(self, f, x1, x2, n):
        self.x = np.linspace(x1, x2, n, endpoint=True)
        self.vec_f = np.vectorize(f)
        self.n = n

    def run(self):
        def _graph_animate(t):
            line.set_ydata(self.vec_f(self.x)*np.cos(t * self.x))
            return line,

        fig, ax = plt.subplots()
        line, = ax.plot(self.x, self.vec_f(self.x)*np.cos(self.x))
        ani = animation.FuncAnimation(fig, _graph_animate, frames=self.n, interval=1000,
                                      repeat=False)
        plt.show()


def f(x):
    return x**3


def main():
    x1 = 0
    x2 = 1
    n = 100

    a = Fourier(f, x1, x2, n)
    a.run()





if __name__ == "__main__":
    main()