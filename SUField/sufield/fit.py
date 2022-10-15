import copy
import os
from random import uniform
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.special as sp
from IPython import embed


class PDFunction:

    def __init__(self, *args) -> None:
        self.init_params = args
        self.params = [*args]

    def update(self, *args):
        self.params = [*args]

    def max(self):
        raise NotImplementedError()

    def __call__(self, t):
        raise NotImplementedError

    def em_step(self, arr, prob):
        raise NotImplementedError


class GammaDistribution(PDFunction):

    def __call__(self, t):
        a, b = self.params
        return b**a / (sp.gamma(a)) * np.e**(-b * t) * t**(a - 1)

    def max(self):
        return (self.params[0] - 1) / (self.params[1])

    def em_step(self, arr, prob):
        target = np.log((prob * arr).sum() / prob.sum()) - (prob * np.log(arr)).sum() / prob.sum()
        coef = prob.sum() / np.maximum((prob * arr).sum(), 1e-8)
        func = lambda x: np.log(x + 1e-5) - sp.digamma(x + 1e-5) - target
        jac = lambda x: 1 / x - sp.gamma(x)
        root = opt.root(func, self.params[0], jac=jac)
        self.update(root.x[0], root.x[0] * coef)


class BetaDistribution(PDFunction):

    def __call__(self, t):
        a, b = self.params
        return sp.gamma(a + b) / sp.gamma(a) / sp.gamma(b) * t**(a - 1) * (1 - t)**(b - 1)

    def max(self):
        return (self.params[0] - 1) / (self.params[0] + self.params[1] - 2)

    def em_step(self, arr, prob):
        target_a = -(prob * np.log(arr)).sum() / prob.sum()
        target_b = -(prob * np.log(1 - arr)).sum() / prob.sum()

        def func(x):
            polys = sp.polygamma(0, [x[0] + x[1], x[0], x[1]])
            return [polys[0] - polys[1] - target_a, polys[0] - polys[2] - target_b]

        def jac(x):
            polys = sp.polygamma(1, [x[0] + x[1], x[0], x[1]])
            return [[polys[0] - polys[1], polys[0]], [polys[0], polys[0] - polys[2]]]

        root = opt.root(func, self.params, jac=jac)
        self.update(root.x[0], root.x[1])


def visualize_callable(func: Callable, boundary: Tuple[float, float], nstep=1000, color='green'):
    low, high = boundary
    x = np.arange(nstep) / nstep * (high - low) + low
    y = func(x)
    plt.plot(x, y, color=color, alpha=0.75)


def distribution_error(func: Callable, data_arr: np.ndarray, steps=50000):
    y = np.histogram(data_arr, bins=steps, density=True)[0]
    x = np.arange(steps) / steps * (data_arr.max() - data_arr.min()) + data_arr.min()
    z = func(x)
    return np.abs(y - z).mean()


class FitRunner:

    def __init__(self, arr: np.ndarray, dist_a: PDFunction, dist_b: PDFunction, init_weight=0.5) -> None:
        self.data_arr = arr
        self.weight = init_weight
        self.dist_a: PDFunction = dist_a
        self.dist_b: PDFunction = dist_b

        self.best_err = float('inf')
        self.opt_params_a = copy.deepcopy(dist_a.params)
        self.opt_params_b = copy.deepcopy(dist_b.params)
        self.opt_weight = init_weight

    def fit(self, step=30, visualize=False, quiet=False, save=None, opt=True):
        for i in range(step):
            calc = lambda x: self.weight * self.dist_a(x) + (1 - self.weight) * self.dist_b(x)
            if not quiet:
                print(f"Step #{i}")
                print(self)
                print(f"Error: {distribution_error(calc, self.data_arr)}")
            if visualize:
                self.visualize(save)
            pdf_a = self.dist_a(self.data_arr)
            pdf_b = self.dist_b(self.data_arr)
            pdf_sum = self.weight * pdf_a + (1 - self.weight) * pdf_b
            prob_a = self.weight * pdf_a / pdf_sum
            prob_b = (1 - self.weight) * pdf_b / pdf_sum
            self.weight = prob_a.sum() / len(prob_a)
            self.dist_a.em_step(self.data_arr, prob_a)
            self.dist_b.em_step(self.data_arr, prob_b)
            error = self.error()
            if error < self.best_err:
                self.best_err = error
                self.opt_params_a = copy.deepcopy(self.dist_a.params)
                self.opt_params_b = copy.deepcopy(self.dist_b.params)
                self.opt_weight = self.weight
        if opt:
            self.dist_a.update(*self.opt_params_a)
            self.dist_b.update(*self.opt_params_b)
            self.weight = self.opt_weight
        return self

    def error(self, steps=50000):
        y = np.histogram(self.data_arr, bins=steps, density=True)[0]
        x = np.arange(steps) / steps * (self.data_arr.max() - self.data_arr.min()) + self.data_arr.min()
        z = self.dist_a(x) * self.weight + self.dist_b(x) * (1 - self.weight)
        return np.abs(y - z).mean()

    def visualize(self, save=None, title="Fitting Results"):
        data_arr = self.data_arr
        plt.title(title)
        plt.hist(data_arr, color='g', bins=500, alpha=0.5, density=True)
        calc = lambda x: (self.weight * self.dist_a(x) + (1 - self.weight) * self.dist_b(x))
        visualize_callable(calc, (data_arr.min(), data_arr.max()))
        visualize_callable(lambda x: self.weight * self.dist_a(x), (data_arr.min(), data_arr.max()), color='red')
        visualize_callable(lambda x: (1 - self.weight) * self.dist_b(x), (data_arr.min(), data_arr.max()), color='blue')
        if save is None:
            plt.show()
            plt.cla()
        else:
            try:
                os.remove(save)
            except:
                pass
            print("Saving...")
            plt.savefig(save)
            plt.cla()

    def judge_compare(self, arr):
        """
        Judge by comparing the values of PDFs
        """
        return self.weight * self.dist_a(arr) > (1 - self.weight) * self.dist_b(arr)

    def judge_intersect(self, arr, init=0.01):
        """
        Judge by finding the first intersection of the two PDFs and using it as threshold
        """
        root = opt.root(lambda x: self.weight * self.dist_a(x) - (1 - self.weight) * self.dist_b(x), init).x[0]
        # root = opt.newton(lambda x: self.weight * self.dist_a(x) - (1 - self.weight) * self.dist_b(x), init)
        return arr < root

    def __str__(self) -> str:
        return (f'Distribution 1 params: {self.dist_a.params}\n') + (
            f'Distribution 2 params: {self.dist_b.params}\n') + (f'Weight: {self.weight}')


def mixture_filter(
    arr,
    dist_a,
    dist_b,
    weight=0.5,
    step=10,
    save=None,
    quiet=True,
    visualize=False,
    bins=500,
):
    arr = np.abs(arr)

    runner = FitRunner(arr, dist_a, dist_b, init_weight=weight)
    if visualize:
        runner.visualize(title='Data and Initial distributions')
    runner.fit(step=step, quiet=quiet)
    if visualize:
        runner.visualize()
        print(runner)
        print(f'Final fitting error: {runner.error()}')
    if save is not None:
        runner.visualize(save=save)

    init_a = runner.dist_a.max()
    init_b = runner.dist_b.max()
    keep_mask_compare = runner.judge_compare(arr)
    keep_mask_intersect = runner.judge_intersect(arr, (init_a + init_b) / 2)
    if visualize:
        print('Initial intersection: ', (init_a + init_b) / 2)
        plt.title('Initial Distribution')
        plt.hist(arr, range=(arr.min(), arr.max()), bins=bins, alpha=0.5, density=False, stacked=True, color='red')
        plt.show()

        plt.title('Filtered Distribution (Comparing)')
        plt.hist(arr[keep_mask_compare],
                 range=(arr.min(), arr.max()),
                 bins=bins,
                 alpha=0.5,
                 density=False,
                 stacked=True,
                 color='green')
        plt.show()

        plt.title('Filtered Distribution (Intersection)')
        plt.hist(arr[keep_mask_intersect],
                 range=(arr.min(), arr.max()),
                 bins=bins,
                 alpha=0.5,
                 density=False,
                 stacked=True,
                 color='blue')
        plt.show()

    return keep_mask_intersect


if __name__ == '__main__':
    # Random data generation
    # dist_cls = GammaDistribution
    # func = lambda x, y: np.random.gamma(x, 1 / y)
    dist_cls = BetaDistribution
    func = np.random.beta

    a1, b1 = 2, 10
    a2, b2 = 4, 2
    weight = 0.4
    arr = np.array([(func(a1, b1) if np.random.uniform(0, 1) < weight else func(a2, b2)) for _ in range(50000)])

    # Initial params
    a1, b1 = 2, 3
    a2, b2 = 3, 2
    weight = 0.5
    dist_a = dist_cls(a1, b1)
    dist_b = dist_cls(a2, b2)

    mask = mixture_filter(arr, dist_a, dist_b, weight=weight, visualize=True, step=1000)
