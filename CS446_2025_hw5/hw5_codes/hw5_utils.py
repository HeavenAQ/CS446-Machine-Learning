from typing import Callable, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.types import Tensor
import torch.utils.data

_EPS = 1e-10

"""
    Produces a contour plot for the prediction function.

    Arguments:
        pred_fxn: Prediction function that takes an n x d tensor of test examples
        and returns your SVM's predictions.
        xmin: Minimum x-value to plot.
        xmax: Maximum x-value to plot.
        ymin: Minimum y-value to plot.
        ymax: Maximum y-value to plot.
        ngrid: Number of points to be plotted between max and min (granularity).
"""

Kerf = Callable[[Tensor, Tensor], Tensor]


def svm_contour(pred_fxn, xmin=-8, xmax=8, ymin=-8, ymax=8, ngrid=33):
    with torch.no_grad():
        xgrid = torch.linspace(xmin, xmax, ngrid)
        ygrid = torch.linspace(ymin, ymax, ngrid)
        (xx, yy) = torch.meshgrid((xgrid, ygrid), indexing="ij")
        x_test = torch.cat(
            (xx.view(ngrid, ngrid, 1), yy.view(ngrid, ngrid, 1)), dim=2
        ).view(-1, 2)
        zz = pred_fxn(x_test)
        zz = zz.view(ngrid, ngrid)
        cs = plt.contour(
            xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(), cmap="coolwarm"
        )
        plt.clabel(cs)
        plt.show()


def poly_implementation(x, y, degree):
    assert x.size() == y.size()
    with torch.no_grad():
        return (1 + (x * y).sum()).pow(degree)


def poly(degree):
    return lambda x, y: poly_implementation(x, y, degree)


def rbf_implementation(x, y, sigma):
    assert x.size() == y.size()
    with torch.no_grad():
        return (-(x - y).norm().pow(2) / 2 / sigma / sigma).exp()


def rbf(sigma):
    return lambda x, y: rbf_implementation(x, y, sigma)


def xor_data() -> Tuple[Tensor, Tensor]:
    x = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float)
    y = torch.tensor([1, -1, 1, -1], dtype=torch.float)
    return x, y


def create_kernel(X1: Tensor, X2: Tensor, kernel: Kerf) -> Tensor:
    r, _ = X1.shape
    c, _ = X2.shape
    K = torch.empty((r, c))
    for i in range(r):
        for j in range(c):
            K[i, j] = kernel(X1[i], X2[j])
    return K


def svm_fit(
    X: Tensor, y: Tensor, lr: float, num_iters: int, kernel: Kerf, c: float | None
):
    K = create_kernel(X, X, kernel)
    Y = torch.diag(y)
    alpha = torch.zeros_like(y, requires_grad=True)
    for _ in range(num_iters):
        f = alpha.sum() - 0.5 * alpha @ Y @ K @ Y @ alpha
        f.backward()
        with torch.no_grad():
            alpha += lr * alpha.grad
            if c:
                alpha = torch.clamp_(alpha, min=0, max=c)
            else:
                alpha = torch.clamp_(alpha, min=0)
            alpha.grad.zero_()

    alpha = alpha.detach()

    support_idx = torch.where(alpha > _EPS)[0]
    if len(support_idx) == 0:
        return alpha, 0.0

    min_alpha_idx = support_idx[torch.argmin(alpha[support_idx])]
    b = y[min_alpha_idx] - (alpha * y * K[:, min_alpha_idx]).sum()
    return alpha, b


def svm_predict(kernel=poly(degree=1)):
    X_train, y_train = xor_data()
    alpha, b = svm_fit(X_train, y_train, 0.05, 10000, kernel, None)

    def predict(X_test: Tensor):
        K_test = create_kernel(X_train, X_test, kernel)
        f = torch.sum((alpha * y_train).unsqueeze(1) * K_test, dim=0) + b
        return f

    return predict


def main():
    poly_deg = 3
    svm_contour(svm_predict(poly(poly_deg)))
    rbf_sigmas = [1, 2, 5]
    for sigma in rbf_sigmas:
        svm_contour(svm_predict(rbf(sigma)))


if __name__ == "__main__":
    main()
