import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


_EPS = 1e-5


def create_kernel(x1, x2, rsize: int, csize: int, kernel):
    try:
        k = kernel(x1, x2)
        if k.shape == (rsize, csize):
            return k
    except Exception:
        pass

    # fall back to 1D
    k = torch.zeros(rsize, csize)
    for i in range(rsize):
        for j in range(csize):
            k[i, j] = kernel(x1[i], x2[j])
    return k


def svm_solver(
    x_train, y_train, lr, num_iters, kernel=hw3_utils.poly(degree=1), c=None
):
    """
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    """
    # TODO
    N = x_train.shape[0]
    alpha = torch.zeros(N, requires_grad=True)
    Y = torch.diag(y_train)

    # Build the kernel beforehand
    K = create_kernel(x_train, x_train, N, N, kernel)
    for _ in range(num_iters):
        # The matrix form of the original dual problem: sum(alpha) - alpha^T(YKY)alpha
        L = alpha.sum() - 0.5 * alpha @ (Y @ K @ Y) @ alpha
        L.backward()

        # update alpha
        with torch.no_grad():
            alpha += lr * alpha.grad

            # clamp the range
            if c:
                alpha = torch.clamp_(
                    alpha,
                    min=0,
                    max=c,
                )
            else:
                alpha = torch.clamp_(
                    alpha,
                    min=0,
                )
            alpha.grad.zero_()
        alpha.requires_grad_()

    return alpha.detach()


def svm_predictor(alpha, x_train, y_train, x_test, kernel=hw3_utils.poly(degree=1)):
    """
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    """
    N = x_train.shape[0]
    M = x_test.shape[0]

    K_train = create_kernel(x_train, x_train, N, N, kernel)
    K_test = create_kernel(x_train, x_test, N, M, kernel)

    support_idx = torch.where(alpha > _EPS)[0]
    if len(support_idx) == 0:
        return torch.zeros(M)

    min_alpha_idx = support_idx[torch.argmin(alpha[support_idx])]

    # y_support - \sum^{N}_{j = 1} \alpha_j * y_j * K(x_j, x_k)
    b = y_train[min_alpha_idx] - (alpha * y_train * K_train[:, min_alpha_idx]).sum()

    # Compute predictions: f(x) = \sum_{i=1}^{N} \alpha_i * y_i * K(x_i, x) + b
    return torch.sum((alpha * y_train).unsqueeze(1) * K_test, dim=0) + b
