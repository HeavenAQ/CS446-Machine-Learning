from typing import List, Tuple
import torch
from torch import FloatTensor, LongTensor, Tensor
import hw2_utils as utils
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split


def gaussian_theta(X: List[FloatTensor], y: LongTensor) -> Tuple[Tensor, Tensor]:
    """
    Arguments:
        X (S x N FloatTensor): features of each object
        y (S LongTensor): label of each object, y[i] = 0/1

    Returns:
        mu (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x N Float Tensor): MAP estimation of sigma in N(mu, sigma2)

    """
    mu = torch.stack(
        [
            X[y == 0].mean(dim=0),
            X[y == 1].mean(dim=0),
        ],
        dim=0,
    )
    sigma2 = torch.stack(
        [
            ((X[y == 0] - mu[0]) ** 2).mean(dim=0),
            ((X[y == 1] - mu[1]) ** 2).mean(dim=0),
        ],
        dim=0,
    )

    return mu, sigma2


def gaussian_p(y: LongTensor):
    """
    Arguments:
        y (S LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    """
    return (y == 0).sum() / len(y)


def gaussian_classify(mu: Tensor, sigma2: Tensor, p: FloatTensor, X: Tensor):
    """
    Arguments:
        mu (2 x N Float Tensor): returned value #1 of `gaussian_MAP`
        sigma2 (2 x N Float Tensor): returned value #2 of `gaussian_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (S x N LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor): label of each object for classification, y[i] = 0/1

    """
    log_p = torch.tensor([torch.log(p), torch.log(1 - p)])

    posteriors = []
    for c in range(2):
        # Broacast likelihood function to every sample
        log_likelihoods = (
            -0.5 * torch.log(2 * torch.pi * sigma2[c])
            - (X - mu[c]) ** 2 / (2 * sigma2[c])
        ).sum(dim=1)  # sum over the features of the same data point
        posteriors.append(log_likelihoods + log_p[c])

    # stack for argmax to compare the results
    posteriors = torch.stack(posteriors, dim=1)
    return torch.argmax(posteriors, dim=1)
