import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch

class SlicedInverseRegression:
    """
    Implementation of SIR (Li, 1991)
    """
    def __init__(self, x=None, y=None, reduced_dim=2, H=10):
        """
        Initialization of SIR
        
        Parameters
        ----------
        x: array
            n by p array representing the covariates of the problem
        y: array
            n dim array representing the target variable
        reduced_dim: int
            the number of directions we want to find
        H: int
            the number of slices
        """
        if x is None: 
            raise ValueError("Covariates are not given!")
        if y is None: 
            raise ValueError("Y is not given!")
        # if type(x) == "Tensor":
        #     self.x = x.detach().numpy().T
        # else:
        #     self.x = x.T
        self.x = torch.stack(x).detach().numpy().T
        self.dim = self.x.shape[-1]
        self.p = self.x.shape[0]
        if self.p < reduced_dim: 
            raise ValueError("The dimension of covariates cannot be smaller than k!")
        
        self.H = H
        self.y = np.array(y)
        self.k = reduced_dim
    
    def split(self):
        """
        The function that splits y into H groups
        
        Returns
        -------
        res: list
            each entry represents the boundary (included) of one slice
        """
        y = np.sort(self.y)
        res = [y[0]]
        num = 0
        step = self.dim // self.H
        while len(res) < self.H:
            num += step
            res.append(y[num])
        res.append(y[-1])
        return res
        
    
    def direction(self):
        """
        The function that finds the directions
        
        Returns
        -------
        direction: list(array)
            each entry is a p-array representing the direction
        """
        sigma = np.cov(self.x) # sample covariance matrix
        invsigma = np.linalg.inv(sqrtm(sigma))
        # print(invsigma)
        # print(type(self.x))
        x_standard = invsigma@(self.x - self.x.mean(axis=1, keepdims=True))
        # print(self.y, self.y.shape)
        slices = self.split()
        # print(slices)
        digitized = np.digitize(self.y, slices, right=True)
        # digitized = self.y
        # slices = np.unique(self.y).shape[0]
        slice_mean = [x_standard[:, digitized == i].mean(axis=1) for i in range(len(slices))]
        
        # print(x_standard[:, digitized == 1])
        
        # principal component analysis
        v = 0
        for h in range(len(slices)):
            weight = (digitized == h).sum() / self.dim
            # print(weight)
            v += weight * np.outer(slice_mean[h], slice_mean[h])

        eigenvalue, eigenvector = np.linalg.eig(v)
        zipped = sorted(zip(eigenvalue, eigenvector.T), key=lambda x: -x[0])

        self.direction = []
        for i in range(self.k):
            beta = zipped[i][1]@invsigma
            entry = beta / np.sqrt(np.sum(beta**2))
            # print(entry)
            # print(entry.real.dtype)
            self.direction.append(entry.real)
        return self.direction
    
    def plot(self):
        """
        Scatterplot of the (at most) first two reduced dimensions
        """
        if len(self.direction) == 0:
            raise ValueError("The reduced direction is not given yet!")
        if len(self.direction) == 1:
            plt.scatter(np.sum(self.x * self.direction[0].reshape((self.p, 1)), axis=0), self.y)
            plt.show()
        if len(self.direction) == 2:
            plt.scatter(np.sum(self.x * self.direction[0], axis=0), np.sum(self.x * self.direction[1], axis=1))
            plt.show()
    
    
if __name__ == '__main__':
    x = []
    y = []
    w = torch.ones(10)
    for i in range(1000):
        x0 = torch.randn(10)
        x.append(x0)
        y.append(x0[0] * (x0[0] + x0[1] + 1) + torch.randn(1)[0])
    
    learner = SlicedInverseRegression(x, y)
    print(learner.direction())