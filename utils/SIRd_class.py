from .SIR_class import SlicedInverseRegression
import numpy as np

class SIRd:
    def __init__(self, x, y, reduced_dim=2, H=10):
        """
        Initialization of SIRd
        
        Parameters
        ----------
        x: list(tensor)
            n_e by n_i by p by p structure representing covariates
        y: list(tensor)
            n_e by n_i by 1 structure representing target variable
        reduced_dim: int
            the number of directions we want to find
        H: int
            the number of slices
        """
        if x is None: 
            raise ValueError("Covariates are not given!")
        if y is None: 
            raise ValueError("Y is not given!")
        
        self.ne = len(x) # number of environments
        self.p = x[0].shape[-1] * x[0].shape[-2]
        if self.p < reduced_dim: 
            raise ValueError("The dimension of covariates cannot be smaller than k!")
        
        self.n = 0
        for i in range(self.ne):
            self.n += x[i].shape[0]
        
        self.H = H
        self.x = x
        self.y = y
        self.k = reduced_dim
        
    @staticmethod
    def proximity(mat1, mat2, K):
        """
        Function to calculte the distance of the given two matrices
        """
        p1 = mat1.T @ mat1
        p2 = mat2.T @ mat2
        
        return np.trace(p1 @ p2) / K
     
    def train(self, weights=None):
        """
        Training of the model
        
        Parameters
        ----------
        weights: NoneType or list
            If weights is None, then the weights are chosen automatically according to ne
            If weights is a given list, then check if the summation of weights are 1 and then use it as the weights in the training process
        
        Returns
        -------
        direction: array
            The array representing reduced direction(s)
        """
        
        # Check weights: 
        if type(weights) == list:
            if sum(weights) != 1:
                raise ValueError("The sum of weights must be 1!")
            if len(weights) != self.ne:
                raise ValueError("The length of weights should be the same as the number of environments!")
        elif weights is None:
            print("computing weights automatically...\n")
            weights = []
            for i in range(self.ne):
                weights.append(self.ne / self.n)
        else:
            raise TypeError("Weights must be None or list!")
        
        # train
        matrices = []
        for i in range(self.ne):
            learner = SlicedInverseRegression(self.x[i], self.y[i], self.k, self.H)
            mat = learner.direction()
            matrices.append(np.array(mat))
            
        M = 0
        
        for i in range(self.ne):
            M += (weights[i] * SIRd.proximity(matrices[i], matrices[-1], self.k)) * matrices[i].T @ matrices[i] / self.k
            
        eigenvalue, eigenvector = np.linalg.eig(M)
        zipped = sorted(zip(eigenvalue, eigenvector.T), key=lambda x: -x[0])

        self.direction = []
        for i in range(self.k):
            beta = zipped[i][1]
            self.direction.append(beta / np.sqrt(np.sum(beta**2)))
        return self.direction
  