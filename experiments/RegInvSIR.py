from utils.SIRd_reg import SIRd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim, autograd
from torchvision import datasets
import torchvision.models as models
# export PYTHONPATH=.
    
class RegressionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activ = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activ(out)
        out = self.fc2(out)
        out = self.activ(out)
        out = self.fc3(out)
        out = self.activ(out)
        out = self.fc4(out)
        return out 


def flatten(obj):
    """
    Flatten a list(list)
    """
    res = []
    for i in obj:
        for j in i:
            res.append(j)
    return res
            
class RegInvSIR:
    def __init__(self, train, train_label, test, test_label, feature_num, hidden_layer, H=10):
        """
        Initialization of Invariant SIR algorithm
        
        Parameters
        ----------
        x: array(array(array))
            n_e by n_i by p array-like representing covariates
        y: array(array)
            n_e by n_i array representing target variable
        feature_num: int
            the number of features we want to use in the training process
        H: int
            the number of slices
        """
        if train is None: 
            raise ValueError("Covariates are not given!")
        if train_label is None: 
            raise ValueError("Y is not given!")
        
        self.ne = len(train) # number of environments
        self.p = torch.numel(train[0][0])
        if self.p < feature_num: 
            raise ValueError("The dimension of covariates cannot be smaller than k!")
        
        self.n = 0
        for i in range(self.ne):
            self.n += len(train[i])
        
        self.H = H
        self.x = train # list(list(tensor))
        self.y = train_label # list(list(tensor))
        self.test = test # list(list(tensor))
        self.test_label = test_label
        self.K = feature_num
        self.hidden = hidden_layer
            
    def feature(self):

        learner = SIRd(self.x, self.y, self.K, self.H)
        directions = learner.train()
        print(directions)
        train = []
        test = []
        for env in range(len(self.x)):
            for n in range(len(self.x[env])):
                train.append([])
                for vec in directions:
                    train[-1].append(torch.from_numpy(vec.real.astype(np.float32)) @ self.x[env][n]) 
        for env in range(len(self.test)):
            for n in range(len(self.test[env])):
                test.append([])
                for vec in directions:
                    test[-1].append(torch.from_numpy(vec.real.astype(np.float32)) @ self.test[env][n]) 
        return torch.tensor(train), torch.tensor(test)
                    


    def train(self, steps):
        """
        training function of InvSIR
        
        Parameters
        ----------
        hidden_dim: int
            indicates the number of hidden layers in neural network
        
        Returns
        -------
        
        """
        train, test = self.feature()
        train = train.view(train.shape[0], 1, train.shape[-1]).cuda()
        test = test.view(test.shape[0], 1, test.shape[-1]).cuda()
        train_labels = torch.tensor(flatten(self.y)).cuda()
        test_labels = torch.tensor(flatten(self.test_label)).cuda()
        # print(features.shape)
        # net = SimpleCNN(self.K).cuda()
        net = RegressionNet(self.K, self.hidden, 1).cuda()
        optimizer = optim.SGD(net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        for step in range(steps):
            logits = net(train)
            logits = logits.view(logits.shape[0])
            # print(logits, logits.shape)
            loss = loss_fn(logits, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                predicted = logits
                print(step)
                print("Loss:", loss.item())
                # print("training accuracy:", predicted.shape)
        out = net(test)
        out = out.view(out.shape[0])
        print("test loss:", loss_fn(out, test_labels))
        return loss_fn(out, test_labels).detach().cpu()
    
        
        
        
