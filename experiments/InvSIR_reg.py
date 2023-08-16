import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from RegInvSIR import RegInvSIR

def PIIF(wyz, wzy, envs, num=1000, scramble=False):
    """
    Example 1/1S in Ahuja et al. (2021)
    
    Parameters
    ----------
    wyz (tensor): the coefficient in the equation Z -> Y
    wzy (tensor): the coefficient in the equation Y -> Z
    envs (list): list of floats indicating standard deviations
    scramble (Boolean): indicates the choice of S function
    m (int): the dimension of Z_inv
    o (int): the dimension of Z_spu
    
    Returns
    -------
    data (tuple): the dataset
    """
    # len(envs) == len(nums)
    m = wyz.shape[0]
    o = wzy.shape[0]
    X = []
    Y = []
    for i in range(len(envs)):
        X.append([])
        Y.append([])
        zinv = envs[i] * torch.randn((m, num))
        ytilde = envs[i] * (torch.randn((m, num)) + wyz @ zinv)
        zspu = (torch.randn((o, num)) + wzy @ ytilde)
        y = 2 * ytilde.sum(dim=0) / (m + o)
        z = torch.cat([zinv, zspu])
        if scramble:
            A = torch.randn((m + o, m + o))
            S = torch.linalg.svd(A)[0]
            x = S @ z
            for i in range(x.shape[1]):
                X[-1].append(x[:, i])
                Y[-1].append(y[i])
        else:
            for i in range(z.shape[1]):
                X[-1].append(z[:, i])
                Y[-1].append(y[i])
        
    return X, Y

def shift_exp(beta, envs, num=1000):
    p = beta.shape[0]
    X = []
    Y = []
    for i in range(len(envs)):
        X.append([])
        Y.append([])
        x = envs[i] * torch.randn((p, num))
        y = torch.exp(beta @ x) + torch.randn(num)
        for i in range(x.shape[1]):
            X[-1].append(x[:, i])
            Y[-1].append(y[i])
    return X, Y

def shift_lin(beta, envs, num=1000):
    p = beta.shape[0]
    X = []
    Y = []
    for i in range(len(envs)):
        X.append([])
        Y.append([])
        x = envs[i] * torch.randn((p, num))
        y = beta @ x + torch.randn(num)
        for i in range(x.shape[1]):
            X[-1].append(x[:, i])
            Y[-1].append(y[i])
    return X, Y

def shift_multifeature(beta, envs, num=1000):
    p = beta.shape[0]
    X = []
    Y = []
    for i in range(len(envs)):
        X.append([])
        Y.append([])
        x = envs[i] * torch.randn((p, num))
        y = torch.sin(beta @ x) + torch.abs(beta @ x) * torch.randn(num)
        for i in range(x.shape[1]):
            X[-1].append(x[:, i])
            Y[-1].append(y[i])
    return X, Y

def example1(setting=1):
    torch.manual_seed(0)
    w1 = torch.ones(10)
    w2 = torch.randn(10)
    if setting == 1:
        envs = [np.sqrt(0.1), np.sqrt(1.5), np.sqrt(2.0)]
        num = 1
        s = False
    elif setting ==2:
        envs = [np.sqrt(0.1), np.sqrt(1.5), np.sqrt(2.0)]
        num = 10
        s = True 
    elif setting == 3:
        envs = [np.sqrt(0.1), np.sqrt(0.3), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(1.5), np.sqrt(2.0)]
        num = 10
        s = True 
    
    acc = []
    for restart in range(10):
        print('restart: ', restart)  
        X, Y = PIIF(w1, w2, envs, scramble=s)
        Xtrain = [X[i] for i in range(len(X) - 1)]
        Ytrain = [Y[i] for i in range(len(Y) - 1)]
        Xtest = [X[-1]]
        Ytest = [Y[-1]]
        learner = RegInvSIR(Xtrain, Ytrain, Xtest, Ytest, num, 64)
        res = learner.train(1000)
        acc.append(res)
    print("Accuracy:", np.mean(acc))
    print("std:", np.std(acc))
    
def example2(setting=1):
    torch.manual_seed(0)
    w1 = torch.ones(10)
    if setting == 1:
        envs = [np.sqrt(0.1), np.sqrt(1.5), np.sqrt(2.0)]
    elif setting ==2:
        envs = [np.sqrt(0.1), np.sqrt(0.3), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(1.5), np.sqrt(2.0)]
    
    acc = []
    for restart in range(10):
        print('restart: ', restart)  
        X, Y = shift_lin(w1, envs)
        Xtrain = [X[i] for i in range(len(X) - 1)]
        Ytrain = [Y[i] for i in range(len(Y) - 1)]
        Xtest = [X[-1]]
        Ytest = [Y[-1]]
        learner = RegInvSIR(Xtrain, Ytrain, Xtest, Ytest, 1, 64)
        res = learner.train(1000)
        acc.append(res)
    print("Accuracy:", np.mean(acc))
    print("std:", np.std(acc))
    
def example3(setting=1):
    torch.manual_seed(0)
    w1 = torch.ones(10)
    if setting == 1:
        envs = [np.sqrt(0.1), np.sqrt(1.5), np.sqrt(2.0)]
    elif setting ==2:
        envs = [np.sqrt(0.1), np.sqrt(0.3), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(1.5), np.sqrt(2.0)]
    
    acc = []
    for restart in range(10):
        print('restart: ', restart)  
        X, Y = shift_multifeature(w1, envs)
        Xtrain = [X[i] for i in range(len(X) - 1)]
        Ytrain = [Y[i] for i in range(len(Y) - 1)]
        Xtest = [X[-1]]
        Ytest = [Y[-1]]
        learner = RegInvSIR(Xtrain, Ytrain, Xtest, Ytest, 2, 64)
        res = learner.train(1000)
        acc.append(res)
    print("Accuracy:", np.mean(acc))
    print("std:", np.std(acc))
    


def plot_example2():
    torch.manual_seed(0)
    w1 = torch.ones(10)
    envs = [np.sqrt(0.1), np.sqrt(1.5), np.sqrt(2.0)]
    # envs = [np.sqrt(0.1), np.sqrt(0.3), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(1.5), np.sqrt(2.0)]
    save1 = []
    save2 = []
    for i in range(1, 11):
        acc = []
        for restart in range(20):
            print('restart: ', restart)  
            X, Y = shift_lin(w1, envs)
            Xtrain = [X[i] for i in range(len(X) - 1)]
            Ytrain = [Y[i] for i in range(len(Y) - 1)]
            Xtest = [X[-1]]
            Ytest = [Y[-1]]
            learner = RegInvSIR(Xtrain, Ytrain, Xtest, Ytest, i, 64)
            res = learner.train(1000)
            acc.append(res)
        print("Accuracy:", np.mean(acc))
        print("std:", np.std(acc))
        save1.append(np.mean(acc))
        save2.append(np.std(acc))
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, 11), save1, marker='o')
    ax.errorbar(np.arange(1, 11), save1, yerr=save2, fmt='none', capsize=5, elinewidth=1)
    ax.set_xlabel('Num of features')
    ax.set_ylabel('MSE')
    ax.set_title('Example 2')
    plt.show()
    
def plot_example3():
    torch.manual_seed(0)
    w1 = torch.ones(10)
    envs = [np.sqrt(0.1), np.sqrt(1.5), np.sqrt(2.0)]
    # envs = [np.sqrt(0.1), np.sqrt(0.3), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(1.5), np.sqrt(2.0)]
    save1 = []
    save2 = []
    for i in range(1, 11):
        acc = []
        for restart in range(20):
            print('restart: ', restart)  
            X, Y = shift_multifeature(w1, envs)
            Xtrain = [X[i] for i in range(len(X) - 1)]
            Ytrain = [Y[i] for i in range(len(Y) - 1)]
            Xtest = [X[-1]]
            Ytest = [Y[-1]]
            learner = RegInvSIR(Xtrain, Ytrain, Xtest, Ytest, i, 64)
            res = learner.train(1000)
            acc.append(res)
        print("Accuracy:", np.mean(acc))
        print("std:", np.std(acc))
        save1.append(np.mean(acc))
        save2.append(np.std(acc))
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, 11), save1, marker='o')
    ax.errorbar(np.arange(1, 11), save1, yerr=save2, fmt='none', capsize=5, elinewidth=1)
    ax.set_xlabel('Num of features')
    ax.set_ylabel('MSE')
    ax.set_title('Example 3')
    plt.show()
    
if __name__ == "__main__":
    # example1(setting=3)
    # example2(1)
    # example3(2)
    # plot_example2()
    plot_example3()