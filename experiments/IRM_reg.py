# Invariant Risk Minimization (IRM) on RMNIST dataset. 

import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch import nn, optim, autograd
from PIL import Image
from InvSIR_reg import PIIF, shift_lin, shift_multifeature

parser = argparse.ArgumentParser(description='IRM for RMNIST')
parser.add_argument('--hidden_dim', type=int, default=20)
parser.add_argument('--input_size', type=int, default=10)
parser.add_argument('--spurious_size', type=int, default=10)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--example_setting', type=int, default=11)
flags = parser.parse_args()
print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))
    
final_train_nlls = []
final_test_nlls = []
torch.manual_seed(0)
for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load data
    w1 = torch.ones(flags.input_size)
    w2 = torch.randn(flags.spurious_size)
    if flags.example_setting == 11:
        envs = [np.sqrt(0.1), np.sqrt(1.5), np.sqrt(2.0)]
        X, Y = PIIF(w1, w2, envs, scramble=False)
    elif flags.example_setting == 12:
        envs = [np.sqrt(0.1), np.sqrt(1.5), np.sqrt(2.0)]
        X, Y = PIIF(w1, w2, envs, scramble=True)
    elif flags.example_setting == 13:
        envs = [np.sqrt(0.1), np.sqrt(0.3), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(1.5), np.sqrt(2.0)]
        X, Y = PIIF(w1, w2, envs, scramble=True)
    elif flags.example_setting == 21:
        envs = [np.sqrt(0.1), np.sqrt(1.5), np.sqrt(2.0)]
        X, Y = shift_lin(w1, envs)
    elif flags.example_setting == 22:
        envs = [np.sqrt(0.1), np.sqrt(0.3), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(1.5), np.sqrt(2.0)]
        X, Y = shift_lin(w1, envs)
    elif flags.example_setting == 31:
        envs = [np.sqrt(0.1), np.sqrt(1.5), np.sqrt(2.0)]
        X, Y = shift_multifeature(w1, envs)
    elif flags.example_setting == 32:
        envs = [np.sqrt(0.1), np.sqrt(0.3), np.sqrt(0.5), np.sqrt(1.0), np.sqrt(1.5), np.sqrt(2.0)]
        X, Y = shift_multifeature(w1, envs)
    else:
        raise ValueError('Not a valid example!')
    
    envs = [{'cov': torch.stack(X[i]).cuda(), 'labels': torch.stack(Y[i]).cuda()} for i in range(len(X))]
   

    # Define and instantiate the model

    class RegressionNet(nn.Module):
        def __init__(self):
            super(RegressionNet, self).__init__()
            self.fc1 = nn.Linear(flags.input_size + flags.spurious_size, flags.hidden_dim)
            self.activ = nn.ReLU()
            self.fc2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            self.fc3 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            self.fc4 = nn.Linear(flags.hidden_dim, 1)

        def forward(self, x):
            out = self.fc1(x)
            out = self.activ(out)
            out = self.fc2(out)
            out = self.activ(out)
            out = self.fc3(out)
            out = self.activ(out)
            out = self.fc4(out)
            return out 

        
    net = RegressionNet().cuda()

    # Define loss function helpers

    def mean_nll(out, y):
        return nn.functional.mse_loss(out.view(out.shape[0]), y)

    def penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def pretty_print(*values):
        col_width = 31
        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)
        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))
        
    optimizer = optim.Adam(net.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train penalty', 'test nll')

    for step in range(flags.steps):
        for env in envs:
            out = net(env['cov'])
            env['nll'] = mean_nll(out, env['labels'])
            env['penalty'] = penalty(out, env['labels'])
            
        train_nll = torch.stack([envs[i]['nll'] for i in range(len(envs) - 1)]).mean()
        train_penalty = torch.stack([envs[i]['penalty'] for i in range(len(envs) - 1)]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in net.parameters():
            weight_norm += w.norm().pow(2)
            
        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight 
                        if step >= flags.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            loss /= penalty_weight
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_nll = envs[-1]['nll']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_nll.detach().cpu().numpy()
            )
    final_train_nlls.append(train_nll.detach().cpu().numpy())
    final_test_nlls.append(test_nll.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_nlls), np.std(final_train_nlls))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_nlls), np.std(final_test_nlls))