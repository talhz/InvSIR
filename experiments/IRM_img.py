# Invariant Risk Minimization (IRM) on RMNIST dataset. 

import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch import nn, optim, autograd
from PIL import Image

parser = argparse.ArgumentParser(description='IRM for RMNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
flags = parser.parse_args()
print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))
    
torch.manual_seed(0)    
final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST and shuffle

    mnist = datasets.MNIST('./datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Function to rotate image

    def rotate_image(image, angle):
        pil_image = to_pil_image(image)
        rotated_image = pil_image.rotate(angle)
        return to_tensor(rotated_image)

    # Build environments

    def make_environment(images, labels, e):
        downsample_transform = transforms.Compose([
            transforms.Resize((16, 16)), 
        ])
        images = downsample_transform(images)
        if isinstance(e, list):
            for image in images:
                n = len(e)
                idx = np.random.randint(0, n)
                image = rotate_image(image, e[idx])
        else:
            for image in images:
                image = rotate_image(image, e)
            
        return {
            'images': (images.float() / 255.).cuda(),
            'labels': labels[:, None].cuda()
        }
        
    envs = [
        make_environment(mnist_train[0][::5], mnist_train[1][::5], 15),
        make_environment(mnist_train[0][1::5], mnist_train[1][1::5], 30),
        make_environment(mnist_train[0][2::5], mnist_train[1][2::5], 45),
        make_environment(mnist_train[0][3::5], mnist_train[1][3::5], 60),
        make_environment(mnist_train[0][4::5], mnist_train[1][4::5], 75),
        make_environment(mnist_val[0], mnist_val[1], 0)
    ]

    # Define and instantiate the model

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            lin1 = nn.Linear(16 * 16, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, 10)
            for lin in [lin1, lin2, lin3]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
        
        def forward(self, input):
            out = input.view(input.shape[0], 16 * 16)
            return self._main(out)
        
    mlp = MLP().cuda()

    # Define loss function helpers

    def mean_nll(logits, y):
        return nn.functional.cross_entropy(logits, y.view(y.shape[0]))

    def mean_accuracy(logits, y):
        softmax = nn.Softmax(dim=1)
        preds = torch.argmax(softmax(logits), dim=1)
        return ((preds - y.view(y.shape[0])).abs() < 1e-2).float().mean()

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
        
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])
            
        train_nll = torch.stack([envs[i]['nll'] for i in range(5)]).mean()
        train_acc = torch.stack([envs[i]['acc'] for i in range(5)]).mean()
        train_penalty = torch.stack([envs[i]['penalty'] for i in range(5)]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
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

        test_acc = envs[5]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy()
            )
    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))