# This program creates the MNIST-R dataset as mentioned in Ghifary et al., 2015

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from utils.SIRd_class import SIRd

parser = argparse.ArgumentParser(description='InvSIR for RMNIST')
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--selected_features', type=int, default=256)
parser.add_argument('--slices', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--steps', type=int, default=501)
flags = parser.parse_args()
print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

# Function to rotate image
def rotate_image(image, angle):
    pil_image = to_pil_image(image)
    rotated_image = pil_image.rotate(angle)
    return to_tensor(rotated_image)
   
def make_environment(images, labels, e):
    """
    The function that makes several environments by rotating original MNIST images
    
    Parameters
    ----------
    data: tuple(tensor)
        a tuple that contains the image tensors and corresponding labels
    n_i: int
        images in each 0-9 class
    e: list
        rotation angles (in degree) of each environment
    plot: Boolean
        whether plot a sample or not
        
    Returns
    -------
    image_rotated: tuple(tensor)
        rotated images and corresponding labels
    """
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
            'images': (images.float() / 255. + torch.rand(images.shape) * 1e-5).cuda(),
            'labels': labels[:, None].cuda()
        }


class ClassNet(nn.Module):
    def __init__(self):
        super(ClassNet, self).__init__()
        lin1 = nn.Linear(flags.selected_features, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 10)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3, nn.Softmax(dim=1))
    def forward(self, input):
        out = self._main(input)
        return out
    
def mean_nll(logits, y):
    return nn.functional.cross_entropy(logits, y.view(y.shape[0]))

def mean_accuracy(logits, y):
    softmax = nn.Softmax(dim=1)
    preds = torch.argmax(softmax(logits), dim=1)
    return ((preds - y.view(y.shape[0])).abs() < 1e-2).float().mean()

def pretty_print(*values):
    col_width = 31
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))
    
torch.manual_seed(0)    
final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)
    mnist = datasets.MNIST('./datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])   

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())
    
    envs = [
        make_environment(mnist_train[0][::5], mnist_train[1][::5], 15),
        make_environment(mnist_train[0][1::5], mnist_train[1][1::5], 30),
        make_environment(mnist_train[0][2::5], mnist_train[1][2::5], 45),
        make_environment(mnist_train[0][3::5], mnist_train[1][3::5], 60),
        make_environment(mnist_train[0][4::5], mnist_train[1][4::5], 75),
        make_environment(mnist_val[0], mnist_val[1], 0)
    ]
    
    images_train = [envs[i]['images'] for i in range(5)]
    labels_train = [envs[i]['labels'] for i in range(5)]
    images_val = [envs[5]['images']]
    labels_val = [envs[5]['labels']]
    
    
    net = ClassNet().cuda()
    optimizer = optim.Adam(net.parameters(), lr=flags.lr)
    dir_learner = SIRd(images_train, labels_train, reduced_dim=flags.selected_features, H=flags.slices)
    directions = dir_learner.train()
    directions = np.array(directions)
    directions = torch.from_numpy(directions).double().cuda()
    
    for env in envs:
        images = env['images']
        new_images = []
        for n in range(images.shape[0]):
            image = images[n]
            input = image.view(image.shape[0] * image.shape[1]).double()
            new_images.append(directions @ input)
        new_images = torch.stack(new_images).to(torch.float32)
        env['images'] = new_images
        
    # train Neural Network
    pretty_print('step', 'train nll', 'train acc', 'test acc')
    for step in range(flags.steps):
        for env in envs:
            logits = net(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            
        train_nll = torch.stack([envs[i]['nll'] for i in range(4)]).mean()
        train_acc = torch.stack([envs[i]['acc'] for i in range(4)]).mean()
        loss = train_nll.clone()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        test_acc = envs[5]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy()
            )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    
    
# plot    
# final_train_accs = []
# final_test_accs = []
# for restart in range(200, 257):
#     flags.selected_features = restart
#     print("Restart", restart)
#     mnist = datasets.MNIST('./datasets/mnist', train=True, download=True)
#     mnist_train = (mnist.data[:50000], mnist.targets[:50000])
#     mnist_val = (mnist.data[50000:], mnist.targets[50000:])   

#     rng_state = np.random.get_state()
#     np.random.shuffle(mnist_train[0].numpy())
#     np.random.set_state(rng_state)
#     np.random.shuffle(mnist_train[1].numpy())
    
#     envs = [
#         make_environment(mnist_train[0][::5], mnist_train[1][::5], 0),
#         make_environment(mnist_train[0][1::5], mnist_train[1][1::5], 15),
#         make_environment(mnist_train[0][2::5], mnist_train[1][2::5], 30),
#         make_environment(mnist_train[0][3::5], mnist_train[1][3::5], 45),
#         make_environment(mnist_train[0][4::5], mnist_train[1][4::5], 60),
#         make_environment(mnist_val[0], mnist_val[1], 90)
#     ]
    
#     # # plot images
#     # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
#     # image = envs[4]['images'][355]
#     # label = envs[4]['labels'][355]
#     # print(label)
#     # plt.imshow(image.cpu())
#     # plt.show()
    
    
#     images_train = [envs[i]['images'] for i in range(4)]
#     labels_train = [envs[i]['labels'] for i in range(4)]
#     images_val = [envs[5]['images']]
#     labels_val = [envs[5]['labels']]
    
    
#     net = ClassNet().cuda()
#     optimizer = optim.Adam(net.parameters(), lr=flags.lr)
#     dir_learner = SIRd(images_train, labels_train, reduced_dim=flags.selected_features, H=flags.slices)
#     directions = dir_learner.train()
#     directions = np.array(directions)
#     directions = torch.from_numpy(directions).double().cuda()
    
#     for env in envs:
#         images = env['images']
#         new_images = []
#         for n in range(images.shape[0]):
#             image = images[n]
#             input = image.view(image.shape[0] * image.shape[1]).double()
#             new_images.append(directions @ input)
#         new_images = torch.stack(new_images).to(torch.float32)
#         env['images'] = new_images
        
#     # train Neural Network
#     pretty_print('step', 'train nll', 'train acc', 'test acc')
#     for step in range(flags.steps):
#         for env in envs:
#             logits = net(env['images'])
#             env['nll'] = mean_nll(logits, env['labels'])
#             env['acc'] = mean_accuracy(logits, env['labels'])
            
#         train_nll = torch.stack([envs[i]['nll'] for i in range(4)]).mean()
#         train_acc = torch.stack([envs[i]['acc'] for i in range(4)]).mean()
#         loss = train_nll.clone()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         test_acc = envs[5]['acc']
#         if step % 100 == 0:
#             pretty_print(
#                 np.int32(step),
#                 train_nll.detach().cpu().numpy(),
#                 train_acc.detach().cpu().numpy(),
#                 test_acc.detach().cpu().numpy()
#             )

#     final_train_accs.append(train_acc.detach().cpu().numpy())
#     final_test_accs.append(test_acc.detach().cpu().numpy())
#     print('Final train acc (mean/std across restarts so far):')
#     print(np.mean(final_train_accs), np.std(final_train_accs))
#     print('Final test acc (mean/std across restarts so far):')
#     print(np.mean(final_test_accs), np.std(final_test_accs))
    
# # final = np.array(final_test_accs)
# final = np.load('plot.npy')
# plt.plot(np.arange(200, 257), final, linestyle='-.', color='blue')
# plt.xlabel('Number of Features')
# plt.ylabel('Accuracy')
# plt.savefig('acc_vs_features')
# plt.show()
# # np.save('plot', final)