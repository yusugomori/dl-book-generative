import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(device=device)
        self.decoder = Decoder(device=device)

    def forward(self, x):
        mean, var = self.encoder(x)
        z = self.reparameterize(mean, var)
        y = self.decoder(z)

        return y

    def reparameterize(self, mean, var):
        eps = torch.randn(mean.size()).to(self.device)
        z = mean + torch.sqrt(var) * eps
        return z

    def lower_bound(self, x):
        mean, var = self.encoder(x)
        kl = - 1/2 * torch.mean(torch.sum(1
                                          + torch.log(var)
                                          - mean**2
                                          - var, dim=1))
        z = self.reparameterize(mean, var)
        y = self.decoder(z)

        reconst = torch.mean(torch.sum(x * torch.log(y)
                                       + (1 - x) * torch.log(1 - y),
                                       dim=1))

        L = reconst - kl

        return L


class Encoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(784, 200)
        self.l2 = nn.Linear(200, 200)
        self.l_mean = nn.Linear(200, 10)
        self.l_var = nn.Linear(200, 10)

    def forward(self, x):
        h = self.l1(x)
        h = torch.relu(h)
        h = self.l2(h)
        h = torch.relu(h)

        mean = self.l_mean(h)
        var = F.softplus(self.l_var(h))

        return mean, var


class Decoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(10, 200)
        self.l2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, 784)

    def forward(self, x):
        h = self.l1(x)
        h = torch.relu(h)
        h = self.l2(h)
        h = torch.relu(h)
        h = self.out(h)
        y = torch.sigmoid(h)

        return y


if __name__ == '__main__':
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    1. Load data
    '''
    root = os.path.join(os.path.dirname(__file__),
                        '.', 'data', 'fashion_mnist')
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda x: x.view(-1)])
    mnist_train = \
        torchvision.datasets.FashionMNIST(root=root,
                                          download=True,
                                          train=True,
                                          transform=transform)
    train_dataloader = DataLoader(mnist_train,
                                  batch_size=100,
                                  shuffle=True)

    '''
    2. Build model
    '''
    model = VAE(device=device).to(device)

    '''
    3. Train model
    '''
    criterion = model.lower_bound
    optimizer = optimizers.Adam(model.parameters())

    def compute_loss(x):
        return -1 * criterion(x)

    def train_step(x):
        model.train()
        loss = compute_loss(x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    epochs = 10

    for epoch in range(epochs):
        train_loss = 0.

        for (x, _) in train_dataloader:
            x = x.to(device)
            loss = train_step(x)

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        print('Epoch: {}, Cost: {:.3f}'.format(
            epoch+1,
            train_loss
        ))

    '''
    4. Test model
    '''
    def gen_noise(batch_size):
        return torch.empty(batch_size, 10).normal_().to(device)

    def generate(batch_size=16):
        model.eval()
        z = gen_noise(batch_size)
        gen = model.decoder(z)
        gen = gen.view(-1, 28, 28)

        return gen

    images = generate(batch_size=16)
    images = images.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    for i, image in enumerate(images):
        plt.subplot(4, 4, i+1)
        plt.imshow(image, cmap='binary_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
