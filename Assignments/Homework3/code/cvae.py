from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.class_size = class_size
        self.latent_size = latent_size
        self.units = 400


        ######################################################
        ###              START OF YOUR CODE                ###
        ######################################################
        ### Define a three layer neural network architecture #
        ### for the recognition_model                        #
        ######################################################

        self.fc1 = nn.Linear(self.input_size + self.class_size, self.units)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.units, self.units)
        self.relu2 = nn.ReLU()
        self.fc31 = nn.Linear(self.units, self.latent_size)
        self.fc32 = nn.Linear(self.units, self.latent_size)

        ######################################################
        ###               END OF YOUR CODE                 ###
        ######################################################


        ######################################################
        ###              START OF YOUR CODE                ###
        ######################################################
        ### Define a three layer neural network architecture #
        ### for the generation_model                         #
        ######################################################

        self.fc4 = nn.Linear(self.latent_size + self.class_size, self.units)
        self.relu3 = nn.ReLU()
        self.fc5 = nn.Linear(self.units, self.units)
        self.relu4 = nn.ReLU()
        self.fc6 = nn.Linear(self.units, self.input_size)
        self.sigmoid = nn.Sigmoid()

        ######################################################
        ###               END OF YOUR CODE                 ###
        ######################################################



    def recognition_model(self, x, c):
        """
        Computes the parameters of the posterior distribution q(z | x, c) using the
        recognition network defined in the constructor
    
        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class
        
        Returns:
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar PyTorch Variable of shape (batch_size, latent_size) for the posterior
          variance in log space
        """
        ###########################
        ######### TO DO ###########
        ###########################
        inputs = torch.cat([x, c], 1)
        out1 = self.fc1(inputs)
        out2 = self.relu1(out1)
        out3 = self.fc2(out2)
        out4 = self.relu2(out3)
        mu = self.fc31(out4)
        logvar = self.fc32(out4)
        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu

    def generation_model(self, z, c): # P(x|z, c)
        """
        Computes the generation output from the generative distribution p(x | z, c)
        using the generation network defined in the constructor
    
        Inputs:
        - z: PyTorch Variable of shape (batch_size, latent_size) for the latent vector
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class
        
        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        """
        ###########################
        ######### TO DO ###########
        ###########################
        inputs = torch.cat([z, c], 1)
        out1 = self.fc4(inputs)
        out2 = self.relu3(out1)
        out3 = self.fc5(out2)
        out4 = self.relu4(out3)
        out5 = self.fc6(out4)
        x_hat = self.sigmoid(out5)
        return x_hat

    def forward(self, x, c):
        """
        Performs the inference and generation steps of the CVAE model using
        the recognition_model, reparametrization trick, and generation_model
    
        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class
        
        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar: PyTorch Variable of shape (batch_size, latent_size)
                  for the posterior logvar
        """
        ###########################
        ######### TO DO ###########
        ###########################
        mu, logvar = self.recognition_model(x.view(-1, 28*28), c)
        z = self.reparametrize(mu, logvar)
        x_hat = self.generation_model(z, c)
        return x_hat, mu, logvar


def to_var(x, use_cuda):
    x = Variable(x)
    if use_cuda:
        x = x.cuda()
    return x


def one_hot(labels, class_size, use_cuda):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return to_var(targets, use_cuda)


def train(epoch, model, train_loader, optimizer, num_classes, use_cuda):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = to_var(data, use_cuda).view(data.shape[0], -1)
        labels = one_hot(labels, num_classes, use_cuda)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lowerbound for conditional vae
    Note: We compute -lowerbound because we optimize the network by minimizing a loss

    Inputs:
    - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
    - x: PyTorch Variable of shape (batch_size, input_size) for the real data
    - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
    - logvar: PyTorch Variable of shape (batch_size, latent_size) for the posterior logvar
    
    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the negative lowerbound.
    """
    ###########################
    ######### TO DO ###########
    ###########################
    # https://arxiv.org/abs/1312.6114
    BCE = F.binary_cross_entropy(x_hat, x.view(-1, 28*28))
    KLD = -0.5* torch.sum(1+logvar-torch.pow(mu, 2)-torch.exp(logvar))
    loss = BCE+KLD
    return loss


def main():
    # Load MNIST dataset
    use_cuda = True
    input_size = 28 * 28
    units = 400
    batch_size = 32
    latent_size = 20 # z dim
    num_classes = 10
    num_epochs = 11
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)

    model = CVAE(input_size, latent_size, num_classes)

    if use_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, num_epochs):
        train(epoch, model, train_loader, optimizer, num_classes, use_cuda)

    # Generate images with condition labels
    c = torch.eye(num_classes, num_classes) # [one hot labels for 0-9]
    c = to_var(c, use_cuda)
    z = to_var(torch.randn(num_classes, latent_size), use_cuda)
    samples = model.generation_model(z, c).data.cpu().numpy()

    fig = plt.figure(figsize=(10, 1))
    gs = gridspec.GridSpec(1, 10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    plt.show()


if __name__ == "__main__":
  main()

