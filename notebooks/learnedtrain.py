import random
import numpy as np
import torch
import torch.nn as nn
import pylops
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm

    
def train(model, criterion, optimizer, data_loader, tv=0.0, device='cpu', plotflag=False):
    """Training step

    Perform a training step over the entire training data (1 epoch of training)

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    criterion : :obj:`torch.nn.modules.loss`
        Loss function
    optimizer : :obj:`torch.optim`
        Optimizer
    data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training dataloader
    tv : :obj:`bool`, optional
        Add TV to loss (if 0.0 no)
    device : :obj:`str`, optional
        Device
    plotflag : :obj:`bool`, optional
        Display intermediate results

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    accuracy : :obj:`float`
        Accuracy over entire dataset

    """
    model.train()
    loss = 0
    for X0_batch, X_batch, Y_batch in tqdm(data_loader):
        optimizer.zero_grad()
        ls = model.train_batch(Y_batch, X_batch, X0_batch, criterion)        
        ls.backward()
        optimizer.step()
        loss += ls.item()
    loss /= len(data_loader)
    
    return loss


def evaluate(model, criterion, data_loader, device='cpu', plotflag=False):
    """Evaluation step

    Perform an evaluation step over the entire training data

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    criterion : :obj:`torch.nn.modules.loss`
        Loss function
    data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Evaluation dataloader
    device : :obj:`str`, optional
        Device
    plotflag : :obj:`bool`, optional
        Display intermediate results

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    accuracy : :obj:`float`
        Accuracy over entire dataset

    """
    model.train()  # not eval because https://github.com/facebookresearch/SparseConvNet/issues/166
    loss = 0
    for X0_batch, X_batch, Y_batch in tqdm(data_loader):
        with torch.no_grad():
            ls = model.train_batch(Y_batch, X_batch, X0_batch, criterion)        
        loss += ls.item()
    loss /= len(data_loader)
    
    #if plotflag:
    #    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    #    ax.plot(y.detach().squeeze()[:5].T, "k")
    #    ax.plot(ypred.detach().squeeze()[:5].T, "r")
    #    ax.set_xlabel("t")
    #    plt.show()
    
    return loss