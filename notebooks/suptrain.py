import torch
import torch.nn as nn


def train(model, criterion, optimizer, data_loader, device='cpu', plotflag=False):
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
    device : :obj:`str`, optional
        Device


    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    
    """
    model.train()
    loss = 0
    for X, y in tqdm(data_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        yprob = model(X)
        ls = criterion(yprob, y)
        ls.backward()
        optimizer.step()
        loss += ls.item()
    loss /= len(data_loader)
    
    return loss


def evaluate(model, criterion, data_loader, device='cpu'):
    """Evaluation step

    Perform an evaluation step over the entire training data

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    criterion : :obj:`torch.nn.modules.loss`
        Loss function
    data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training dataloader
    device : :obj:`str`, optional
        Device

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    
    """
    model.eval()
    loss = 0
    for X, y in tqdm(data_loader):
        X, y = X.to(device), y.to(device)
        with torch.no_grad(): # use no_grad to avoid making the computational graph...
            yprob = model(X)
            ls = criterion(yprob, y)
        loss += ls.item()
    loss /= len(data_loader)
    return loss