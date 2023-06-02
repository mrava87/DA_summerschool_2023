import numpy as np
import torch


def forward(x, Op):
    """Forward
    
    Applies the operator 
    
    :param x: Tensor, model estimate of 
    :param Op: nn.Module, Operator
    :return: y
    """
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i] = (Op @ x[i].ravel()).reshape(x.shape[1:])
    return y


def gradient(x, Op, y):
    """Gradient
    
    Applies the gradient of the l2 norm of the residual 
    
    :param x: Tensor, model estimate of 
    :param Op: nn.Module, Operator
    :param y: Tensor, noisy measurements
    :return: grad_x
    """
    # True during training, False during testing
    does_require_grad = x.requires_grad

    with torch.enable_grad():
        # Necessary for using autograd
        x.requires_grad_(True)
        ypred = Op.apply(x)
        # L2 Loss
        loss = 0.5*torch.sum((y - ypred)**2)
        # We retain the graph during training only
        grad_x = torch.autograd.grad(loss, inputs=x, retain_graph=does_require_grad,
                                     create_graph=does_require_grad)[0]
    # Set requires_grad back to it's original state
    x.requires_grad_(does_require_grad)

    return grad_x