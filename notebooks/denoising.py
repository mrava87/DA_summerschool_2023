import torch
from math import sqrt
import numpy as np
import torch


def denoising(model, img, mu, sigma, half=False, device="cpu", verb=False):
    """Denoising with NN

    Denoise an input 2d dataset with a non-blind denoiser

    Parameters
    ----------
    model : :obj:`torch.nn`
        NN denoiser
    img : :obj:`np.ndarray`
        2D input to denoise
    mu : :obj:`float`
        Step-size of proximal of denoiser. Together with ``sigma`` is
        used to create the noise standard deviation to add as extra input channel
        to the denoiser
    sigma : :obj:`float`
        Regularization factor of denoiser. Together with ``mu`` is
        used to create the noise standard deviation to add as extra input channel
        to the denoiser
    half : :obj:`bool`, optional
        Apply model in half precision
    device : :obj:`str`, optional
        Device to use
    verb : :obj:`bool`, optional
        Verbosity

    Returns
    -------
    denoise_img : obj:`np.ndarray`
        2D denoised output

    """
    # If input is all zeros, simply return it as is
    if img.min() == 0 and img.min() == 0:
        return img.cpu().detach().numpy()

    # Shifting and Scaling to bring the input between 0 and 1 as required by the denoiser
    if verb: print(f'img, min{img.min()}, max{img.max()}')

    img_min = img.min()
    numerator = img - img_min
    scaled_img = numerator/numerator.max()
    if verb: print(f'scaled_img, min{scaled_img.min()}, max{scaled_img.max()}')

    # Padding
    extra_left, extra_right = 0, 1024-img.shape[1]
    extra_top, extra_bottom = 0, 1024-img.shape[0]
    
    scaled_img = np.pad(scaled_img, ((extra_top, extra_bottom), (extra_left, extra_right)),
                        mode='constant', constant_values=0)
        
    # Create sigma map
    sigma = sqrt(sigma * mu)
    sigmamap = torch.tensor(sigma, dtype=torch.float).repeat(1, 1, scaled_img.shape[0], scaled_img.shape[1])
    scaled_img = torch.cat((torch.from_numpy(scaled_img).unsqueeze(0).unsqueeze(0), sigmamap), dim=1)

    # Denoise
    if half:
        scaled_img = scaled_img.half()
    with torch.no_grad():
        denoise_img = model(scaled_img.to(device))
    if half:
        denoise_img = denoise_img.float()
    denoise_img = denoise_img[0, 0, :img.shape[0], :img.shape[1]]

    # Renormalize back
    denoise_img = denoise_img * (numerator.max()) + img_min
    if verb: print(f'Denoise_img, min{denoise_img.min()}, max{denoise_img.max()}')
    return denoise_img.cpu().detach().numpy()
