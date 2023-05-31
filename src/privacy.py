import random
import string
import torch
import math
from utils import l1norm_clipped, transforming, l2norm_clipped, transforming_v2, transform_back


def AddLaplaceNoise(grads: torch.Tensor, sens: float, epsilon: float) -> torch.Tensor:
    noise = torch.distributions.laplace.Laplace(loc=0, scale=sens / epsilon)
    new_grads = grads.add(noise.sample())
    return new_grads


def AddGaussianNoise(grads: torch.Tensor, sens: float, epsilon: float, delta: float) -> torch.Tensor:
    sigma = torch.sqrt(torch.tensor(2 * torch.log(torch.tensor(1.25 / delta, dtype=torch.float32)))) * sens / epsilon
    if sigma <= 0: sigma = 0.0001
    noise = torch.distributions.Normal(loc=0, scale=sigma)
    new_grads = grads.add(noise.sample())
    return new_grads


def one_gaussian(epsilon, delta, sensitivity):
    sigma = (sensitivity/epsilon) * math.sqrt(2 * math.log(1.25/delta))
    return torch.distributions.Normal(loc=0, scale=sigma).sample()


def one_laplace(epsilon, sensitivity):
    return torch.distributions.Laplace(loc=0, scale=sensitivity/epsilon).sample()


def randomizer(grads: torch.Tensor, epsilon: float, delta: float, clip_c: float, left: float, right: float, mechanism) -> torch.Tensor:
    # min(right - left, 2 * clip_c)
    # perturbed = torch.rand_like(grads)
    left = max(left, -clip_c)
    right = min(right, clip_c)
    sensitivity = right - left
    if mechanism == 'laplace':
        clip_grads = l1norm_clipped(updates=grads, clip_c=clip_c)
        # norm_grads = transforming(clip_grads, left, right, -clip_c, clip_c)
        # norm_grads = transforming(clip_grads, -clip_c, clip_c, left, right)
        norm_grads = transforming_v2(clip_grads, clip_c)
        sensitivity = 1
        perturbed = AddLaplaceNoise(norm_grads, sensitivity, epsilon)
        # perturbed = transforming(perturbed, 0, 1, left, right)
        perturbed = transform_back(perturbed, clip_c)

    elif mechanism == 'gaussian':
        clip_grads = l2norm_clipped(updates=grads, threshold=clip_c)
        # norm_grads = transforming(clip_grads, left, right, -clip_c, clip_c)
        # norm_grads = transforming(clip_grads, -clip_c, clip_c, left, right)
        norm_grads = transforming_v2(clip_grads, clip_c)
        sensitivity = 1
        perturbed = AddGaussianNoise(norm_grads, sensitivity, epsilon, delta)
        perturbed = transform_back(perturbed, clip_c)
    else:
        perturbed = grads
    return perturbed


def sample_randomizer(vector:torch.Tensor, choice:list, epsilon: float, delta: float, clip_c: float,
                      left: float, right: float, mechanism) -> torch.Tensor:
    clip_grads = l1norm_clipped(updates=vector, clip_c=clip_c)
    sens = 2 * clip_c
    for idx, value in enumerate(vector):
        if idx in choice:
            norm_vec = clip_grads[idx]
            # norm_vec = transforming(clip_grads[idx], -clip_c, clip_c, left, right)
            if mechanism == 'laplace':
                vector[idx] = norm_vec + one_laplace(epsilon, sens)
            elif mechanism == 'gaussian':
                vector[idx] = norm_vec + one_gaussian(epsilon, delta, sens)
        else:
            vector[idx] = 0
    return vector



