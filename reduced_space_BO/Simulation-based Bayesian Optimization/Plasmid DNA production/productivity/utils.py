# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import torch

from matplotlib import pyplot as plt


def plot_convergence(X, y, maximize=False):
    """
    Plot convergence history: distance between consecutive x's and value of
    the best selected sample
    Parameters
    ----------
    X : torch.tensor, shape=(n_samples, dim)
        History of evaluated input values
    y : torch.tensor, shape=(n_samples,)
        History of evaluated objective values
    Returns
    -------
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    dist = torch.norm(X[1:] - X[:-1], dim=-1).cpu().numpy()
    if maximize:
        cum_best = np.maximum.accumulate(y.cpu().numpy())
    else:
        cum_best = np.minimum.accumulate(y.cpu().numpy())

    
    def ci(y):
        covar = 1.96 * y.std(axis=0) / np.sqrt(y.shape[0])
        
        return covar

    #print(cum_best.std(axis=0))

    axes[0].plot(dist, '.-', c='r',)
    axes[0].set_xlabel('Iteration', fontsize=14)
    axes[0].set_ylabel(r"$d(x_i - x_{i - 1})$", fontsize=14)
    axes[0].set_title("Distance between consecutive x's", fontsize=14)
    axes[0].grid(True)


    axes[1].plot(cum_best, '.-')
    axes[1].set_xlabel('Iteration', fontsize=14)
    axes[1].set_ylabel('Best y', fontsize=14)
    axes[1].set_title('Value of the best selected sample', fontsize=14)
    axes[1].grid(True)
 
    

    #axes[1].errorbar(list(range(15+1)),cum_best, yerr = ci(cum_best))
                 

    fig.tight_layout()
    plt.savefig('convergence.jpeg', dpi=400)