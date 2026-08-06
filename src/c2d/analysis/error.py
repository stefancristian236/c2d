import numpy as np
from scipy.stats import norm, gennorm, laplace
import matplotlib.pyplot as plt
from pathlib import Path
import os




############################################################################
## performa laplacian fitting
## input an 1D array
## returns the fig, axes for the plot
def laplacian_fitting(arr : np.ndarray[float]):
    
    #### direcotry initiliszation
    ### verify if the directory exists otherwise create it
    dir_path = Path('../../out')
    if dir_path.is_dir() == False:
        os.makedirs('../../out')
    dir_path = Path('../../out/plots')
    if dir_path.is_dir() == False:
        os.makedirs('../../out/plots')
    dir_path = Path('../../out/plots/distributions')
    if dir_path.is_dir() == False:
        os.makedirs('../../out/plots/distributions')
    ##########################################################
    
    #calculate fitting parameters
    lb, ub = laplace.fit(arr)
    
    #generate the x axis
    x = np.linspace(laplace.ppf(0.001, loc = lb, scale = ub),
                    laplace.ppf(0.999, loc = lb, scale = ub),
                    1000
                    )
    
    #generate the statistic momnets
    mean = laplace.mean(loc = lb, scale = ub)
    median = laplace.median(loc = lb, scale = ub)
    std = laplace.std(loc = lb, scale = ub)
    var = laplace.var(loc = lb, scale = ub)
    
    #initiate plotting
    fig, ax = plt.subplots(figsize = (8, 6))
    
    #plot histogram
    ax.hist(arr,
            bins = 'auto',
            density = True,
            alpha = .6,
            color = 'blue',
            label = 'Error Bins'
            )
    
    #plot the pdf overlay
    ax.plot(x,
            laplace.pdf(x, loc = lb, scale = ub),
            'r--',
            lw = 1,
            alpha = .8,
            label = f"Fitted Laplace\nMean: {mean:.4f}\nMedian: {median:.4f}\nStd: {std:.4f}"
            )
    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_ylim([np.min(laplace.pdf(x, loc = lb, scale = ub)), np.max(laplace.pdf(x, loc = lb, scale = ub))])
    ax.set_title('Laplacian Distribution')
    ax.set_xlabel('Error Values')
    ax.set_ylabel('Density')
    ax.grid(alpha = .5)
    ax.legend()
    plt.savefig('../out/plots/distributions/LaplaceDist', dpi = 1000)
    plt.show()
    
    return fig, ax
###############################################################################################


def generalised_gaussian_fitting(arr : np.ndarray[float]) -> tuple[Figure, Axes]:
    
    #### direcotry initiliszation
    ### verify if the directory exists otherwise create it
    dir_path = Path('../../out')
    if dir_path.is_dir() == False:
        os.makedirs('../../out')
    dir_path = Path('../../out/plots')
    if dir_path.is_dir() == False:
        os.makedirs('../../out/plots')
    dir_path = Path('../../out/plots/distributions')
    if dir_path.is_dir() == False:
        os.makedirs('../../out/plots/distributions')
    ##########################################################
    
    #calculate fitting parameters   
    beta, loc, scale = gennorm.fit(arr)
    
    #generate the x axis
    x = np.linspace(
        gennorm.ppf(0.01, beta, loc=loc, scale=scale),
        gennorm.ppf(0.99, beta, loc=loc, scale=scale),
        1000
    )
    
    #calculate the statistical moments
    mean = gennorm.mean(beta, loc=loc, scale=scale)
    median = gennorm.median(beta, loc=loc, scale=scale)
    std = gennorm.std(beta, loc=loc, scale=scale)
    var = gennorm.var(beta, loc=loc, scale=scale)
    
    #initiate plotting
    fig, ax = plt.subplots(figsize = (8, 6))
    
    #plot histogram
    ax.hist(arr,
            bins = 'auto',
            density = True,
            alpha = .6,
            color = 'blue',
            label = 'Error Bins'
            )
    
    #plot the pdf overlay
    ax.plot(x,
            gennorm.pdf(x, beta, loc = loc, scale = scale),
            'r--',
            lw = 1,
            alpha = .8,
            label = f"Fitted Gaussian\nBeta: {beta:.4f}\nMean: {mean:.4f}\nMedian: {median:.4f}\nStd: {std:.4f}"
            )
    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_ylim([np.min(laplace.pdf(x, loc = loc, scale = scale)), np.max(laplace.pdf(x, loc = loc, scale = scale))])
    ax.set_title('Generalised Gaussian Distribution')
    ax.set_xlabel('Error Values')
    ax.set_ylabel('Density')
    ax.grid(alpha = .5)
    ax.legend()
    plt.savefig('../out/plots/distributions/GenGaussianDist', dpi = 1000)
    plt.show()
    
    return fig, ax
###########################################################################################

def normal_fitting(arr : np.ndarray[float]):
        
    #### direcotry initiliszation
    ### verify if the directory exists otherwise create it
    dir_path = Path('../../out')
    if dir_path.is_dir() == False:
        os.makedirs('../../out')
    dir_path = Path('../../out/plots')
    if dir_path.is_dir() == False:
        os.makedirs('../../out/plots')
    dir_path = Path('../../out/plots/distributions')
    if dir_path.is_dir() == False:
        os.makedirs('../../out/plots/distributions')
    ##########################################################
    
    #calculate fitting parameters   
    loc, scale = norm.fit(arr)
    
    #generate the x axis
    x = np.linspace(
        norm.ppf(0.01, loc=loc, scale=scale),
        norm.ppf(0.99, loc=loc, scale=scale),
        1000
    )
    
    #calculate the statistical moments
    mean = norm.mean(loc=loc, scale=scale)
    median = norm.median(loc=loc, scale=scale)
    std = norm.std(loc=loc, scale=scale)
    var = norm.var(loc=loc, scale=scale)
    
    #initiate plotting
    fig, ax = plt.subplots(figsize = (8, 6))
    
    #plot histogram
    ax.hist(arr,
            bins = 'auto',
            density = True,
            alpha = .6,
            color = 'blue',
            label = 'Error Bins'
            )
    
    #plot the pdf overlay
    ax.plot(x,
            norm.pdf(x, loc = loc, scale = scale),
            'r--',
            lw = 1,
            alpha = .8,
            label = f"Fitted Normal Distribution\nMean: {mean:.4f}\nMedian: {median:.4f}\nStd: {std:.4f}"
            )
    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_ylim([np.min(laplace.pdf(x, loc = loc, scale = scale)), np.max(laplace.pdf(x, loc = loc, scale = scale))])
    ax.set_title('Normal Distribution')
    ax.set_xlabel('Error Values')
    ax.set_ylabel('Density')
    ax.grid(alpha = .5)
    ax.legend()
    plt.savefig('../out/plots/distributions/NormalDist', dpi = 1000)
    plt.show()
    
    return fig, ax
###########################################################################################
