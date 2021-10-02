"""
Author(s): Wei Chen (wchen459@umd.edu)
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

##########################
########## utils #########
##########################

def mean_err(metric_list):
    n = len(metric_list)
    mean = np.mean(metric_list)
    std = np.std(metric_list)
    err = 1.96*std/n**.5 # standard error, 1.96 is the approximate value of the 97.5 percentile point of the normal distribution
    return mean, err

def optimize_kde(X):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(-3, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, n_jobs=8, cv=5, verbose=1)
    grid.fit(X)
    
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    
    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_
    
    return kde


##########################
########### MMD ##########
##########################

def gaussian_kernel(X, Y, sigma=1.0):
    beta = 1. / (2. * sigma**2)
    dist = pairwise_distances(X, Y)
    s = beta * dist.flatten()
    return np.exp(-s)

def maximum_mean_discrepancy(gen_func, X_test):
    
    X_gen = gen_func() #gen_func randomly generate samples.
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
      
    mmd = np.mean(gaussian_kernel(X_gen, X_gen)) - \
            2 * np.mean(gaussian_kernel(X_gen, X_test)) + \
            np.mean(gaussian_kernel(X_test, X_test))
            
    return np.sqrt(mmd)
    
def ci_mmd(n, gen_func, X_test):
    mmds = np.zeros(n)
    for i in range(n):
        mmds[i] = maximum_mean_discrepancy(gen_func, np.squeeze(X_test))
    return mean_err(mmds)


##########################
########### LSC ##########
##########################

def sample_line(d, m, bounds):
    # Sample m points along a line parallel to a d-dimensional space's basis
    basis = np.random.choice(d) 
    c = np.zeros((m, d))
    c[:,:] = np.random.rand(d) # sample an arbitrary random vector
    c[:,basis] = np.linspace(0.0, 1.0, m) # sample points along one direction of that random vector
    c = bounds[0] + (bounds[1]-bounds[0])*c
    return c

def consistency(gen_func, latent_dim, bounds):
    
    n_eval = 100 # number of lines to be evaluated
    n_points = 50 #number of points sampled on each line
    mean_cor = 0 # sum of the correlation on each line
    
    for i in range(n_eval):
        
        c = sample_line(latent_dim, n_points, bounds)
        dist_c = np.linalg.norm(c - c[0], axis=1)
        
        X = gen_func(c)
        X = X.reshape((n_points, -1))
        dist_X = np.linalg.norm(X - X[0], axis=1)
        
        mean_cor += np.corrcoef(dist_c, dist_X)[0,1]

    return mean_cor/n_eval

def ci_cons(n, gen_func, latent_dim=2, bounds=(0.0, 1.0)):
    conss = np.zeros(n)
    for i in range(n):
        conss[i] = consistency(gen_func, latent_dim, bounds)
    return mean_err(conss)


##########################
########### RVOD #########
##########################

def variation(X):
    var = 0
    for x in X:
        diff = np.diff(x, axis=0)
        cov = np.cov(diff.T)
        var += np.trace(cov)/cov.shape[0]
    return var/X.shape[0]
    
def ci_rsmth(n, gen_func, X_test):
    rsmth = np.zeros(n)
    for i in range(n):
        X_gen = gen_func(2000)
        rsmth[i] = variation(np.squeeze(X_test))/variation(X_gen)
    return mean_err(rsmth)


##########################
########### MLL ##########
##########################

def mean_log_likelihood(X_gen, X_test):
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    kde = optimize_kde(X_gen)
    return kde.score(X_test) / X_test.shape[0]
    
def ci_mll(n, gen_func, X_test):
    mlls = np.zeros(n)
    for i in range(n):
        X_gen = gen_func(2000)
        mlls[i] = mean_log_likelihood(X_gen, np.squeeze(X_test))
    return mean_err(mlls)


##########################
######## Diversity #######
##########################
def variance(X):
    cov = np.cov(X.T)
    var = np.trace(cov)/cov.shape[0]
#    var = np.mean(np.var(X, axis=0))
#    var = np.linalg.det(cov)
#    var = var**(1./cov.shape[0])
    return var

def rdiv(X_train, X_gen):
    ''' Relative div '''
    X_train = np.squeeze(X_train)
#    train_div = np.sum(np.var(X_train, axis=0))
#    gen_div = np.sum(np.var(X_gen, axis=0))
    X_train = X_train.reshape((X_train.shape[0], -1))
    train_div = variance(X_train)
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    gen_div = variance(X_gen)
#    n = 100
#    gen_div = train_div = 0
#    for i in range(n):
#        a, b = np.random.choice(X_gen.shape[0], 2, replace=False)
#        gen_div += np.linalg.norm(X_gen[a] - X_gen[b])
#        c, d = np.random.choice(X_train.shape[0], 2, replace=False)
#        train_div += np.linalg.norm(X_train[c] - X_train[d])
    rdiv = gen_div/train_div
    return rdiv

def ci_rdiv(n, X_train, gen_func, d=None, k=None, bounds=None):
    rdivs = np.zeros(n)
    for i in range(n):
        if d is None or k is None or bounds is None:
            X_gen = gen_func(X_train.shape[0])
        else:
            latent = np.random.uniform(bounds[0], bounds[1])*np.ones((X_train.shape[0], d))
            latent[:, k] = np.random.uniform(bounds[0], bounds[1], size=X_train.shape[0])
            X_gen = gen_func(latent)
#            from shape_plot import plot_samples
#            plot_samples(None, X_gen[:10], scatter=True, s=1, alpha=.7, c='k', fname='gen_%d' % k)
        rdivs[i] = rdiv(X_train, X_gen)
    return mean_err(rdivs)