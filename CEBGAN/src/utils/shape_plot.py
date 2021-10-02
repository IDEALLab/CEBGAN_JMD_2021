"""
Plots samples or new shapes in the semantic space.

Author(s): Wei Chen (wchen459@umd.edu), Jonah Chazan (jchazan@umd.edu)
"""
import itertools
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

def gen_grid(d, points_per_axis, lb=0., rb=1.):
    ''' Generate a grid in a d-dimensional space 
        within the range [lb, rb] for each axis '''
    
    lincoords = []
    for i in range(0, d):
        lincoords.append(np.linspace(lb, rb, points_per_axis))
    coords = list(itertools.product(*lincoords))
    
    return np.array(coords)

def plot_shape(xys, z1, z2, ax, scale, scatter, symm_axis, **kwargs):
#    mx = max([y for (x, y) in m])
#    mn = min([y for (x, y) in m])
    xscl = scale# / (mx - mn)
    yscl = scale# / (mx - mn)
#    ax.scatter(z1, z2)
    if scatter:
        if 'c' not in kwargs:
            kwargs['c'] = cm.rainbow(np.linspace(0,1,xys.shape[0]))
#        ax.plot( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
        ax.scatter( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), edgecolors='none', **kwargs)
    else:
        ax.plot( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), **kwargs)
        
    if symm_axis == 'y':
#        ax.plot( *zip(*[(-x * xscl + z1, y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
        plt.fill_betweenx( *zip(*[(y * yscl + z2, -x * xscl + z1, x * xscl + z1)
                          for (x, y) in xys]), color='gray', alpha=.2)
    elif symm_axis == 'x':
#        ax.plot( *zip(*[(x * xscl + z1, -y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
        plt.fill_between( *zip(*[(x * xscl + z1, -y * yscl + z2, y * yscl + z2)
                          for (x, y) in xys]), color='gray', alpha=.2)

def plot_samples(Z, X, label=None, scale=0.8, points_per_axis=None, scatter=True, symm_axis=None, fname=None, **kwargs):
    
    ''' Plot shapes given design sapce and latent space coordinates '''
    
    plt.rc("font", size=12)
    if Z is None or Z.shape[1] != 2 or points_per_axis is None:
        N = X.shape[0]
        points_per_axis = int(N**.5)
        bounds = (0., 1.)
        Z = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
        
    scale /= points_per_axis
        
    # Create a 2D plot
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
        
    for (i, z) in enumerate(Z):
        plot_shape(X[i], z[0], .3*z[1], ax, scale, scatter, symm_axis, **kwargs)
        if label is not None:
            text = 'aoa: {:.2f}'.format(label[i])
            plt.annotate(text, xy = (0.3*scale+z[0], .3*z[1]), size=20*scale)
    
#    plt.xlabel('c1')
#    plt.ylabel('c2')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()
    if fname:
        plt.savefig(fname+'.svg', dpi=600)
    else:
        plt.show()
    plt.close()

def plot_comparision(Z, G, Ps, lb_G=None, lb_P=None, scale=0.8, points_per_axis=None, scatter=True, symm_axis=None, fname=None, **kwargs):
    
    ''' Plot comparision between groundtruth and prediction '''
    
    plt.rc("font", size=12)
    if Z is None or Z.shape[1] != 2 or points_per_axis is None:
        N = G.shape[0]
        points_per_axis = int(N**.5)
        bounds = (0., 1.)
        Z = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
        
    scale /= points_per_axis
        
    # Create a 2D plot
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    for (i, z) in enumerate(Z):
        plot_shape(G[i], z[0], .3*z[1], ax, scale, scatter, symm_axis, lw=2, c='k', **kwargs)
        if lb_G is not None:
            label = 'aoa_true: {:.2f}'.format(lb_G[i])
            plt.annotate(label, xy = (0.1*scale+z[0], .3*z[1]), c='grey', size=20*scale)

    for j, P in enumerate(Ps):
        color = list(mcolors.TABLEAU_COLORS.items())[j][1]
        for (i, z) in enumerate(Z):
            plot_shape(P[i], z[0], .3*z[1], ax, scale, scatter, symm_axis, lw=1, c=color, alpha=1, **kwargs)
            if lb_P is not None:
                label = 'aoa_pred: {:.2f}'.format(lb_P[i])
                plt.annotate(label, xy = (0.5*scale+z[0], .3*z[1]), c='salmon', size=20*scale)
    
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.axis('equal')
    plt.tight_layout()
    if fname:
        plt.savefig(fname+'.svg', dpi=600)
    else:
        plt.show()
    plt.close()

def plot_synthesized(Z, gen_func, d=2, scale=.8, points_per_axis=None, scatter=True, symm_axis=None, fname=None, **kwargs):
    
    ''' Synthesize shapes given latent space coordinates and plot them '''
    
    if d == 1:
        latent = Z[:,:1]
    if d == 2:
        latent = Z
    if d >= 3:
        latent = np.random.normal(scale=0.5, size=(Z.shape[0], d))
    X = gen_func(latent)
    
    plot_samples(Z, X, scale, points_per_axis, scatter, symm_axis, fname=fname, **kwargs)

def plot_grid(points_per_axis, gen_func, d=2, bounds=(0.0, 1.0), scale=.8, scatter=True, 
              symm_axis=None, fname=None, **kwargs):
    
    ''' Uniformly plots synthesized shapes in the latent space
        K : number of samples for each point in the latent space '''
    
    if d == 1:
        Z = np.linspace(bounds[0], bounds[1], points_per_axis)
        Z = np.vstack((Z, np.zeros(points_per_axis))).T
        plot_synthesized(Z, gen_func, 1, scale, points_per_axis, scatter, symm_axis, fname, **kwargs)
    if d == 2:
        Z = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid
        plot_synthesized(Z, gen_func, 2, scale, points_per_axis, scatter, symm_axis, fname, **kwargs)
    if d >= 3:
        Z = np.ones((points_per_axis**2, d)) * (bounds[0]+bounds[1])/2 # [N^2, 3]
        zgrid = np.linspace(bounds[0], bounds[1], points_per_axis) # N
        for i in range(points_per_axis):
            Zxy = gen_grid(2, points_per_axis, bounds[0], bounds[1]) # Generate a grid [N^2, 2]
            Zz = np.ones((points_per_axis**2, 1)) * zgrid[i] # [N^2, 1]
            Z[:, :3] = np.hstack((Zxy, Zz)) # zero after 3rd dimension [N^2, 3] = [N^2, 2 + 1]
            plot_synthesized(Z, gen_func, 2, scale, points_per_axis, scatter, symm_axis, '%s_%.2f' % (fname, zgrid[i]), **kwargs)
        
