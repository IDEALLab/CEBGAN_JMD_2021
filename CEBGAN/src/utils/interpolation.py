"""
B-spline approximation.

Author(s): Wei Chen (wchen459@umd.edu), Qiuyi Chen (qchen88@umd.edu)

Reference(s): 
    [1] Lepine, Jerome, Guibault, Francois, Trepanier, Jean-Yves, Pepin, Francois. (2001). 
        Optimized nonuniform rational B-spline geometrical representation for aerodynamic 
        design of wings. AIAA journal, 39(11), 2033-2041.
    [2] Lepine, J., Trepanier, J. Y., & Pepin, F. (2000, January). Wing aerodynamic design 
        using an optimized NURBS geometrical representation. In 38th Aerospace Sciences 
        Meeting and Exhibit (p. 669).

n+1 : number of control points
m+1 : number of data points
"""

import numpy as np
from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz


def interpolate(Q, N, k, D=20, resolution=1000):
    r"""Interpolate N points whose concentration is based on curvature. 

    Args:
        Q: Existing points to be interpolated.
        N: Number of data points to produce.
        k: Degree of spline.
        D: Shifting constant. The higher the more uniform the data points are.
    """
    tck, u = splprep(Q, u=None, k=k, s=1e-6, per=0, full_output=0)
    uu = np.linspace(u.min(), u.max(), resolution)
    x, y = splev(uu, tck, der=0)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    cv = np.abs(ddx*dy - dx*ddy)/(dx*dx + dy*dy)**1.5 + D
    cv_int = cumtrapz(cv, uu, initial=0)
    fcv = interp1d(cv_int, uu)
    cv_int_samples = np.linspace(0, cv_int.max(), N)
    u_new = fcv(cv_int_samples)
    x_new, y_new = splev(u_new, tck, der=0)
    xy_new = np.vstack((x_new, y_new))
    return xy_new