# -*- coding: utf-8 -*-
"""

Rotated Hyper-Ellipsoid Function d = 5

Source:
https://www.sfu.ca/~ssurjano/rothyp.html

Optimal solution:
x=(0,...,0), f(x) = 0 

"""

from __future__ import print_function
import numpy as np
from numdifftools import Jacobian

dim = 5 # arbitrary dimensions possible

# Define objective function and derivatives in one
def objfunc_with_derivs(x, known_derivs=[], return_all = False):
    f = objfunc(x)
    derivs_list = np.inf*np.ones((len(x)))
    
    if return_all == False:
        return f
    else:
        if known_derivs == []:
            return f, derivs_list
        else:
            for i in range(len(known_derivs)):
                component = known_derivs[i]
                derivs_list[component] = objfun_derivs(x,component)
            return f, derivs_list

# Define objective function
def objfunc(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    fac = np.arange(n+1)[::-1][:n]
    return sum(fac*x**2)

# Define gradient function
def objfun_derivs(x,component = 0):
    x = np.asarray_chkfinite(x)
    i = component
    n = len(x)
    deriv = 2*(n-i)*x[i]
    return deriv

# Define gradient function for SQP using finite differences for unknown derivatives
def calc_deriv_sqp(x, known_derivs_for_sqp,findiff_method='forward'):
    func, deriv = objfunc_with_derivs(x, known_derivs=known_derivs_for_sqp, return_all = True)
    jac = Jacobian(objfunc_with_derivs,method=findiff_method)(x)[0]
    for i in range(len(deriv)):
        deriv_i = deriv[i]
        if deriv_i == np.inf:
            deriv[i] = jac[i]
    return deriv

# Define Starting Point
n = dim
x0 = np.ones(n)

# Define bounds (required for global optimization)
x_lb = -65.536 * np.ones(n)
x_ub = 65.536 * np.ones(n)

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500


