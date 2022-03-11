# -*- coding: utf-8 -*-
"""

Trid Function d = 5

Source:
https://gist.github.com/denis-bz/da697d8bc74fae4598bf

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
    return sum( (x - 1) **2 ) - sum( x[:-1] * x[1:] )

# Define gradient function
def objfun_derivs(x,component = 0):
    x = np.asarray_chkfinite(x)
    d = len(x)
    if component == 0:
        deriv = 2*x[0] -2 -x[1]
    elif component == d-1:
        deriv = 2*x[d-1] -2 -x[d-2]
    else:
        i = component
        deriv = 2*x[i] -2 -x[i-1] -x[i+1]
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


### For tests with noisy data ###
# Define objective function and derivatives in one with noise
def objfunc_with_derivs_noisy(x, known_derivs=[], return_all = False):
    noise = (1.0 + 1e-2 * np.random.normal(size=(1,))[0])
    f = objfunc(x) * noise
    derivs_list = np.inf*np.ones((len(x)))
    
    if return_all == False:
        return f
    else:
        if known_derivs == []:
            return f, derivs_list
        else:
            for i in range(len(known_derivs)):
                component = known_derivs[i]
                derivs_list[component] = objfun_derivs(x,component) * noise
            return f, derivs_list


# Define gradient function for SQP using finite differences for unknown derivatives
def calc_deriv_sqp_noisy(x, known_derivs_for_sqp,findiff_method='forward'):
    func, deriv = objfunc_with_derivs_noisy(x, known_derivs=known_derivs_for_sqp, return_all = True)
    jac = Jacobian(objfunc_with_derivs_noisy,method=findiff_method)(x)[0]
    for i in range(len(deriv)):
        deriv_i = deriv[i]
        if deriv_i == np.inf:
            deriv[i] = jac[i]
    return deriv



# Define Starting Point
n = dim
x0 = np.zeros(n)

# Define bounds (required for global optimization)
x_lb = -n**2 * np.ones(n)
x_ub = n**2 * np.ones(n)

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500


