# -*- coding: utf-8 -*-
"""

Sum2 Function d=10

Source:
https://gist.github.com/denis-bz/da697d8bc74fae4598bf

"""

from __future__ import print_function
import numpy as np
from numdifftools import Jacobian

dim = 10

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
    j = np.arange( 1., n+1 )
    f = sum( j * x**2 )
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    deriv = j[component] * 2 * x[component]
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
x_lb = -10. * np.ones(n)
x_ub = 10. * np.ones(n)

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500


