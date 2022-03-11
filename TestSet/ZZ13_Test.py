# -*- coding: utf-8 -*-
"""
Test Function d=3

Source:
https://medium.com/@rhome/convex-optimization-unconstrained-836a44182f9d

Optimal Solution:
x = (0,0), f(x) = 3

"""

from __future__ import print_function
import numpy as np
from numdifftools import Jacobian


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
    x1=x[0]
    x2=x[1]
    f = x1**2 + 2*x2**2 +3;
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    x1=x[0]
    x2=x[1]
    if component == 0:
        deriv = 2*x1
    elif component == 1:
        deriv = 4*x2
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
x0 = np.array([1.,2.,])
n = len(x0)

# Define bounds (required for global optimization)
x_lb = np.array([-10., -10.])
x_ub = np.array([10., 10.])

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500


