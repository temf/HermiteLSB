# -*- coding: utf-8 -*-
"""

Test Function d=2

Optimal Solution:
x = (1,3), f(x) = -10

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
    f = x[0]**2+ x[1]**2 - 2*x[0] - 6*x[1]
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    if component == 0:
        deriv = 2*x[0] - 2
    elif component == 1:
        deriv = 2*x[1] - 6
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
x0 = np.array([0., 0.])

# Define bounds (required for global optimization)
x_lb = np.array([-10.0, -10.0])
x_ub = np.array([10., 10.])

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500

# Define number of interpolation points
n = len(x0)


