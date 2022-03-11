# -*- coding: utf-8 -*-
"""

Dixon Function (modified) d = 10


"""

from __future__ import print_function
import numpy as np
from numdifftools import Jacobian

dim = 10
# Optimal solution at x = (1,...,1) with f(x) = 0

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
    term1 = (1-x[0])**2
    term2 = (1-x[dim-1])**2
    term3 = 0
    for i in range(len(x)-1):
        term3 = term3 + (x[i]**2 - x[i+1])**2
    
    f = term1 + term2 + term3
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    i = component
    if component == 0:
        term1 = -2. + 2*x[0]
        term2 = 0.
        term3 = 4*x[0]**3 - 4*x[0]*x[1]
        deriv = term1 + term2 + term3
    elif component == dim-1:
        term1 = 0.
        term2 = -2 + 2*x[dim-1]
        term3 = -2*x[dim-2]**2 + 2*x[dim-1]
        deriv = term1 + term2 + term3
    else:
        term1 = 0.
        term2 = 0.
        term3 = -2*x[i-1]**2 + 2*x[i] + 4*x[i]**3 - 4*x[i+1]*x[i]
        deriv = term1 + term2 + term3
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
x0 = (-1)*np.ones(n)

# Define bounds (required for global optimization)
x_lb = -5. * np.ones(n)
x_ub = 5. * np.ones(n)

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 2000


