# -*- coding: utf-8 -*-
"""

Generalized Quartic Function d=10

Source:
https://gist.github.com/denis-bz/da697d8bc74fae4598bf

Optimal solution:
x = (0,...,0), f(x) = 0

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
    f = 0
    for i in range(dim-1):
        f = f + x[i]**2 + (x[i+1] + x[i]**2)**2
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    i = component
    if component == 0:
        deriv = 2*x[0] + 4*x[1]*x[0] + 4*x[0]**3
    elif component == dim-1:
        deriv = 2*x[dim-1] + 2*x[dim-2]**2
    else:
        term1 = 2*x[i] + 4*x[i+1]*x[i] + 4*x[i]**3
        term2 = 2*x[i] + 2*x[i-1]**2
        deriv = term1 + term2
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
x_lb = -5. * np.ones(n)
x_ub = 5. * np.ones(n)

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 2000

