# -*- coding: utf-8 -*-
"""

Styblinski-Tang Function d=3

Source:
https://www.sfu.ca/~ssurjano/stybtang.html

Optimal solution:
x = ~(-2.9,...,-2.9), f(x) = ~-39.16599d

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
    d = len(x);
    sumup = 0;
    for i in range(d):
        xi = x[i];
        new = xi**4 - 16*xi**2 + 5*xi;
        sumup  = sumup + new;
    f = sumup/2;
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    i = component
    deriv = 2*x[i]**3-16*x[i]+2.5
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
x0 = np.zeros(3)
n = len(x0)

# Define bounds (required for global optimization)
x_lb = -5 * np.ones(n)
x_ub = 5 * np.ones(n)

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500


