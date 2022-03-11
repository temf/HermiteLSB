# -*- coding: utf-8 -*-
"""

Powell Function d=4

Source:
https://www.sfu.ca/~ssurjano/powell.html
http://www.optimization-online.org/DB_FILE/2012/03/3382.pdf

Optimal solution:
x = (0,0,0,0), f(x) = 0

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
    term1 = (x[0] + 10*x[1])**2;
    term2 = 5 * (x[2] - x[3])**2;
    term3 = (x[1] - 2*x[2])**4;
    term4 = 10 * (x[0] - x[3])**4;
    f = term1 + term2 + term3 + term4;
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    if component == 0:
        term1 = 2*x[0] + 20*x[1]
        term2 = 0.
        term3 = 0.
        term4 = 40 * (x[0] - x[3])**3;
        deriv = term1 + term2 + term3 + term4;
    elif component == 1:
        term1 = 20*x[0] + 200*x[1]
        term2 = 0.
        term3 = 4 * (x[1] - 2*x[2])**3
        term4 = 0.
        deriv = term1 + term2 + term3 + term4;
    elif component == 2:
        term1 = 0.
        term2 = 10 * (x[2] - x[3])
        term3 = 4 * (x[1] - 2*x[2])**3 * (-2)
        term4 = 0.
        deriv = term1 + term2 + term3 + term4;
    elif component == 3:
        term1 = 0.
        term2 = -10 * (x[2] - x[3])
        term3 = 0.
        term4 = -40 * (x[0] - x[3])**3;
        deriv = term1 + term2 + term3 + term4;
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
x0 = np.array([3,-1,0,1])
n = len(x0)

# Define bounds (required for global optimization)
x_lb = -4 * np.ones(n)
x_ub = 5 * np.ones(n)

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500


