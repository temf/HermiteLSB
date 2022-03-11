# -*- coding: utf-8 -*-
"""

Sphere Function d=4

Source:
https://www.sfu.ca/~ssurjano/spheref.html

Optimal Solution:
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
    d = len(x)
    sumup = 0
    for i in range(d):
        xi = x[i]
        sumup = sumup + xi**2

    f = sumup;
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    deriv = 2*x[component]
    
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
x0 = np.array([1.,1.,1.,1.])
n = len(x0)

# Define bounds (required for global optimization)
x_lb = np.array([-5.12, -5.12, -5.12, -5.12])
x_ub = np.array([5.12, 5.12, 5.12, 5.12])

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500


