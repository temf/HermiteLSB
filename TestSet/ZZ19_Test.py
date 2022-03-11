# -*- coding: utf-8 -*-
"""

Test Function (modified) d=2

Source:
https://www.unibw.de/ingmathe/teaching/optimierung/wt2021/folien_optimierung_vl_3_2020.pdf/view

Optimal solution:
x = (-0.667,0), f(x) = -0.148

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
    term1 = -(x2**2)*(x1-1)
    term2 = -(x1**2)*(x1+1)
    f = term1 + term2
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    x1=x[0]
    x2=x[1]
    if component == 0:
        term1 = -x2**2
        term2 = -x1*(3*x1+2)
        deriv = term1 + term2
    elif component == 1:
        term1 = -2*x2*(x1-1)
        term2 = 0.
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
x0 = np.array([-0.5,0.2])
n = len(x0)

# Define bounds (required for global optimization)
x_lb = np.array([-1.,-0.5])
x_ub = np.array([.5,.5])

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500


