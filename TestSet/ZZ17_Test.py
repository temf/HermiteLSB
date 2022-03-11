# -*- coding: utf-8 -*-
"""

Bazaraa-Shetty Function d=2

Source:
TU Berlin, NL Opt. Skript WiSe 05/06, Dietmar Hömberg

Optimal Solution:
x = (2,1), f(x) = 0

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
    term1 = (x1-2)**4
    term2 =  (x1 - 2*x2)**2
    f = term1 + term2
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    x1=x[0]
    x2=x[1]
    if component == 0:
        term1 = 4*(x1-2)**3
        term2 =  2*(x1 - 2*x2)
        deriv = term1 + term2
    elif component == 1:
        term1 = 0.
        term2 =  2*(x1 - 2*x2)*(-2)
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
x0 = np.array([2.2,1.2,])
n = len(x0)

# Define bounds (required for global optimization)
x_lb = np.array([1.6, 0.8])
x_ub = np.array([2.6, 1.3])

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500


