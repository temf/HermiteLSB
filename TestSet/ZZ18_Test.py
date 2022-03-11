# -*- coding: utf-8 -*-
"""

Test Function d=5

Source:
Beispiel 1.5.5
TU Berlin, NL Opt. Skript WiSe 05/06, Dietmar Hömberg

Optimal Solution:
x = (1,1,1,1,1), f(x) = 0

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
    x3=x[2]
    x4=x[3]
    x5=x[4]
    term1 = 2*x1**2 + 2*x2**2 + x3**2 + x4**2 + 0.5*x5**2
    term2 = -4*(x1+x2)
    term3 = -2*(x3+x4)
    term4 = -x5 + 6.5
    f = term1 + term2 + term3 + term4
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    if component == 0:
        term1 = 4*x1
        term2 = -4.
        term3 = 0.
        term4 = 0.
        deriv = term1 + term2 + term3 + term4
    elif component == 1:
        term1 = 4*x2
        term2 = -4.
        term3 = 0.
        term4 = 0.
        deriv = term1 + term2 + term3 + term4
    elif component == 2:
        term1 = 2*x3
        term2 = 0.
        term3 = -2.
        term4 = 0.
        deriv = term1 + term2 + term3 + term4
    elif component == 3:
        term1 = 2*x4
        term2 = 0.
        term3 = -2.
        term4 = 0.
        deriv = term1 + term2 + term3 + term4
    elif component == 4:
        term1 = x5
        term2 = 0.
        term3 = 0.
        term4 = -1.
        deriv = term1 + term2 + term3 + term4
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
x0 = np.array([0.,0.,0.,0.,0.])
n = len(x0)

# Define bounds (required for global optimization)
x_lb = np.array([-5.,-5.,-5.,-5.,-5.])
x_ub = np.array([5.,5.,5.,5.,5.])

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500

