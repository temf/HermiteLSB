# -*- coding: utf-8 -*-
"""

Colville Function d=4

Source:
https://www.sfu.ca/~ssurjano/colville.html

Optimal solution:
x = ~(1,1,1,1), f(x) = 0

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
    x1 = x[0];
    x2 = x[1];
    x3 = x[2];
    x4 = x[3];
    
    term1 = 100 * (x1**2-x2)**2;
    term2 = (x1-1)**2;
    term3 = (x3-1)**2;
    term4 = 90 * (x3**2-x4)**2;
    term5 = 10.1 * ((x2-1)**2 + (x4-1)**2);
    term6 = 19.8*(x2-1)*(x4-1);
    f = term1 + term2 + term3 + term4 + term5 + term6;
    return f

# Define gradient function
def objfun_derivs(x,component = 0):
    x1 = x[0];
    x2 = x[1];
    x3 = x[2];
    x4 = x[3];
    if component == 0:
        term1 = 100 * 2 * (x1**2-x2) * 2*x1;
        term2 = 2*(x1-1);
        term3 = 0.
        term4 = 0.
        term5 = 0.
        term6 = 0.
        deriv = term1 + term2 + term3 + term4 + term5 + term6;
    elif component == 1:
        term1 = 100 * 2 * (x1**2-x2) * (-1)
        term2 = 0.
        term3 = 0.
        term4 = 0.
        term5 = 10.1 * 2 * (x2-1);
        term6 = 19.8*(x4-1);
        deriv = term1 + term2 + term3 + term4 + term5 + term6;
    elif component == 2:
        term1 = 0.
        term2 = 0.
        term3 = 2*(x3-1)
        term4 = 90 * 2 * (x3**2-x4) * 2*x3;
        term5 = 0.
        term6 = 0.
        deriv = term1 + term2 + term3 + term4 + term5 + term6;
    elif component == 3:
        term1 = 0.
        term2 = 0.
        term3 = 0.
        term4 = 90 * 2 * (x3**2-x4) * (-1);
        term5 = 10.1 * 2 * (x4-1);
        term6 = 19.8*(x2-1);
        deriv = term1 + term2 + term3 + term4 + term5 + term6;
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
x0 = np.zeros(4)
n = len(x0)

# Define bounds (required for global optimization)
x_lb = -10 * np.ones(n)
x_ub = 10 * np.ones(n)

# Set random seed (for reproducibility)
np.random.seed(0)

# Define maximum number of function evaluations
max_fcteval = 500


