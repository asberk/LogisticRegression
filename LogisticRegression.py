#! /usr/bin/python
## performs logistic regression using gradient descent

import numpy as np
from numpy.random import *
from pandas import * 

def logistRg(**kwargs):
    ### LOGISTRG: Logistic Regression Classifier
    ## Binary classifier, performs logistic regression on a training dataset; 
    ## Returns intercept beta0 and slope beta of logistic function 
    ##        f(x) = 1/(1 + exp(-beta0 - dot(beta,x)))
    ### Possible input arguments are:
    ##      data : pandas DataFrame object with y_j values in a column "y"
    ##         x : an ndarray or DataFrame object with #rows = #training samples
    ##         y : a list, ndarray or DataFrame object corresponding with x
    ##      func : type of regression (e.g. logistic)
    ##     alpha : a scalar; the learning rate (default .1)
    ##       eps : convergence parameter (default 1e-4)
    ##  max_iter : maximum number of iterations (in case convergence is slow or fails)
    ## initGuess : initial guess for beta0, beta

    ## Possibly something to think about using:
    ## for key, value in kwargs.iteritems():
    ##     exec key + "=" + repr(value)
    
    if kwargs.has_key('data'):
        y = data['y']
        ## y = data.pop('y')
        x = data[[j for j in data.columns if j[0]=='x']]
        ## x = data.copy()
    elif kwargs.has_key('x') and kwargs.has_key('y'):
        ## For now, assume x data are organized x1, x2, x3, etc. 
        ## Maybe their organization won't even matter.
        if not (y==1 | y==-1).all():
            raise ValueError('all y values (for now) must be in the range -1, ..., 1')
    else:
        raise ValueError('input data not recognized')
    
    xcols = list(x.columns)
    x = np.array(x) # coerce to array type
    y = np.array(y) # coerce to array type
    m = x.shape[0] # number of training samples
    n = x.shape[1] # dimension of sample data
    
    if m < 2:
        raise ValueError('not enough training samples')
    
    ## initial guess for parameters
    if kwargs.has_key('beta0'):
        beta0 = kwargs['beta0']
    else:
        beta0 = 0
    
    if kwargs.has_key('beta'):
        beta = kwargs['beta']
    else:
        beta = [0,0,0]
    
    
    ## learning rate
    if kwargs.has_key('alpha'):
        alpha = kwargs['alpha']
    else:
        alpha = .1
        
    if kwargs.has_key('eps'):
        eps = kwargs['eps']
    else:
        eps = 1e-4
    
    if kwargs.has_key('max_iter'):
        max_iter = kwargs['max_iter']
    else:
        max_iter = 1000
    
    if kwargs.has_key('func'):
        func = kwargs['func']
    else:
        func = logist
    
    J = cost_function(x,y,func, beta=beta, beta0=beta0)
    cur_iter = 1
    
    converged = False
    while not converged:
        grad0, gradj = grad_logist(x,y, beta=beta, beta0=beta0)
        beta0 -= alpha * grad0
        beta -= alpha * gradj
        
        e = cost_function(x,y,func, beta=beta, beta0=beta0)
        
        if abs(J-e) < eps:
            print 'Converged. Iterations: ', cur_iter
            converged = True
        
        if cur_iter == max_iter: 
            print 'Max iterations reached'
            converged = True
        
        J = e
        cur_iter += 1
        
    return (beta0, beta)


def grad_logist(x, y, beta, beta0):
    # if isPandasSeries(x) or isPandasDF(x):
    #     x = x.values
    # if isPandasSeries(y):
    #     y = y.values
    ip = np.dot(x, beta)
    exptl = np.exp(-beta0 - ip)
    Dfunc_0 = exptl/map(lambda a: a**2, (1.+exptl))
    grad0 = (logist(x, beta0, beta)-as_col(y)) * as_col(Dfunc_0)
    gradj = np.sum(grad0 * x, axis=0)/x.shape[0]
    grad0 = grad0.sum()/x.shape[0]
    return (grad0, gradj)

def cost_function(x, y, func, **parms):
    ## Arguments:
    ##     x : training data
    ##     y : training data classes
    ##  func : regression function
    ## parms : dict of parameters for func (kwargs)
    
    ## equivalent way of computing squares of elements:
    ## return np.sum(map(lambda a: a**2, func(x,**parms)-y))/(2*x.shape[0])
    return np.sum([a**2 for a in func(x, **parms)-y])/(2*x.shape[0])

def logist(x, beta0, beta):
    ip = np.dot(x, beta)
    return as_col(1./(1. + np.exp(-beta0 - ip)))

def as_col(vv):
    return vv.reshape((-1,1))


def isPandasSeries(obj):
    return all( [ qq in str(type(obj)) for qq in ['Series', 'pandas'] ] ) 

def isPandasDF(obj):
    return all( [ qq in str(type(obj)) for qq in ['DataFrame', 'pandas'] ] )


x = np.r_[randn(10,3), randn(10,3) + 2]
y = np.r_[[[-1] for _ in range(10)], [[1] for _ in range(10)]]

columns = ['x'+str(j+1) for j in range(x.shape[1])] + ['y']
data = DataFrame(np.c_[x,y], columns=columns)

print cost_function(x,y, logist, beta0=0, beta=[0,0,0])

beta0, beta = logistRg(data=data, eps=1e-6, alpha=.25, max_iter=5000)

print data

print beta0
print beta

print cost_function(x,y, logist,beta0=beta0, beta=beta)
