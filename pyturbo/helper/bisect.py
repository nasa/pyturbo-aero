import numpy as np


def bisect(func,lb,ub,tol,verbose=True,**kwargs):
    """
        Bisection method, solves for 0 given a function, lower bound, upper bound
        inputs:
            func - function handle
            lb - lower bound
            ub - upper bound
            tol - tolerance, when to stop the minimization
            *args - arguments that will be passed in addition to the function 
        returns
            x - value that minimizes the function
            flag = 1 (good,passed) -1 (max iterations reached)

    """
    max_iter=120
    points = np.zeros(max_iter)
    errors = np.zeros(max_iter)+1
    
    flag = 1
    c = lb
    while iter<max_iter:
        c = (lb+ub)/2
        fc =func(c,**kwargs)

        if (fc == 0 or abs(fc) < tol):
            if (verbose):
                print('Iter {0}: fc={1}'.format(iter,abs(fc)))
            break
        
        fa = func(lb,**kwargs)

        points[iter] = ub
        errors[iter] = fa
        if (np.sign(fc) == np.sign(fa)):
            lb = c
        else:
            ub = c
        if (verbose):
            print('Iter {0}: fc={1}'.format(iter,abs(fc)))
        iter+=1
        
    if (iter>=max_iter):
        flag = -1
        i = errors.index(min(errors)) 
        c = points[i]
    return c,flag
