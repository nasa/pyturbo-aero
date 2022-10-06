import numpy as np

def derivative_1(t,x):
    """ derivative_1 Summary of this function goes here
        Detailed explanation goes here
    """
    dt = t[1]-t[0]
    n = len(t)
    ddx = np.zeros(len(t))      
    ## First Order
    ddx[0] = (4*x[1]-x[2]-3*x[0])/(2*dt)

    ## Central differencing for everything else
    for i in range(1,n-1):
        ddx[i] = (x[i+1]-x[i-1])/(t[i+1]-t[i-1])
    
    ddx[n-1] = (3*x[n-1]-4*x[n-2]+x[n-3])/(2*dt)

    return ddx

 
def derivative_2(t,x):
    """ derivative_2 Summary of this function goes here
        Detailed explanation goes here
    """
    dt = t[1]-t[0]

    n = len(t)
    ddx = np.zeros(len(t))      
       
    ## Central for everything else
    ddx[0]=0
    for i in range(1,n-1):
        ddx[i] = (x[i+1]-2*x[i]+x[i-1])/(dt*dt)
    
    ddx[n-1]=0
    return ddx

