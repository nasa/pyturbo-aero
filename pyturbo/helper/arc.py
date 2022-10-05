from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .convert_to_ndarray import convert_to_ndarray

class arc():
    def __init__(self,xc,yc,radius,alpha_start,alpha_stop):
        self.x = xc
        self.y = yc
        self.radius = radius
        self.alpha_start = alpha_start
        self.alpha_stop = alpha_stop
        self.sortY = False
        self.sortYasc = True 

    def get_point(self,t):
        t = convert_to_ndarray(t)
        alpha = (self.alpha_stop-self.alpha_start)*t + self.alpha_start
        x,y = np.zeros(len(t)),np.zeros(len(t))

        for i in range(len(alpha)):
            x[i] = self.x + self.radius*cos(radians(alpha[i]))
            y[i] = self.y + self.radius*sin(radians(alpha[i]))

        # Check sorting
        if (self.sortY):
            alpha_test = (self.alpha_start-self.alpha_stop)*t + self.alpha_stop
            x_test = self.x + self.radius*cos(radians(alpha_test))
            y_test = self.y + self.radius*sin(radians(alpha_test))
            if (self.sortYasc==1): # sort ascending
                # check the start and end to see if y<y_test
                if (y_test[-1]<y[-1]):
                    x = x_test; y = y_test; 
            else: # sort descending
                if (y_test[-1]>y[-1]):
                    x = x_test; y = y_test; 
        return x,y
        
    def get_deriv(self,t):
        [x,y] = self.get_point(t)
        dx = -(y-self.y)     
        dy = x-self.x    
    
    
    def plot(self,color='b'):
        t = np.linspace(0,1,20)
        [x, y] = self.get_point(t)
        fig1, ax1 = plt.subplots()
    
        ax1.plot(x, y, color=color, linestyle='-')
        ax1.plot(self.x, self.y, color='k', marker='o')
        plt.show()

def arclen(x,y):
    """
        Computes the length between points 
        Inputs
            x,y - this can be a list or a numpy array but not a scalar
    """
    x = convert_to_ndarray(x)
    y = convert_to_ndarray(y)
    if (len(x)<2):
        return 0

    dx,dy = np.diff(x,axis=0), np.diff(y,axis=0)

    L = np.sqrt(dx*dx+dy*dy)
    L = np.insert(L,0,0)
    return L

def arclen3(x,y,z):
    """
        Equally space points along a bezier curve using arc length
    """
    x = convert_to_ndarray(x)
    y = convert_to_ndarray(y)
    z = convert_to_ndarray(z)
    if (len(x)<2):
        return 0

    dx,dy,dz = np.diff(x,axis=0), np.diff(y,axis=0), np.diff(z,axis=0)

    L = np.sqrt(dx*dx+dy*dy+dz*dz)
    L = np.insert(L,0,0)
    return L
