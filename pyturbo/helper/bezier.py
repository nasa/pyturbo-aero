from typing import List
import numpy as np
from scipy.special import comb
import scipy.interpolate as sp_intp
from scipy.optimize import minimize_scalar 
import matplotlib.pyplot as plt
import matplotlib as mpl
import math 
from scipy.special import comb
from .arc import arclen3, arclen
from .convert_to_ndarray import convert_to_ndarray

# https://www.journaldev.com/14893/python-property-decorator
# https://www.codementor.io/sheena/advanced-use-python-decorators-class-function-du107nxsv
# https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/

class bezier():
    def __init__(self, x,y):
        self.n = len(x)
        self.c = np.zeros(self.n) 
        self.x = x
        self.y = y
        self.dx = x[-1]-x[0]
        self.dy = y[-1]-y[0]
        self.t = []
        for i in range(0,self.n):
            self.c[i] = comb(self.n-1, i, exact=False) # use floating point

    def flip(self):
        '''
            Reverses the direction of the bezier curve
            returns:
                flipped bezier curve
        '''
        return bezier(np.flip(self.x),np.flip(self.y))

    @property
    def get_x_y(self):
        return self.x,self.y
    
    def get_curve_length(self):
        """
            Get a point or points along a bezier curve
            Inputs:
                t - scalar, list, or numpy array
                equal_space - try to space points equally 
            Outputs: 
                Bx, By - scalar or numpy array
        """
        [x,y] = self.get_point(np.linspace(0,1,100))
        d = np.zeros((len(x),1))
        for i in range(0,len(x)-1):
            d[i] = math.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
        
        return sum(d) # Linear approximation of curve length

    def equal_space(self,t:np.ndarray,x,y):
        """
            Equally space points along a bezier curve using arc length
            Inputs 
        """
        arcL = arclen(x,y)
        mean_old = np.mean(arcL)
        mean_err = 1
        while (mean_err>1.0E-3):
            target_len = np.sum(arcL)/len(t) # we want equal length
            csum = np.cumsum(arcL)
            f = sp_intp.PchipInterpolator(t,csum)
            t_start = np.min(t)   
            t_end = np.max(t)
            for i in range(0,t.size-2):
                f2 = lambda x: abs(f(x)-f(t_start)-target_len)
                temp = minimize_scalar(f2,bounds=(t_start,t_end),method="bounded",tol=1e-6)
                t[i+1] = temp.x
                t_start = t[i+1]

            x,y = self.get_point(t,equal_space=False)
            arcL = arclen(x,y)
            mean_new = np.mean(arcL)
            mean_err = abs(mean_old-mean_new)/abs(mean_new)
            mean_old = mean_new
        return x,y

    def get_point(self,t,equal_space=False):
        """
            Get a point or points along a bezier curve
            Inputs:
                t - scalar, list, or numpy array
                equal_space - try to space points equally 
            Outputs: 
                Bx, By - scalar or numpy array
        """
        t = convert_to_ndarray(t)
        Bx,By = np.zeros(t.size),np.zeros(t.size)
        for i in range(len(t)):
            tempx,tempy = 0.0, 0.0
            for j in range(0,self.n):
                u = self.c[j]*pow(1-t[i], self.n-j-1)*pow(t[i],j)
                tempx += u*self.x[j]
                tempy += u*self.y[j]
                
            Bx[i],By[i] = tempx,tempy
        self.t = t

        if (equal_space and len(Bx)>2):
            Bx,By = self.equal_space(t,Bx,By)
            return Bx,By
        elif (len(Bx)==1):
            return Bx[0], By[0] # if it's just one point return floats
        return Bx,By


    def plot2D(self,equal_space=False):
        fig = plt.figure(); plt.clf()
        t = np.linspace(0,1,100)
        [x,y] = self.get_point(t,equal_space)
        _, ax1 = plt.subplots()
        
        ax1.plot(x, y,'.b')
        ax1.plot(self.x, self.y,'or')

        ax1.set_xlabel("x-label")
        ax1.set_ylabel("y-label")
        plt.show()

    def get_point_dt(self,t):
        """
         Gets the derivative
        """
        if type(t) is not np.ndarray:
            t = np.array([t],dtype=float) # the brackets [] are really helpful. scalars have to be converted to array before passing
        tmin = np.min(t)
        tmax = np.max(t)
            
        Bx = np.zeros(len(t))
        By = np.zeros(len(t))
        for i in range(len(t)):                      
            tempx = 0; tempy = 0;                
            if (t[i] == 1): # Use downwind
                tempx = self.x[-1] - self.x[-2]
                tempy = self.y[-1]- self.y[-2]             
            elif (t[i] == 0):
                tempx = self.x[1] - self.x[0]
                tempy = self.y[1]- self.y[0]          
            else:
                for j in range(self.n-1): # n-1      
                    b = (comb(self.n-2,j,True)*t[i]**j) * (1-t[i])**(self.n-2-j)    # Bn-1                    
                    tempx = tempx + b*(self.x[j+1]-self.x[j]) # Note: j+1 = j
                    tempy = tempy + b*(self.y[j+1]-self.y[j])                   
                
                tempx = tempx*(self.n-1)
                tempy = tempy*(self.n-1)
            
            Bx[i] = tempx
            By[i] = tempy
        return Bx,By

class bezier3:
    def __init__(self,x,y,z):
        self.n = len(x)
        self.x = x
        self.y = y
        self.z = z
        self.c = np.zeros(self.n)
        for i in range(self.n):
            self.c[i] = comb(self.n-1, i, exact=False) # use floating point
    
    def get_point(self,t,equal_space = True):
        """
            Gets the x,y,z coordinate at a particular time instance 
            Inputs:
                t - 0 to 1
        """

        t = convert_to_ndarray(t)
        Bx,By,Bz = np.zeros(t.size),np.zeros(t.size),np.zeros(t.size)
        for i in range(len(t)):
            tempx,tempy,tempz = 0.0, 0.0, 0.0
            for j in range(0,self.n):
                u = self.c[j]*np.power(1-t[i], self.n-j-1)*pow(t[i],j)
                tempx += u*self.x[j]
                tempy += u*self.y[j]
                tempz += u*self.z[j]

            Bx[i],By[i],Bz[i] = tempx,tempy,tempz
        self.t = t

        if (equal_space and len(Bx)>2):
            Bx,By,Bz = equal_space(t,Bx,By,Bz)
            return Bx,By,Bz
        elif (len(Bx)==1):
            return Bx[0], By[0],Bz[0] # if it's just one point return floats
        return Bx,By,Bz
    
    def get_point_dt(self,t):
        """
         Gets the derivative
        """
        t = convert_to_ndarray(t)

        tmin = np.min(t)
        tmax = np.max(t)
            
        Bx = np.zeros((len(t),1))
        By = np.zeros((len(t),1))
        Bz = np.zeros((len(t),1))
        for i in range(len(t)):                      
            tempx = 0; tempy = 0; tempz=0              
            if (t[i] == 1): # Use downwind
                tempx = self.x[-1] - self.x[-2]
                tempy = self.y[-1]- self.y[-2]                  
            elif (t[i] == 0):
                tempx = self.x(2) - self.x(1)
                tempy = self.y(2)- self.y(1)                 
            else:
                for j in range(self.n-1): # n-1      
                    b = (comb(self.n-1,j,True)*t[i]**j) * (1-t[i])**(self.n-1-j)    # Bn-1                    
                    tempx = tempx + b*(self.x[j+2]-self.x[j+1]) # Note: j+1 = j
                    tempy = tempy + b*(self.y[j+2]-self.y[j+1])                   
                    tempz = tempz + b*(self.z[j+2]-self.z[j+1]) 
                tempx = tempx*(self.n-1)
                tempy = tempy*(self.n-1)
                tempz = tempz*(self.n-1)

            Bx[i] = tempx
            By[i] = tempy
            Bz[i] = tempz
        return Bx,By,Bz


def time_this(original_function):
    def new_function(*args,**kwargs):
        import datetime                 
        before = datetime.datetime.now()                     
        x = original_function(*args,**kwargs)                
        after = datetime.datetime.now()                      
        print("Elapsed Time = {0}".format(after-before))
        return x                                             
    return new_function                                   

@time_this
def func_a(stuff):
    import time
    time.sleep(3)

class pw_bezier2D:
    def __init__(self,array:List[bezier]):
        '''
            Initializes the piecewise bezier curve from an array of bezier curves
        '''
        self.bezierArray = array
    
        x = np.zeros(len(self.bezierArray)+1)
        y = np.zeros(len(self.bezierArray)+1)
        dist = np.zeros(len(self.bezierArray))
        tArray = np.zeros(len(self.bezierArray))
        x[0] = self.bezierArray[0].x[0]
        y[0] = self.bezierArray[0].y[0]
        
        for i in range(len(self.bezierArray)):
            x[i+1] = self.bezierArray[i].x[-1]
            y[i+1] = self.bezierArray[i].y[-1]
            dist[i] = math.sqrt((x[i]-x[i+1])**2 + (y[i]-y[i+1])**2)
        
        dmax = np.sum(dist)
        for i in range(len(dist)):
            tArray[i]=np.sum(dist[0:i])/dmax
        
        self.tArray =tArray
        self.dist=dist
        self.dmax = dmax

    def get_point(self,t):
        t = convert_to_ndarray(t)

        t.sort();  
        n = len(self.bezierArray); 
        x = np.zeros(int(len(t)*n-n))
        y = np.zeros(int(len(t)*n-n))
        t_start=0; lenT = len(t)

        for i in range(0,n): # loop for each bezier curve
            [xx,yy] = self.bezierArray[i].get_point(t)
            if (i<n):
                x[t_start:t_start+lenT-1] = xx[0:-1]
                y[t_start:t_start+lenT-1] = yy[0:-1]
                t_start = t_start+lenT-1
            else:
                x[t_start:] = xx
                y[t_start:] = yy
        return x,y

    def get_dt(self,t):
        t = convert_to_ndarray(t)
        t = t.sort(); js = 1; tadj = 0
        dx = np.zeros((len(t),1))
        dy = np.zeros((len(t),1))
        for i in range(0,len(t)):
            for j in range(js,len(self.tArray)):
                if (t[i]<=self.tArray[j]):
                    # Scale t
                    ts = (t(i)-tadj) * self.dmax/self.dist[j]
                    dx[i],dy[i] = self.bezierArray[j].get_point_dt(ts)
                    break
                else:
                    js = j+1
                    tadj = self.tArray[j]
        return dx,dy
                    
        
    def shift(self,x,y):
        for i in range(0,len(self.bezierArray)):
            for j in range(0,len(self.bezierArray[i].x)):
                self.bezierArray[i].x[j] = self.bezierArray[i].x[j]+x
                self.bezierArray[i].y[j] = self.bezierArray[i].y[j]+y
            
    
    def Plot(self):
        for i in range(0,len(self.bezierArray)):
            self.bezierArray[i].plot()

