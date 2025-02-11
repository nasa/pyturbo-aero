from typing import List, Tuple, Union
import numpy as np
import numpy.typing as npt
from scipy.special import comb
import scipy.interpolate as sp_intp
from scipy.optimize import minimize_scalar 
import matplotlib.pyplot as plt
import math 
from scipy.special import comb
from .arc import arclen3, arclen
from .convert_to_ndarray import convert_to_ndarray
import numpy.typing as npt 

# https://www.journaldev.com/14893/python-property-decorator
# https://www.codementor.io/sheena/advanced-use-python-decorators-class-function-du107nxsv
# https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/

class bezier:
    n:int           # Number of control points
    c:np.ndarray    # bezier coefficients
    x:np.ndarray    # x-control points
    y:np.ndarray    # y-control points


    def __init__(self, x:List[float],y:List[float]):
        """Initializes a 2D bezier curve

        Args:
            x (List[float]): x coordinates
            y (List[float]): y coordinates
        """
        self.n = len(x)
        self.x = convert_to_ndarray(x)
        self.y = convert_to_ndarray(y)

    def flip(self):
        """Reverses the direction of the bezier curve

        Returns:
            bezier: flipped bezier curve
        """
        return bezier(np.flip(self.x),np.flip(self.y))

    @property
    def get_x_y(self) -> Tuple[List[float],List[float]]:
        return self.x,self.y
    
    def get_curve_length(self) -> float:
        """Gets the curve length

        Returns:
            float: curve length
        """
        [x,y] = self.get_point(np.linspace(0,1,100))
        d = np.zeros((len(x),1))
        for i in range(0,len(x)-1):
            d[i] = math.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
        
        return sum(d)[0] # Linear approximation of curve length

    def __equal_space__(self,t:npt.NDArray,x:npt.NDArray,y:npt.NDArray):
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
            f2 = lambda x,y: abs(f(x)-f(y)-target_len)
            for i in range(0,t.size-2):
                temp = minimize_scalar(f2,bounds=(t_start,t_end),method="bounded",tol=1e-6,args=(t_start))
                t[i+1] = temp.x
                t_start = t[i+1]

            x,y = self.get_point(t,equal_space=False)
            arcL = arclen(x,y)
            mean_new = np.mean(arcL)
            mean_err = abs(mean_old-mean_new)/abs(mean_new)
            mean_old = mean_new
        return x,y

    def __call__(self,t:Union[float,npt.NDArray],equal_space:bool=False):
        return self.get_point(t,equal_space)
    
    @staticmethod
    def __B__(n:int,i:int,t:float):
        return comb(n, i) * ( t**i ) * (1 - t)**(n-i)
    
    def get_point(self,t:Union[float,npt.NDArray],equal_space:bool=False):
        """Get a point or points along a bezier curve

        Args:
            t (Union[float,npt.NDArray]): scalar, list, or numpy array 
            equal_space (bool, optional): True = space points equally. Defaults to False.

        Returns:
            Tuple: containing x and y points
        """
        t = convert_to_ndarray(t)
        x = 0; y = 0  
        for i in range(self.n):
            x += self.__B__(self.n-1,i,t)*self.x[i]
            y += self.__B__(self.n-1,i,t)*self.y[i]
        
        if (equal_space and len(x)>2):
            x,y = self.__equal_space__(t,x,y)
            return x,y
        return x,y
    
    def get_point_dt(self,t:Union[npt.NDArray]):
        """Gets the derivative dx,dy as a function of t 

        Args:
            t (Union[np.ndarray]): Array or float from 0 to 1 
            
        Returns:
            tuple: containing

                **dx** (npt.NDArray): Derivative of x as a function of t 
                **dy** (npt.NDArray): Derivative of y as a function of t 
        """
        t = convert_to_ndarray(t)
        
        dx = t*0; dy = t*0
        for i in range(self.n-1):
            dx += self.__B__(self.n-2,i,t)*(self.x[i+1]-self.x[i])
            dy += self.__B__(self.n-2,i,t)*(self.y[i+1]-self.y[i])
        dx*=self.n
        dy*=self.n
        return dx,dy

    def get_point_dt2(self,t:Union[npt.NDArray]):
        t = convert_to_ndarray(t)
        
        dx2 = t*0; dy2 = t*0
        for i in range(self.n-2):
            dx2 = self.__B__(self.n-2,i,t)*(self.n-1)*self.n*(self.x[i+2]-2*self.x[i+1]+self.x[i])
            dy2 = self.__B__(self.n-2,i,t)*(self.n-1)*self.n*(self.y[i+2]-2*self.y[i+1]+self.y[i])
        return dx2,dy2
    
    def rotate(self,angle:float):
        """Rotate 

        Args:
            angle (float): _description_
        """
        angle = np.radians(angle)
        rot_matrix = np.array([[math.cos(angle) -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
        ans = (rot_matrix*self.x.transpose()).transpose()
        self.x = ans[:,0]
        self.y = ans[:,1]
        
    def plot2D(self,equal_space=False):
        """Creates a 2D Plot of a bezier curve 

        Args:            
            equal_space (bool, optional): Equally spaces the points using arc length. Defaults to False.
            figure_num (int, optional): if you want plots to be on the same figure. Defaults to None.
        """

        
        t = np.linspace(0,1,100)
        [x,y] = self.get_point(t,equal_space)        
        plt.plot(x, y,'-b')
        plt.plot(self.x, self.y,'or')

        plt.xlabel("x-label")
        plt.ylabel("y-label")
        plt.axis('scaled')


class bezier3:
    def __init__(self,x,y,z):
        self.n = len(x)
        self.x = convert_to_ndarray(x)
        self.y = convert_to_ndarray(y)
        self.z = convert_to_ndarray(z)
        self.c = np.zeros(self.n)
        for i in range(self.n):
            self.c[i] = comb(self.n-1, i, exact=False) # use floating point
    
    def __equal_space__(self,t:np.ndarray,x:np.ndarray,y:np.ndarray,z:np.ndarray):
        """Equally space points along a bezier curve using arc length
            
        Args:
            t (np.ndarray): position along bezier curve. Example: t = np.linspace(0,1,100)
            x (np.ndarray): x-coordinate as numpy array
            y (np.ndarray): y-coordinates as numpy array
            z (np.ndarray): z-coordinates as numpy array

        Returns:
            (Tuple): containing
                - *x* (np.ndarray): new values of x that are equally spaced 
                - *y* (np.ndarray): new values of y that are equally spaced 

        """
        arcL = arclen3(x,y,z)
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

            x,y,z = self.get_point(t,equal_space=False)
            arcL = arclen3(x,y,z)
            mean_new = np.mean(arcL)
            mean_err = abs(mean_old-mean_new)/abs(mean_new)
            mean_old = mean_new
        return x,y,z
    
    def __call__(self,t:Union[float,npt.NDArray],equal_space:bool=False):
        return self.get_point(t,equal_space)
    
    def get_point(self,t,equal_space = True):
        """Gets the point(s) at a certain percentage along the piecewise bezier curve

        Args:
            t (Union[List[float],float,np.ndarray]): percentage(s) along a bezier curve. You can specify a float, List[float], or a numpy array
            equal_space (bool, optional): Equally space points using arc length. Defaults to False.

        Returns:
            (Tuple): containing
                - *x* (np.ndarray): x-coordinates
                - *y* (np.ndarray): y-coordinates
                
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
            Bx,By,Bz = self.__equal_space__(t,Bx,By,Bz)
            return Bx,By,Bz
        elif (len(Bx)==1):
            return Bx[0], By[0],Bz[0] # if it's just one point return floats
        return Bx,By,Bz
    
    def get_point_dt(self,t:Union[float,List[float],np.ndarray]):
        """Gets the derivative dx,dy as a function of t 

        Args:
            t (Union[np.ndarray]): Array or float from 0 to 1 
            
        Returns:
            tuple: containing

                **dx** (npt.NDArray): Derivative of x as a function of t 
                **dy** (npt.NDArray): Derivative of y as a function of t 
        """
        t = convert_to_ndarray(t)
        
        def B(n:int,i:int,t:float):
            c = math.factorial(n)/(math.factorial(i)*math.factorial(n-i))
            return c*t**i *(1-t)**(n-i)
        
        dx = 0; dy = 0
        for i in range(self.n-1):
            dx += B(self.n-1,i,t)*self.n*(self.x[i+1]-self.x[i])
            dy += B(self.n-1,i,t)*self.n*(self.y[i+1]-self.y[i])
            dz += B(self.n-1,i,t)*self.n*(self.z[i+1]-self.z[i])

        return dx,dy


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
        """Initializes the piecewise bezier curve from an array of bezier curves

        Args:
            array (List[bezier]): Bezier curves as an array
        """
        
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

    def get_point(self,t:Union[List[float],float,np.ndarray],equal_space:bool=False):
        """Gets the point(s) at a certain percentage along the piecewise bezier curve

        Args:
            t (Union[List[float],float,np.ndarray]): percentage(s) along a bezier curve. You can specify a float, List[float], or a numpy array
            equal_space (bool, optional): Equally space points using arc length. Defaults to False.

        Returns:
            (Tuple): containing
                - *x* (np.ndarray): x-coordinates
                - *y* (np.ndarray): y-coordinates

        """
        n = len(self.bezierArray)
        t = convert_to_ndarray(t) 
        x = np.zeros(len(t)+(n-1)*(len(t)-1))
        y = np.zeros(len(t)+(n-1)*(len(t)-1))
        t_start=0; lenT = len(t)
        for i in range(0,n): # loop for each bezier curve
            [xx,yy] = self.bezierArray[i].get_point(t)
            if i == 0:
                x[0:lenT] = xx
                y[0:lenT] = yy
                t_start += lenT
            else:
                x[t_start:t_start+lenT-1] = xx[1:]
                y[t_start:t_start+lenT-1] = yy[1:]
                t_start += lenT-1
            

        if len(t)>0:
            t = np.linspace(0,1,len(x))
            x = sp_intp.interp1d(t,x)(np.linspace(0,1,lenT))
            y = sp_intp.interp1d(t,y)(np.linspace(0,1,lenT))
            return x,y

        if (equal_space and len(x)>2):
            x,y = self.__equal_space__(t,x,y)
        return x,y

    def __equal_space__(self,t:np.ndarray,x:np.ndarray,y:np.ndarray):
        """Equally space points along a bezier curve using arc length
            
        Args:
            t (np.ndarray): position along bezier curve. Example: t = np.linspace(0,1,100)
            x (np.ndarray): x-coordinate as numpy array
            y (np.ndarray): y-coordinates as numpy array

        Returns:
            (Tuple): containing
                - *x* (np.ndarray): new values of x that are equally spaced 
                - *y* (np.ndarray): new values of y that are equally spaced 

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
            
    
    def plot(self):
        for i in range(0,len(self.bezierArray)):
            self.bezierArray[i].plot2D()

