from typing import List, Optional, Tuple, Union
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
from scipy.interpolate import Rbf
from shapely.geometry import Polygon, Point


# https://www.journaldev.com/14893/python-property-decorator
# https://www.codementor.io/sheena/advanced-use-python-decorators-class-function-du107nxsv
# https://stackabuse.com/pythons-classmethod-and-staticmethod-explained/

class bezier:
    n:int           # Number of control points
    c:npt.NDArray    # bezier coefficients
    x:npt.NDArray    # x-control points
    y:npt.NDArray    # y-control points


    def __init__(self, x:Union[List[float],npt.NDArray],y:Union[List[float],npt.NDArray]):
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
        return bezier(np.flip(self.x),np.flip(self.y)) # type: ignore

    @property
    def get_x_y(self) -> Tuple[npt.NDArray,npt.NDArray]:
        return self.x,self.y
    
    def get_curve_length(self) -> float:
        """Gets the curve length

        Returns:
            float: curve length
        """
        [x,y] = self.get_point(np.linspace(0,1,100))
        d = np.zeros((len(x),))
        for i in range(0,len(x)-1):
            d[i] = math.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
        
        return sum(d) # Linear approximation of curve length

    

    def __call__(self,t:Union[float,npt.NDArray],equally_space_pts:bool=False):
        return self.get_point(t,equally_space_pts)
    
    
    def get_point(self,t:Union[float,npt.NDArray,List[float]],equally_space_pts:bool=False) -> Tuple[npt.NDArray,npt.NDArray]:
        """Get a point or points along a bezier curve

        Args:
            t (Union[float,npt.NDArray]): scalar, list, or numpy array 
            equally_space_pts (bool, optional): True = space points equally. Defaults to False.

        Returns:
            Tuple: containing x and y points
        """
        t = convert_to_ndarray(t)
        x = t*0; y = t*0  
        for i in range(self.n):
            x += bernstein_poly(self.n-1,i,t)*self.x[i]
            y += bernstein_poly(self.n-1,i,t)*self.y[i]
        
        if (equally_space_pts and len(x)>2): # type: ignore
            pts = equal_space(x,y) # type: ignore
            return pts[1],pts[2]
        return x,y
    
    def get_point_dt(self,t:Union[float,npt.NDArray]):
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
            dx += bernstein_poly(self.n-2,i,t)*(self.x[i+1]-self.x[i])
            dy += bernstein_poly(self.n-2,i,t)*(self.y[i+1]-self.y[i])
        dx*=self.n
        dy*=self.n
        return dx,dy

    def get_point_dt2(self,t:Union[float,npt.NDArray]):
        t = convert_to_ndarray(t)
        dx2 = t*0; dy2 = t*0
        for i in range(self.n-2):
            dx2 = bernstein_poly(self.n-2,i,t)*(self.n-1)*self.n*(self.x[i+2]-2*self.x[i+1]+self.x[i])
            dy2 = bernstein_poly(self.n-2,i,t)*(self.n-1)*self.n*(self.y[i+2]-2*self.y[i+1]+self.y[i])
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
     
    def __call__(self,t:Union[float,npt.NDArray],equally_space_pts:bool=False):
        return self.get_point(t,equally_space_pts)
    
    def get_point(self,t:Union[float,npt.NDArray],equally_space_pts = True):
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

        if (equally_space_pts and len(Bx)>2):
            pts = equal_space(Bx,By,Bz)
            return pts[1],pts[2],pts[3]
        elif (len(Bx)==1):
            return Bx[0],By[0],Bz[0] # if it's just one point return floats
        return Bx,By,Bz
    
    def get_point_dt(self,t:Union[float,npt.NDArray]):
        """Gets the derivative dx,dy as a function of t 

        Args:
            t (Union[float,npt.NDArray]): Array or float from 0 to 1 
            
        Returns:
            tuple: containing

                **dx** (npt.NDArray): Derivative of x as a function of t 
                **dy** (npt.NDArray): Derivative of y as a function of t 
                **dz** (npt.NDArray): Derivative of z as a function of t 
        """

        def B(n:int,i:int,t:Union[float,npt.NDArray]) -> Union[float,npt.NDArray]:
            c = math.factorial(n)/(math.factorial(i)*math.factorial(n-i))
            return c*t**i *(1-t)**(n-i)
        
        dx = 0*t; dy = 0*t; dz = 0*t
        for i in range(self.n-1):
            dx += B(self.n-1,i,t)*self.n*(self.x[i+1]-self.x[i])
            dy += B(self.n-1,i,t)*self.n*(self.y[i+1]-self.y[i])
            dz += B(self.n-1,i,t)*self.n*(self.z[i+1]-self.z[i])

        return dx,dy,dz
    
class pw_bezier2D:
    
    bezierArray: List[bezier]
    tArray: npt.NDArray
    dist: npt.NDArray
    dmax: float
    
    
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

    def get_point(self,t:Union[List[float],float,np.ndarray],equally_space_pts:bool=False):
        """Gets the point(s) at a certain percentage along the piecewise bezier curve

        Args:
            t (Union[List[float],float,np.ndarray]): percentage(s) along a bezier curve. You can specify a float, List[float], or a numpy array
            equally_space_pts (bool, optional): Equally space points using arc length. Defaults to False.

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

        if (equally_space_pts and len(x)>2):
            pts = equal_space(x,y)
            x = pts[1]
            y = pts[2]
        return x,y

    
    def get_dt(self,t:Union[List[float],npt.NDArray]):
        t = convert_to_ndarray(t)
        js = 1; tadj = 0
        dx = np.zeros((len(t),1))
        dy = np.zeros((len(t),1))
        for i in range(0,len(t)):
            for j in range(js,len(self.tArray)):
                if (t[i]<=self.tArray[j]):
                    # Scale t
                    ts = (t[i]-tadj) * self.dmax/self.dist[j]
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


def equal_space(x:npt.NDArray,y:npt.NDArray,z:npt.NDArray=np.array([])) -> npt.NDArray:
    """Equally space points based on arc length

    Args:
        x (npt.NDArray): x values
        y (npt.NDArray): y values
        z (npt.NDArray): z values. Defaults to []
        
    Returns:
        npt.NDArray: containing t,x,y or t,x,y,z points 
    
    """
    t = np.linspace(0,1,len(x))
    x_t = sp_intp.PchipInterpolator(t,x)
    y_t = sp_intp.PchipInterpolator(t,y)
    if len(z) == len(x):
        z_t = sp_intp.PchipInterpolator(t,y)
        arcL = arclen3(x,y,z)
    else:
        arcL = arclen(x,y)
    mean_old = np.mean(arcL)
    mean_err = 1
    
    while (mean_err>1.0E-3):
        target_len = np.sum(arcL)/len(t) # we want equal length
        csum = np.cumsum(arcL)
        f = sp_intp.PchipInterpolator(t,csum) # Arc length as a function of t 
        t_start = np.min(t)   
        t_end = np.max(t)
        f2 = lambda x,y: abs(f(x)-f(y)-target_len)
        for i in range(0,t.size-2):
            temp = minimize_scalar(f2,bounds=(t_start,t_end),method="bounded",tol=1e-6,args=(t_start))
            t[i+1] = temp.x # type: ignore
            t_start = t[i+1]

        x = x_t(t)
        y = y_t(t)
        if len(z) == len(x):
            z = z_t(t) # type: ignore
            arcL = arclen3(x,y,z)
        else:
            arcL = arclen(x,y)
        mean_new = np.mean(arcL)
        mean_err = abs(mean_old-mean_new)/abs(mean_new)
        mean_old = mean_new
    
    if len(z) == len(x):
        return np.vstack([t,x,y,z])
    else:
        return np.vstack([t,x,y])
    

def bernstein_poly(n:int,i:int,t:Union[float,npt.NDArray]):
    """Compute Bernstein polynomial B_i^n at t."""
    t = np.asarray(t)
    term1 = np.where(t == 0, float(i == 0), t ** i)
    term2 = np.where(t == 1, float(i == n), (1 - t) ** (n - i))
    return comb(n, i) * term1 * term2


class BezierSurface:
    bounds:npt.NDArray
    perimeter_pts:npt.NDArray
    inside_pts:npt.NDArray
    
    def __init__(self,perimeter_pts:npt.NDArray, inside_pts:npt.NDArray):
        """
        Generate a Bézier surface from a grid of control points.
        
        Parameters:
            control_points: 3D NumPy array of shape (m, n, 3). Optional
            rbf: Number of samples per dimension
            
        Returns:
            X, Y, Z grids of surface points
        """
        self.perimeter_pts = perimeter_pts
        self.inside_pts = inside_pts


    def __call__(self,resolution:int=20):
        control_points = self.__control_pts__()
        m, n = control_points.shape
        
        u_vals = np.linspace(0, 1, resolution)
        v_vals = np.linspace(0, 1, resolution)
        surface = np.zeros((resolution, resolution, 3))
        
        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                point = np.zeros(3)
                for r in range(m):
                    for s in range(n):
                        bern_u = bernstein_poly(m - 1,r, u)
                        bern_v = bernstein_poly(n - 1,s, v)
                        point += bern_u * bern_v * control_points[r][s]
                surface[i][j] = point
        
        X = surface[:, :, 0]
        Y = surface[:, :, 1]
        Z = surface[:, :, 2]
        return X, Y, Z
    
    def __control_pts__(self):
        control_points = np.vstack([self.perimeter_pts,self.inside_pts])
        _, idx = np.unique(control_points, axis=0, return_index=True)
        control_points = control_points[np.sort(idx)]
        return control_points
    

    def plot_bezier_surface(self,resolution: int = 50):
        control_points = self.__control_pts__()

        X, Y, Z = self(resolution)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X, Y, Z, cmap='viridis', color='red', edgecolor='none', alpha=0.8) # type: ignore
        ax.scatter(*control_points.reshape(-1, 3).T, color='black', label='Control Points')

        ax.set_title("Bézier Surface")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z") # type: ignore
        ax.legend()
        plt.tight_layout()
        plt.show()