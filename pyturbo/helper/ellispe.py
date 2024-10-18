from typing import Union
import numpy as np
import numpy.typing as npt
from .convert_to_ndarray import convert_to_ndarray
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d

class ellispe:
    xc:float
    yc:float    
    a:float
    b:float
    
    def __init__(self,xc:float,yc:float,xmajor:float,yminor:float,alpha_start:float,alpha_stop:float):
        """_summary_

        Args:
            xc (float): x coordinate of center of ellsipe
            yc (float): y coordinate of center of ellsipe
            xmajor (float): major axis x 
            yminor (float): minor axis y
            alpha_start (float): starting angle
            alpha_stop (float): ending angle
        """
        self.xc = float(xc)
        self.yc = float(yc)
        self.a = float(xmajor)
        self.b = float(yminor)
        self.alpha_start = alpha_start
        self.alpha_stop = alpha_stop
        self.sortY = False
        self.sortYasc = True 

    def get_point(self,t:Union[float,npt.NDArray]):
        """_summary_

        Args:
            t (Union[float,npt.NDArray]): _description_
        """
        def ellispe_y(x:Union[float,npt.NDArray]):
            y1 = np.sqrt(self.b**2*(1 - (x-self.xc)**2 / self.a**2)) + self.yc
            y2 = -np.sqrt(self.b**2*(1 - (x-self.xc)**2 / self.a**2)) + self.yc
            return y1, y2
        
        t = convert_to_ndarray(t)
        alpha = (self.alpha_stop-self.alpha_start)*t + self.alpha_start
        x = np.linspace(self.xc-self.a,self.xc+self.a,40)
        y1,y2 = ellispe_y(x)
        y1[0] = y1[-1] # fixes the nan 
        y2[0] = y2[-1]
        # Clockwise
        theta1 = np.degrees(np.arctan2(y1-self.yc, x-self.xc)) # 180 to 0
        theta2 = np.degrees(np.arctan2(y2-self.yc, x-self.xc)) # -180 to 0 
        if theta2[0]>0:
            theta2[0] *= -1 
        
        theta = np.concatenate([theta2,np.flip(theta1)[1:]]) # 180 to 0 to -180
        x = np.concatenate([x,np.flip(x)[1:]])
        y = np.concatenate([y1,np.flip(y2)[1:]])
        
        alpha = np.linspace(self.alpha_start,self.alpha_stop,len(t))
        x = interp1d(theta, x)(alpha)
        y = interp1d(theta, y)(alpha)
        
        return x,y

    def plot(self):
        t = np.linspace(0,1,100)
        x,y = self.get_point(t)
        plt.figure(num=1,clear=True)
        plt.plot(x,y)
        plt.axis('scaled')
        plt.show()
        