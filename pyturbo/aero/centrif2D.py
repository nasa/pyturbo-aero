from enum import Enum
import numpy as np
from typing import List
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from ..helper import bezier,line2D,ray2D,arc,ray2D_intersection,exp_ratio,convert_to_ndarray,derivative,dist,pw_bezier2D,bisect
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

    
class Centrif2D:
    """Constructing the 2D profiles for a centrif compressor or turbine
    Profiles are constructed in the meridional plane and fitted between hub and shroud
    """
    camber:bezier
    alpha1:float
    alpha2:float
    stagger:float
    x1:float
    x2:float
    ss:bezier
    ps:bezier
    
    ss_te:arc
    ps_te:arc
    
    le_thickness:float
    ss_bezier:bezier
    ss_x:List[float]
    ss_y:List[float]    # this is rtheta 
    
    ps_bezier:bezier
    ps_x:List[float]
    ps_y:List[float]
    
    ss_pts:npt.NDArray
    ps_pts:npt.NDArray
    ss_te_pts:npt.NDArray
    ps_te_pts:npt.NDArray
    te_cut:bool
    
    def __init__(self) -> None:
        pass
    
    def add_camber(self,alpha1:float,alpha2:float,stagger:float,x1:float,x2:float) -> None:
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.x1 = x1
        self.x2 = x2 
        self.stagger = stagger
        
        h = np.tan(stagger)-(1-x2)*np.tan(alpha2)
        x = [0,x1,x2,1]
        rtheta = [0, x1*np.tan(stagger),h,np.tan(stagger)]
        self.camber = bezier(x,rtheta)
        self.ss_x=[]; self.ss_y = []
        self.ps_x=[]; self.ps_y = []
    
    def add_le_thickness(self,thickness:float):
        """Add thickness to the leading edge 

        Args:
            thickness (float): thickness as a percent of the axial chord 
        """
        dx,dy = self.camber.get_point_dt(0)
        m = np.sqrt(dx**2 + dy**2) # magnitude

        self.le_thickness = thickness
        if abs(dy) <1E-6:
            self.ss_x.append(0)
            self.ss_y.append(thickness)
            self.ps_x.append(0)
            self.ps_y.append(-thickness)
        else:
            self.ss_x.append(thickness*(dx/m))
            self.ss_y.append(thickness*(-dy/m))
            
            self.ps_x.append(thickness*(-dx/m))
            self.ps_y.append(thickness*(dy/m))
            
    
    def match_le_thickness(self)->None:
        """Matches the 2nd derivative at the leading edge 
        """
        ss_dx2,ss_dy2 = self.ssBezier.get_point_dt2(0)
        
        dx,dy = self.camber.get_point_dt(0)
        m = np.sqrt(dx**2 + dy**2) # magnitude

        def match_ps_deriv2(thickness:float,camber_dx:float,camber_dy:float): 
            # adjust thickness to match 2nd derivative 
            self.ps_x[0] = thickness*(-camber_dx/m)
            self.ps_y[0] = thickness*(camber_dy/m)
            ps_dx2,ps_dy2 = self.psBezier.get_point_dt2(0)
            return abs(ps_dy2/ps_dx2 - ss_dy2/ss_dx2)
        
        temp = minimize_scalar(match_ps_deriv2,bounds=(0,self.le_thickness*15),method="bounded") 
        #! Check temp variable to see if it converged and if it doesn't, use the default le_thickness
        print('check')

    def add_ss_thickness(self,thickness_array:List[float],expansion_ratio:float=1.2):
        """builds the suction side 

        Args:
            thickness_array (List[float]): _description_
            expansion_ratio (float, optional): _description_. Defaults to 1.2.
        """
        t =  exp_ratio(expansion_ratio,len(thickness_array)+2,1) # 1 point for the leading edge and 1 for TE starting point before radius is added
        x, y = self.camber.get_point(t)
        dx, dy = self.camber.get_point_dt(t)
        m = np.sqrt(dx**2 + dy**2) # magnitude
        indx = 0 
        for i in range(1,len(t)-1):
            self.ss_x.append(x[i]+dx*thickness_array[indx]/m[i])
            self.ss_y.append(y[i]-dy*thickness_array[indx]/m[i])
            
        
    def add_ps_thickness(self,thickness_array:List[float],expansion_ratio:float=1.2):
        """Builds the pressure side

        Args:
            thickness_array (List[float]): _description_
            expansion_ratio (float, optional): _description_. Defaults to 1.2.
        """
        t =  exp_ratio(expansion_ratio,len(thickness_array)+2,1)
        x, y = self.camber.get_point(t)
        dx, dy = self.camber.get_point_dt(t)
        m = np.sqrt(dx**2 + dy**2) # magnitude
        indx = 0 
        for i in range(1,len(t)-1):
            self.ps_x.append(x[i]-dx*thickness_array[indx]/m[i])
            self.ps_y.append(y[i]+dy*thickness_array[indx]/m[i])
        

    def add_te_radius(self,radius:float,wedge_ss:float,wedge_ps:float):
        """Add a trailing edge that's rounded

        Args:
            radius (float): nondimensional trailing edge radius normalized
            wedge_ss (float): suction side wedge angle
            wedge_ps (float): pressure side wedge angle 
        """
        x,y = self.camber.get_point(1)
        dx,dy = self.camber.get_point_dt(1) # Gets the slope at the end
        m = np.sqrt(dx**2+dy**2)
        
        theta = np.atan2(dy,dx)
        
        self.ps_te = arc(x,y,radius,theta+90-wedge_ps,theta)
        self.ss_te = arc(x,y,radius,theta,theta+90-wedge_ss)
         
    def add_te_cut(self,radius:float):
        x,y = self.camber.get_point(1)
        self.te_cut = True
        self.ss_te_pts = np.linspace(y-radius,y,10)
        self.ps_te_pts = np.linspace(y,y+radius,10) 

    def build(self,npts:int,npt_te:int=20):
        # Build the pressure and suction sides 
        
        if not self.te_cut:
            t = np.linspace(0,1,npt_te)
            
            ps_te_x,ps_te_y = self.ps_arc.get_point(t)
            
            ss_te_x,ss_te_y = self.ss_arc.get_point(t)
            ss_te_x = np.flip(ss_te_x)
            ss_te_y = np.flip(ss_te_y)
            
        
        
        pass