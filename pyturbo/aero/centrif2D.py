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
        """Defines a camber line

        Args:
            alpha1 (float): inlet flow angle
            alpha2 (float): outlet flow angle
            stagger (float): stagger angle
            x1 (float): Location of LE angle control point 
            x2 (float): Location of TE angle control point
        """
        self.alpha1 = np.radians(alpha1)
        self.alpha2 = np.radians(alpha2)
        self.x1 = x1
        self.x2 = x2 
        self.stagger = np.radians(stagger)
        
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
            thickness_array (List[float]): thickness defined perpendicular to the camber line
            expansion_ratio (float, optional): Expansion ratio where thickness arrays are defined. Defaults to 1.2.
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
            thickness_array (List[float]): Thickness array to use 
            expansion_ratio (float, optional): Expansion ratio where thickness arrays are defined. Defaults to 1.2.
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
        """Cuts the trailing edge instead of having a rounded TE

        Args:
            radius (float): Trailing edge radius where to define the cut
        """
        _,y = self.camber.get_point(1)
        self.te_cut = True
        self.ss_te_pts = np.linspace(y-radius,y,10)
        self.ps_te_pts = np.linspace(y,y+radius,10) 

    def build(self,npts:int,npt_te:int=20):
        """Build the 2D Geometry 

        Args:
            npts (int): number of points to define the pressure and suction sides
            npt_te (int, optional): number of points used to define trailing edge. Defaults to 20.
        """
        if not self.te_cut:
            t = np.linspace(0,1,npt_te)
            ps_te_x,ps_te_y = self.ps_arc.get_point(t)
            
            ss_te_x,ss_te_y = self.ss_arc.get_point(t)
            ss_te_x = np.flip(ss_te_x)
            ss_te_y = np.flip(ss_te_y)

            self.ss_te_pts = np.concatenate([ss_te_x,ss_te_y])
            self.ps_te_pts = np.concatenate([ps_te_x,ps_te_y])
        
        self.ps_x[-1]=ps_te_x[0]
        self.ps_y[-1]=ps_te_y[0]
        self.ss_x[-1]=ss_te_x[0]
        self.ss_y[-1]=ss_te_y[0]
        
        self.ps_bezier = bezier(self.ps_x, self.ps_y)
        self.ss_bezier = bezier(self.ss_x, self.ss_y)
        
        self.ps_pts = np.zeros(shape=(npts+npt_te,2))
        self.ss_pts = np.zeros(shape=(npts+npt_te,2))
        
        
        self.ps_pts[:npts,0],self.ps_y[:npts,1] = self.ps_bezier.get_point(npts)
        self.ss_pts[:npts,0],self.ss_y[:npts,1] = self.ss_bezier.get_point(npts)
        
        self.ps_pts[npts:,0] = self.ps_te_pts[1:,0]
        self.ps_pts[npts:,1] = self.ps_te_pts[1:,1]
        
        self.ss_pts[npts:,0] = self.ss_te_pts[1:,0]
        self.ss_pts[npts:,1] = self.ss_te_pts[1:,1]
    
    def plot_camber(self):
        """Plots the camber of the airfoil
        
        Returns:
            None
        """
        t = np.linspace(0,1,50)
        # plt.ion()
        marker_style = dict(markersize=8, markerfacecoloralt='tab:red')

        [xcamber, ycamber] = self.camber.get_point(t)
        
        plt.figure(num=1, clear=True)
        plt.plot(xcamber,ycamber, color='black', linestyle='solid', 
            linewidth=2)
        plt.plot(self.camber.x,self.camber.y, color='red', marker='o',linestyle='--',**marker_style)        
        plt.gca().set_aspect('equal')
        plt.show()
    
    def plot(self):
        """Plots the airfoil

        Returns:
            None
        """
        t = np.linspace(0,1,200)
        [xcamber, ycamber] = self.camber.get_point(t)
        [xPS, yPS] = self.ps_bezier.get_point(t)
        [xSS, ySS] = self.ss_bezier.get_point(t)

        plt.figure(num=1,clear=True)
        plt.plot(xcamber,ycamber, color='black', linestyle='solid', 
            linewidth=2)
        plt.plot(xPS,yPS, color='blue', linestyle='solid', 
            linewidth=2)
        plt.plot(xSS,ySS, color='red', linestyle='solid', 
            linewidth=2)
        plt.plot(self.ps_bezier.x,self.ps_bezier.y, color='blue', marker='o',markerfacecolor="None",markersize=8)
        plt.plot(self.ss_bezier.x,self.ss_bezier.y, color='red', marker='o',markerfacecolor="None",markersize=8)
        # Plot the line from camber to the control points
        # suction side
        for indx in range(len(self.ss_pts)):
            x = self.ss_pts[indx,0]
            y = self.ss_pts[indx,1]
            d = dist(x,y,xcamber,ycamber)
            min_indx = np.where(d == np.amin(d))[0][0]
            plt.plot([x,xcamber[min_indx]],[y,ycamber[min_indx]], color='black', linestyle='dashed')
        # pressure side
        for indx in range(0,len(self.ps_pts)):
            x = self.ps_pts[indx,0]
            y = self.ps_pts[indx,1]
            d = dist(x,y,xcamber,ycamber)
            min_indx = np.where(d == np.amin(d))[0][0]
            plt.plot([x,xcamber[min_indx]],[y,ycamber[min_indx]], color='black', linestyle='dashed')
        # Plot the Trailing Edge
        t = np.linspace(0,1,20)
        plt.plot(self.ps_te_pts[:,0],self.ps_te_pts[:,1], color='blue', linestyle='solid')

        plt.plot(self.ss_te_pts[:,0],self.ss_te_pts[:,1], color='red', linestyle='solid')
        plt.gca().set_aspect('equal')
        plt.show()