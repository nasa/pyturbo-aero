from enum import Enum
import numpy as np
from typing import List, Tuple
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from ..helper import bezier,arc,ellispe,exp_ratio,convert_to_ndarray
import matplotlib.pyplot as plt
from geomdl import NURBS, knotvector
    
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
    
    ss_te:arc
    ps_te:arc
    
    le_thickness:float
    ss_bezier:NURBS.BSpline
    ss_bezier_te:NURBS.BSpline
    ss_x:List[float]
    ss_y:List[float]    # this is rtheta
    
    ps_bezier:NURBS.BSpline
    ps_bezier_te:NURBS.BSpline
    ps_x:List[float]
    ps_y:List[float]
    
    ss_pts:npt.NDArray
    ps_pts:npt.NDArray
    ss_te_pts:npt.NDArray
    ps_te_pts:npt.NDArray
    te_cut:bool
    
    def __init__(self) -> None:
        pass
    
    def add_camber(self,alpha1:float,alpha2:float,
                   stagger:float,x1:float=0.1,x2:float=0.85,aggressivity:Tuple[float,float]=(0.8,0.9)) -> None:
        """Defines a camber line

        Args:
            alpha1 (float): inlet flow angle
            alpha2 (float): outlet flow angle
            stagger (float): stagger angle
            x1 (float): Location of LE angle control point [0,1] use percent axial chord. Default 0.1
            x2 (float): Location of TE angle control point [0,1] use percent axial chord. Default 0.85
            aggressivity (Tuple[float,float]): controls how aggressive the curve is at the exit
        """
        self.alpha1 = np.radians(alpha1)
        self.alpha2 = np.radians(alpha2)
        self.x1 = x1
        self.x2 = x2
        self.stagger = np.radians(stagger)
        
        h = np.tan(self.stagger)-(1-x2)*np.tan(self.alpha2)
        hm = np.tan(self.stagger)/2
        
        x = list(); rtheta = list()
        x = [0,x1]; rtheta = [0,x1*np.tan(self.alpha1)] 
        x_end = [x2,1]; rtheta_end = [h,np.tan(self.stagger)]
        
        np.tan(self.stagger)-(x2-x1)*np.tan(self.alpha2)
        
        
        xm = [x1+(x2-x1)*aggressivity[0]]
        hm = [rtheta[1] + (h-rtheta[1])*(aggressivity[1])]
        
        x.extend(xm)
        rtheta.extend(hm)
        x.extend(x_end)
        rtheta.extend(rtheta_end)
        
        x = [float(p) for p in x]   # convert all to floats
        rtheta = [float(p) for p in rtheta]
        
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

        self.ss_x.append(0); self.ss_y.append(0)
        self.ps_x.append(0); self.ps_y.append(0)
        
        self.le_thickness = thickness
        if abs(dy) <1E-6:
            self.ss_x.append(0)
            self.ss_y.append(-thickness)
            self.ps_x.append(0)
            self.ps_y.append(thickness)
        else:
            self.ps_x.append(thickness*(dx/m))
            self.ps_y.append(thickness*(-dy/m))
            
            self.ss_x.append(thickness*(-dx/m))
            self.ss_y.append(thickness*(dy/m))
            
    
   

    def add_ps_thickness(self,thickness_array:List[float],expansion_ratio:float=1.2):
        """builds the pressure side 

        Args:
            thickness_array (List[float]): thickness defined perpendicular to the camber line
            expansion_ratio (float, optional): Expansion ratio where thickness arrays are defined. Defaults to 1.2.
        """
        thickness_array = convert_to_ndarray(thickness_array)
        t =  exp_ratio(expansion_ratio,len(thickness_array)+2,1) # 1 point for the leading edge and 1 for TE starting point before radius is added
        x, y = self.camber.get_point(t)
        dx, dy = self.camber.get_point_dt(t)
        m = np.sign(thickness_array)*np.sqrt(thickness_array**2/(dx[1:-1]**2 + dy[1:-1]**2)) # magnitude
        for i in range(1,len(t)-1):
            self.ps_x.append(x[i]-dx[i]*m[i-1])
            self.ps_y.append(y[i]+dy[i]*m[i-1])
            
    def add_ss_thickness(self,thickness_array:List[float],expansion_ratio:float=1.2):
        """Builds the suction side

        Args:
            thickness_array (List[float]): Thickness array to use 
            expansion_ratio (float, optional): Expansion ratio where thickness arrays are defined. Defaults to 1.2.
        """
        thickness_array = convert_to_ndarray(thickness_array)
        t =  exp_ratio(expansion_ratio,len(thickness_array)+2,1)
        x, y = self.camber.get_point(t)
        dx, dy = self.camber.get_point_dt(t)
        # m^2 * (dx^2+dy^2) = thickness^2
        m = np.sign(thickness_array)*np.sqrt(thickness_array**2/(dx[1:-1]**2 + dy[1:-1]**2)) # magnitude
        for i in range(1,len(t)-1):
            self.ss_x.append(x[i]+dx[i]*m[i-1])
            self.ss_y.append(y[i]-dy[i]*m[i-1])
        

    def add_te_radius(self,radius_scale:float=0.6,wedge_ss:float=10,wedge_ps:float=10,elliptical:float=1):
        """Add a trailing edge that's rounded

        Args:
            radius_scale (float): 0 to 1 as to how the radius shrinks with respect to spacing between ss and ps last control points
            wedge_ss (float): suction side wedge angle
            wedge_ps (float): pressure side wedge angle 
            elliptical (float): 1=circular, any value >1 controls how it is elliptical
        """
        radius = radius_scale*(self.ps_y[-1] - self.ss_y[-1])/2
        radius_e = radius*elliptical
        xn,yn = self.camber.get_point(1)
        def dist(t):
            x,y = self.camber.get_point(t)
            d = np.sqrt((x-xn)**2+(y-yn)**2)
            return np.abs(radius_e-d)
        
        t = minimize_scalar(dist,bounds=[0,1])
        dx,dy = self.camber.get_point_dt(t.x)     # Gets the slope at the end
        xs,ys = self.camber.get_point(t.x)
        theta = np.degrees(np.atan2(dy,dx))
        x,y = self.camber.get_point(t.x)
        
        self.te_cut = False
        if radius_e == 1:
            ps_te = arc(x,y,radius,theta-wedge_ps+90,theta)
            ss_te = arc(x,y,radius,theta,theta-90+wedge_ss)
            
            ps_te_x, ps_te_y = ps_te.get_point(np.linspace(0,1,10))
            ss_te_x, ss_te_y = ss_te.get_point(np.linspace(0,1,10))
            self.ps_te_pts = np.column_stack([ps_te_x,ps_te_y])
            self.ss_te_pts = np.column_stack([ss_te_x,ss_te_y])
            self.ss_te_pts = np.flipud(self.ss_te_pts)
        else: # Create an ellispe
            ellispe_te = ellispe(x,y,np.sqrt((x-xn)**2+(y-yn)**2),radius,
                                 alpha_start=90-wedge_ps,
                                 alpha_stop=-90+wedge_ss)
            te_x, te_y = ellispe_te.get_point(np.linspace(0,1,20))
            
            theta = np.radians(theta)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])[:,:,0]
            self.ps_te_pts = np.matmul(rot,self.ps_te_pts.transpose()).transpose()
            self.ss_te_pts = np.matmul(rot,self.ss_te_pts.transpose()).transpose()

        # c = np.sqrt((xn-x)**2 + (yn-y)**2)/radius
        # ray = ray2D(xn,yn,-dx,-dy)
        # t_ps = ray.perpendicular(self.ps_te_pts[:,0],self.ps_te_pts[:,1]) 
        # t_ss = ray.perpendicular(self.ss_te_pts[:,0],self.ss_te_pts[:,1])
        # c_ps = np.flip((c-1)*((t_ps - t_ps.min() )/(t_ps.max()-t_ps.min()))+1)
        # c_ss = np.flip((c-1)*((t_ss - t_ss.min() )/(t_ss.max()-t_ss.min()))+1)
        
        # ps_te_pts[:,0] = c_ps*ps_te_pts[:,0]
        # ss_te_pts[:,0] = c_ss*ss_te_pts[:,0]
        
        # theta = -theta
        # rot = np.array([[np.cos(theta), -np.sin(theta)],
        #        [np.sin(theta), np.cos(theta)]])[:,:,0]
        # self.ps_te_pts = np.matmul(rot,ps_te_pts.transpose()).transpose()
        # self.ss_te_pts = np.matmul(rot,ss_te_pts.transpose()).transpose()
        
        
         
    def add_te_cut(self):
        """Cuts the trailing edge instead of having a rounded TE

        Args:
            radius (float): Trailing edge radius where to define the cut
        """        
        radius = (self.ps_y[-1] - self.ss_y[-1])/2
        
        x,y = self.camber.get_point(1)
        self.te_cut = True
        
        self.ss_te_pts = np.concat([1+0*np.linspace(y-radius,y,10), np.linspace(y-radius,y,10)],axis=1)
        self.ps_te_pts = np.flipud(np.concat([1+0*np.linspace(y+radius,y,10), np.linspace(y,y+radius,10)],axis=1))
        
    def build(self,npts:int):
        """Build the 2D Geometry 

        Args:
            npts (int): number of points to define the pressure and suction sides
            npt_te (int, optional): number of points used to define trailing edge. Defaults to 20.
        """    
        ps = NURBS.Curve(); # knots = # control points + order of curve
        ps.degree = 3 # cubic
        ctrlpts = np.concatenate([ 
                                    np.column_stack([self.ps_x, self.ps_y]),
                                    self.ps_te_pts
                                ])
        ctrlpts = np.column_stack([ctrlpts, ctrlpts[:,1]*0]) # Add empty column for z axis
        ps.ctrlpts = ctrlpts
        ps.delta = 1/npts
        # Knots = degree + npts 
        ps.knotvector = knotvector.generate(ps.degree,ctrlpts.shape[0])
        
        ss = NURBS.Curve()
        ss.degree = 3 # Cubic
        ctrlpts = np.concatenate([ 
                                    np.column_stack([self.ss_x, self.ss_y]),
                                    self.ss_te_pts
                                ])
        ctrlpts = np.column_stack([ctrlpts, ctrlpts[:,1]*0]) # Add empty column for z axis
        ss.ctrlpts = ctrlpts
        ss.knotvector = knotvector.generate(ss.degree,ctrlpts.shape[0])
        ss.delta = 1/npts
        
        self.ss_pts = np.array(ss.evalpts)
        self.ps_pts = np.array(ps.evalpts)
     
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
        plt.title('Camber')
        plt.show()
    
    def plot(self):
        """Plots the airfoil

        Returns:
            None
        """
        t = np.linspace(0,1,200)
        [xcamber, ycamber] = self.camber.get_point(t)
    

        plt.figure(num=1,clear=True)
        plt.plot(xcamber,ycamber, color='black', linestyle='solid', 
            linewidth=2)
        plt.plot(self.ps_pts[:,0],self.ps_pts[:,1],'b',label='pressure side')
        plt.plot(self.ss_pts[:,0],self.ss_pts[:,1],'r',label='suction side')
        plt.plot(self.ps_x,self.ps_y,'ob',label='ps ctrl pts')
        plt.plot(self.ss_x,self.ss_y,'or',label='ss ctrl pts')
        plt.plot(self.ps_te_pts[:,0],self.ps_te_pts[:,1],'ok',label='ps te ctrl pts')
        plt.plot(self.ss_te_pts[:,0],self.ss_te_pts[:,1],'om',label='ss te ctrl pts')
        plt.legend()
        plt.axis('scaled')
        plt.show()