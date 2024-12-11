from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union
from pyturbo.helper import convert_to_ndarray, line2D, bezier, exp_ratio
import numpy.typing as npt 
import numpy as np 
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

class WaveDirection(Enum):
    x:int = 0
    r:int = 1

@dataclass
class CentrifProfile:
    percent_span:float
    LE_Thickness:float
    TE_Radius:float
    
    LE_Metal_Angle:float
    TE_Metal_Angle:float
    
    LE_Metal_Angle_Loc:float
    TE_Metal_Angle_Loc:float
    
    ss_thickness:List[float]
    ps_thickness:List[float]
    
    warp_angle:float                        # angle of warp/theta
    warp_displacements:List[float]          # percent of warp_angle
    warp_displacement_locs:List[float]      # percent chord
    
    splitter_camber_start = 0
    

class Centrif:
    hub:npt.NDArray
    shroud:npt.NDArray
    profiles:List[CentrifProfile]
    blade_position:Tuple[float,float] # Start and End positions
    
    func_xhub:PchipInterpolator
    func_rhub:PchipInterpolator
    func_xshroud:PchipInterpolator
    func_rshroud:PchipInterpolator
    
    camber_t_th:List[bezier]
    LE_Lean:List[Tuple[float,float]]
    TE_Lean:List[Tuple[float,float]]
    LE_Stretch:List[Tuple[float,float]]
    TE_Stretch:List[Tuple[float,float]]
    
    LE_thickness:List[float]
    LE_Angle:List[float]
    LE_percent_span:List[float]
    TE_Angle:List[float]
    TE_radius:float
    __tip_clearance:float = 0

    le_wave:npt.NDArray
    ss_wave:npt.NDArray
    te_wave:npt.NDArray
    
    t_chord:npt.ArrayLike
    t_span:npt.ArrayLike
    npts_chord:int
    npts_span:int
    
    def __init__(self):
        self.profiles = list()
        self.blade_position = (0,1)
        
    
    def set_blade_position(self,t_start:float,t_end:float):
        """Sets the starting location of blade along the hub. 

        Args:
            t_start (float): starting percentage along the hub. 
            t_end (float): ending percentage along the hub
        """
        self.blade_position = (t_start,t_end)
        
    def add_hub(self,x:Union[float,npt.NDArray],r:Union[float,npt.NDArray]):
        """Adds Data for the hub 

        Args:
            x (Union[float,npt.NDArray]): x coordinates for the hub 
            r (Union[float,npt.NDArray]): radial coordinates for the hub 
        """
        self.hub = np.vstack([convert_to_ndarray(x),convert_to_ndarray(r)]).transpose()
        
    def add_shroud(self,x:Union[float,npt.NDArray],r:Union[float,npt.NDArray]):
        """_summary_

        Args:
            x (Union[float,npt.NDArray]): x coordinates for the hub 
            r (Union[float,npt.NDArray]): radial coordinates for the hub 
        """
        self.shroud = np.vstack([convert_to_ndarray(x),convert_to_ndarray(r)]).transpose()
    
    def add_profile(self,profile:CentrifProfile):
        """Add warp and adjustment

        Args:
            warp_angle (float): _description_
            warp_adjustment (List[WarpAdjustment]): 
        """
        self.profiles.append(profile)
        
    
    def tip_clearance(self):
        return self.__tip_clearance
            
    def add_LE_Wave(self,wave:Union[List[float],npt.NDArray],direction:WaveDirection):
        self.le_wave = convert_to_ndarray(wave)
    
    def add_SS_Wave(self,wave:Union[List[float],npt.NDArray]):
        self.ss_wave = convert_to_ndarray[wave]
    
    def __get_camber_xr_point__(self,t_span:float,t_chord:float) -> npt.NDArray:
        # Returns the x,r point. Doesn't require vertical line test 
        shroud_pts = np.hstack([self.func_xshroud(t_chord),self.func_rshroud(t_chord)])
        hub_pts = np.hstack([self.func_xhub(t_chord),self.func_rhub(t_chord)])    
        l = line2D(hub_pts,shroud_pts)
        x,r = l.get_point(t_span)
        return np.array([x,r])
    
    def __get_camber_xr__(self,t_span:float) -> npt.NDArray:
        # Returns xr for the camber line. Doesn't require vertical line test 
        shroud_pts = np.vstack([self.func_xshroud(self.t_chord),self.func_rshroud(self.t_chord)]).transpose()
        hub_pts = np.vstack([self.func_xhub(self.t_chord),self.func_rhub(self.t_chord)]).transpose()
        xr = np.zeros((self.npts_chord,2))
        for j in range(self.npts_chord):
            l = line2D(hub_pts[j,:],shroud_pts[j,:])
            xr[j,0],xr[j,1] = l.get_point(t_span)
        return xr
    
    def get_camber_points(self,i:int):
        """Get the camber in cylindrical coordinates x,r,th

        Args:
            i (int): camber index

        Returns:
            (npt.NDArray): x, r, theta 
            
        """
        xr = self.__get_camber_xr__(self.profiles[i].percent_span)
        _,th = self.camber_t_th[i].get_point(self.t_chord)
        xrth = np.hstack([xr,np.asmatrix(th).transpose()])
        return xrth
    
    def get_camber_length(self,i:int):
        xrth = self.get_camber_points(i)
        dr = np.diff(xrth[:,1])
        dx = np.diff(xrth[:,0])
        dth = np.diff(xrth[:,2])
        return np.sum(np.sqrt(dr**2+xrth[:,1]**2 * dth**2 + dx**2))
    
    def __build_camber__(self):
        t = np.linspace(0,1,self.hub.shape[0])
        self.func_xhub = PchipInterpolator(t,self.hub[:,0])
        self.func_rhub = PchipInterpolator(t,self.hub[:,1])
        self.func_xshroud = PchipInterpolator(t,self.shroud[:,0])
        self.func_rshroud = PchipInterpolator(t,self.shroud[:,1])
            
        # Build camber_xr        
        self.camber_t_th = list()
        i = 0
        for profile in self.profiles:
            # r1 = starting radius, r2 = ending radius 
            t_start = self.blade_position[0]; t_end = self.blade_position[-1]
            l = line2D([t_start,0],[t_end,np.radians(profile.warp_angle)])
            n = np.array([l.dx,-l.dy])/np.linalg.norm([l.dx,-l.dy])
            l.get_point(profile.LE_Metal_Angle_Loc)[1]
            # warp_displacement_locs: percent chord
            # warp displacement: percent of warp_angle
            camber_bezier_t_th = np.zeros(shape=(4+len(profile.warp_displacements),2))           # Bezier Control points in the t,theta plane
            camber_bezier_t_th[0,:] = [t_start, 0]
            camber_bezier_t_th[1,:] = [profile.LE_Metal_Angle_Loc, profile.LE_Metal_Angle_Loc*np.tan(np.radians(profile.LE_Metal_Angle))]
            camber_bezier_t_th[-2,:] = [profile.TE_Metal_Angle_Loc, (t_end-profile.TE_Metal_Angle_Loc)*np.tan(np.radians(profile.TE_Metal_Angle))+np.radians(profile.warp_angle)]
            camber_bezier_t_th[-1,:] = [t_end, np.radians(profile.warp_angle)]
            
            dx = self.__get_camber_xr_point__(profile.percent_span,self.t_chord[-1])[0] - self.__get_camber_xr_point__(profile.percent_span,self.t_chord[0])[0]
            dr = self.__get_camber_xr_point__(profile.percent_span,self.t_chord[-1])[1] - self.__get_camber_xr_point__(profile.percent_span,self.t_chord[0])[1]
            dth = np.radians(profile.warp_angle)
            camb_len = np.sum(np.sqrt(dr**2+dth**2 * dth**2 + dx**2))
            
            j = 2
            for loc,displacement in zip(profile.warp_displacement_locs, profile.warp_displacements):
                displacement = camb_len*displacement
                x1,y1 = l.get_point(loc)
                x2 = n[1]*displacement + x1
                y2 = n[0]*displacement + y1
                camber_bezier_t_th[j,0] = x2
                camber_bezier_t_th[j,1] = y2
                j+=1
            i+=1 
            self.camber_t_th.append(bezier(camber_bezier_t_th[:,0],camber_bezier_t_th[:,1]))
    
    def __build_hub_shroud__(self):
        self.hub_pts = np.vstack([
                self.func_xhub(np.linspace(0,1,self.npts_chord*2)),
                self.func_xhub(np.linspace(0,1,self.npts_chord*2))*0, 
                self.func_rhub(np.linspace(0,1,self.npts_chord*2))]).transpose()
        self.shroud_pts = np.vstack([
            self.func_xshroud(np.linspace(0,1,self.npts_chord*2)),
            self.func_xshroud(np.linspace(0,1,self.npts_chord*2))*0, 
            self.func_rshroud(np.linspace(0,1,self.npts_chord*2))]).transpose()
    
    def __apply_thickness__(self):
        t = exp_ratio(0,1)
        # Apply thickness in theta direction 
        for i,profile in enumerate(self.profiles):
            self.camber_t_th[i].get_point_dt()
        
    def build(self,npts_span:int=100, npts_chord:int=100):
        """Build the centrif blade 

        Args:
            npts_span (int, optional): _description_. Defaults to 100.
            npts_chord (int, optional): _description_. Defaults to 100.
        """
        self.npts_chord = npts_chord; self.npts_span = npts_span
        self.t_chord = np.linspace(0,1,npts_chord)
        self.t_span = np.linspace(0,1,npts_span)
        
        splitter_start = self.profiles[0].splitter_camber_start
        if splitter_start == 0:
            t = self.t_chord * (self.blade_position[1]-self.blade_position[0]) + self.blade_position[0]
        else:
            t = self.t_chord * (self.blade_position[1]-splitter_start) + splitter_start
        
        self.__build_camber__()
        self.__build_hub_shroud__()
        
    def plot_camber(self,plot_hub_shroud:bool=True):
        """Plot the camber line
        """
        t = np.linspace(0,1,100)
        fig = plt.figure(num=1,dpi=150)
        for i,b in enumerate(self.camber_t_th):
            [x,y] = b.get_point(t)        
            plt.plot(x, y,'-b',label=f"curve {i}")
            plt.plot(b.x, b.y,'or',label=f"curve {i}")
        plt.legend()
        plt.show()
        
        fig = plt.figure(num=2,dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        # Plots the camber and control points 
        k = 0
        for camber,profile in zip(self.camber_t_th,self.profiles):
            xc = np.zeros(shape=(len(camber.x),1))      # Bezier control points x,r,th
            rc = np.zeros(shape=(len(camber.x),1))
            thc = np.zeros(shape=(len(camber.x),1))
            for i in range(len(camber.x)):              # Camber x is really t 
                xc[i],rc[i] = self.__get_camber_xr_point__(profile.percent_span,camber.x[i])
                _,thc[i] = camber.get_point(camber.x[i])
            
            ax.plot3D(xc[1:-1],thc[1:-1],rc[1:-1],'or',markersize=4)
            ax.plot3D(xc[0],thc[0],rc[0],'ok',markersize=4)
            ax.plot3D(xc[-1],thc[-1],rc[-1],'ok',markersize=4)
            
            xrth = self.get_camber_points(k)
            ax.plot3D(xrth[:,0],xrth[:,2],xrth[:,1],'-b',linewidth=2)
            k+=1
        # Plots the hub and shroud 
        if plot_hub_shroud:
            ax.plot3D(self.hub_pts[:,0],self.hub_pts[:,1],self.hub_pts[:,2],'k',linewidth=2.5)
            ax.plot3D(self.shroud_pts[:,0],self.shroud_pts[:,1],self.shroud_pts[:,2],'k',linewidth=2.5)
        ax.view_init(68,-174)
        plt.axis('equal')
        plt.show()
        
    
    def plot_front_view(self):
        pass