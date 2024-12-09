from dataclasses import dataclass
import enum
from typing import List, Tuple, Union
from ..helper import convert_to_ndarray, line2D, bezier
import numpy.typing as npt 
import numpy as np 
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

@enum
class WaveDirection:
    x:int = 0
    r:int = 1

@dataclass
class BladeDesign:
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
    profiles:List[BladeDesign]
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
    
    def add_profile(self,profile:BladeDesign):
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
        shroud_pts = np.hstack([self.func_xshroud(self.t_chord),self.func_rshroud(self.t_chord)])
        hub_pts = np.hstack([self.func_xhub(self.t_chord),self.func_rhub(self.t_chord)])    
        xr = np.zeros((self.npts_chord,2))
        for j in range(self.npts_chord):
            l = line2D(hub_pts[j,:],shroud_pts[j,:])
            xr[j,0],xr[j,1] = l.get_point(t_span)
        return xr
    
    def __build_camber__(self):
        t = np.linspace(0,1,self.hub.shape[0])
        self.func_xhub = PchipInterpolator(t,self.hub[:,0])
        self.func_rhub = PchipInterpolator(t,self.hub[:,1])
        self.func_xshroud = PchipInterpolator(t,self.shroud[:,0])
        self.func_rshroud = PchipInterpolator(t,self.shroud[:,1])
            
        # Build camber_xr        
        self.camber_t_th = list()

        for profile in self.profiles:
            # r1 = starting radius, r2 = ending radius 
            t_start = profile.t_start; t_end = self.blade_position[-1]
            r1 = profile.percent_span*(self.func_rshroud(t_start)-self.func_rhub(t_start)) + self.func_rhub(t_start)
            r2 = profile.percent_span*(self.func_rshroud(t_end)-self.func_rhub(t_end)) + self.func_rhub(t_end)
            
            camb_2D_line = np.zeros((2+len(profile.warp_displacements),3)) # [[x,th,r]]
            camb_2D_line[0,1] = 0; camb_2D_line[-1,1] = profile.warp_angle                      # Include warp angle
            l = line2D([camb_2D_line[0],r1],[camb_2D_line[-1],r2])  # [th,r]
            mp = np.array([-l.dx,l.dy]) # Perpendicular slope
            
            # warp_displacement_locs: percent chord
            # warp displacement: percent of warp_angle
            camber_bezier_t_th = np.zeros(shape=(2+len(profile.warp_displacements),2))           # Bezier Control points in the x,theta plane
            camber_bezier_t_th[0,:] = [0, 0]
            camber_bezier_t_th[-1,:] = [0, profile.warp_angle]
            
            i = 1
            for loc,displacement in zip(profile.warp_displacement_locs, profile.warp_displacements):
                th,_ = l.get_point(loc)
                camber_bezier_t_th[i,:] = mp*displacement*profile.warp_angle+np.hstack([loc,th]) # [th,r]
                i+=1
            self.camber_t_th.append(bezier(camber_bezier_t_th))
        
        
    def build(self,npts_span:int=100, npts_chord:int=100):
        self.npts_chord = npts_chord; self.npts_span = npts_span
        self.t_chord = np.linspace(0,1,npts_chord); self.t_span = np.linspace(0,1,npts_span)
        
        self.__build_camber__()
        pass
    def plot_camber(self):
        splitter_start = self.profiles[0].splitter_camber_start
        if splitter_start == 0:
            t = self.t_chord * (self.blade_position[1]-self.blade_position[0]) + self.blade_position[0]
        else:
            t = self.t_chord * (self.blade_position[1]-splitter_start) + splitter_start

        plt.figure(num=1)
        plt.plot()
    def plot_front_view(self):
        pass