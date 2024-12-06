from dataclasses import dataclass
from typing import List, Tuple, Union
from ..helper import convert_to_ndarray, Lean, line2D
import numpy.typing as npt 
import numpy as np 
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt 

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
    warp_displacements:List[float]          # percent of rmax-rmin
    warp_displacement_locs:List[float]      # 0 to 1
    
    t_start:float                           
    t_end:float

class Centrif:
    hub:npt.NDArray
    shroud:npt.NDArray
    profiles:List[BladeDesign]
    blade_position:Tuple[float,float] # Start and End positions
    
    func_xhub:PchipInterpolator
    func_rhub:PchipInterpolator
    func_xshroud:PchipInterpolator
    func_rshroud:PchipInterpolator
    
    camber_pts:npt.NDArray
    LE_Lean:List[Lean]
    TE_Lean:List[Lean]
    
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
        pass
        
        
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
            
    def add_LE_Wave(self,wave:Union[List[float],npt.NDArray]):
        self.le_wave = convert_to_ndarray(wave)
    
    def add_SS_Wave(self,wave:Union[List[float],npt.NDArray]):
        self.ss_wave = convert_to_ndarray[wave]
    
    
    def __get_camber_xr_point__(self,t_span:float,t_chord:float) -> npt.NDArray:
        # Returns the x,r point 
        shroud_pts = np.hstack([self.func_xshroud(t_chord),self.func_rshroud(t_chord)])
        hub_pts = np.hstack([self.func_xhub(t_chord),self.func_rhub(t_chord)])    
        l = line2D(hub_pts,shroud_pts)
        x,r = l.get_point(t_span)
        return np.array([x,r])
    
    def __get_camber_xr__(self,t_span:float) -> npt.NDArray:
        # Returns xr for the camber line 
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
        t_pos = np.linspace(self.blade_position[0], self.blade_position[-1],self.npts_chord)
                
        for profile in self.profiles:
            # r1 = starting radius, r2 = ending radius 
            t_start = profile.t_start; t_end = self.blade_position[-1]
            r1 = profile.percent_span*(self.func_rshroud(t_start)-self.func_rhub(t_start)) + self.func_rhub(t_start)
            r2 = profile.percent_span*(self.func_rshroud(t_end)-self.func_rhub(t_end)) + self.func_rhub(t_end)
            
            camb_2D_line = np.zeros((2+len(profile.warp_displacements),3)) # [[x,th,r]]
            camb_2D_line[0,1] = 0; camb_2D_line[-1,1] = profile.warp_angle
            l = line2D([camb_2D_line[0],r1],[camb_2D_line[-1],r2])  # [th,r]
            mp = np.array([-l.dx,l.dy]) # Perpendicular slope
            
            camber_bezier_xth = np.zeros(shape=(2+len(profile.warp_displacements),2))           # Bezier Control points in the x,theta plane
            camber_bezier_xth[0,:] = self.__get_camber_xr_point__(profile.percent_span,t_start) # Leading edge
            camber_bezier_xth[-1,:] = self.__get_camber_xr_point__(profile.percent_span,t_end)  # Trailing edge
            for locs,displacement in zip(profile.warp_displacement_locs,profile.warp_displacements):
                th,r = l.get_point(locs)
                camber_bezier_xth[0,1:] = mp*displacement+np.array([th,r]) # [th,r]

            
        
        
        
    def build(self,npts_span:int=100, npts_chord:int=100):
        self.npts_chord = npts_chord; self.npts_span = npts_span
        self.t_chord = np.linspace(0,1,npts_chord); self.t_span = np.linspace(0,1,npts_span)
        
        self.__build_camber__()
        pass
    def plot_front_view(self):
        pass