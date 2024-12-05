from dataclasses import dataclass
from typing import List, Tuple, Union
from ..helper import convert_to_ndarray
import numpy.typing as npt 
import numpy as np 
from scipy.interpolate import PchipInterpolator

@dataclass
class WarpAdjustment:
    percent_span:float
    thickness:float
    percent_warp_line:float

class BladeDesign:
    percent_span:float
    LE_Thickness:float
    LE_Angle:float
    TE_Radius:float
    TE_Angle:float
    
    warp_angle:float
    warp_thicknesses:List[float]
    warp_t:List[float]
    
class Centrif:
    hub:npt.NDArray
    shroud:npt.NDArray
    blade_position:Tuple[float,float]
    
    func_xhub:PchipInterpolator
    func_rhub:PchipInterpolator
    func_xshroud:PchipInterpolator
    func_rshroud:PchipInterpolator
    
    camber_pts:npt.NDArray
    warp_angle:float # Hub, mid, tip etc 
    warp_adjustments:List[WarpAdjustment]
    LE_Lean:List[float]
    TE_Lean:List[float]
    
    LE_thickness:List[float]
    LE_Angle:List[float]
    LE_percent_span:List[float]
    TE_Angle:List[float]
    TE_radius:float
    __tip_clearance:float = 0 
    

    def __init__(self):
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
    
    def add_warp(self,warp_angle:float, warp_adjustment:List[WarpAdjustment]):
        """Add warp and adjustment

        Args:
            warp_angle (float): _description_
            warp_adjustment (List[WarpAdjustment]): 
        """
        self.warp_angle = warp_angle
        self.warp_adjustments = warp_adjustment
    
    def tip_clearance(self):
        return self.__tip_clearance
    def add_LE(self,angle:float,thickness:float,percent_span:float):
        
    def add_LE_Wave(self,wave:Union[List[float],npt.NDArray]):
        self.LE_Wave = convert_to_ndarray(wave)
        
    def plot_front(self):
