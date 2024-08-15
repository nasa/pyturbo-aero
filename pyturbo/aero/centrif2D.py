from enum import Enum
import numpy as np
from typing import List
from math import cos,sin,radians,degrees,pi,atan2,sqrt,atan
from scipy.optimize import minimize_scalar
from ..helper import bezier,line2D,ray2D,arc,ray2D_intersection,exp_ratio,convert_to_ndarray,derivative,dist,pw_bezier2D,bisect
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import copy

    
    
class Centrif2D:
    """Constructing the 2D profiles for a centrif compressor or turbine
    Profiles are constructed in the meridional plane and fitted between hub and shroud
    """
    
    def __init__(self) -> None:
        pass
    def add_camber(self,alpha1:float,alpha2:float,stagger:float,x1:float,x2:float) -> None:
        pass
    
    def add_le_thickness(self,thickness:float):
        pass

    def add_ss_thickness(self,thickness_array:List[float],expansion_ratio:float=1.2):
        pass

    def add_ps_thickness(self,thickness_array:List[float],expansion_ratio:float=1.2):
        pass

    def add_te_radius(self,radius:float,wedge_ss:float,wedge_ps:float):
        pass

    def add_te_cut(self):
        pass

    def build(self):
        pass