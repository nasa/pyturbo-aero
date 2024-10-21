from typing import List
from .centrif2D import Centrif2D
from ..helper import bezier
import numpy as np 

class Centrif3D():
    profiles:List[Centrif2D]
    leans:List[bezier]
    splitters:List[np.ndarray]
    
    def __init__(self,profiles:List[Centrif2D]):
        self.profiles = profiles
        self.leans = list()
        
    def add_lean(self,lean_pts:List[float],percent_camber:float):
        """Adds lean to the 3D blade. Lean goes from hub to shroud

        Args:
            lean_pts (List[float]): points defining the lean. Example [-0.4, 0, 0.4] this is at the hub, mid, tip
            percent_camber (float): Where the lean is applied 
        """
        self.leans.append(bezier(lean_pts,np.linspace(0,1,len(lean_pts))))
        
    def add_splitter(self,x_scale:float,loc:float=0):
        """Add a splitter blade 

        Args:
            x_scale (float): _description_
            loc (float): location of leading edge as a percent of camberline. 0 = at leading edge, 1 - splitter is at the end
        """
        for p in self.profiles:
            p.build()
        self.splitters.append()    
    
    def add_hub(self,x:List[float],r:List[float]):
        """Adds Data for the hub 

        Args:
            x (List[float]): x coordinates for the hub 
            r (List[float]): radial coordinates for the hub 
        """