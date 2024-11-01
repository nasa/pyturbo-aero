
from typing import List
from .centrif3D import Centrif3D

class Passage3D:
    blade:Centrif3D
    s_c:List[float]
    
    def __init__(self,blade:Centrif3D,nblades:int) -> None:
        pass
    
    def add_splitter(self,x_scale:float,loc:float=0):
        """Add a splitter blade 

        Args:
            x_scale (float): _description_
            loc (float): location of leading edge as a percent of camberline. 0 = at leading edge, 1 - splitter is at the end
        """
        for p in self.profiles:
            p.build()
        self.splitters.append()
    
    def add_pattern(chord_scaling:List[float],pitch_to_chord_variations:float):
        
        pass