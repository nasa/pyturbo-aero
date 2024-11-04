
from typing import List
from .centrif3D import Centrif3D
import copy 
from itertools import combinations
import numpy as np 

class Passage3D:
    blade:Centrif3D
    s_c:List[float]
    nblades:int 
    splitter:Centrif3D 
    n_splitter:int = 0
    
    def __init__(self,blade:Centrif3D,nblades:int) -> None:
        """Initialize a 3D Passage 

        Args:
            blade (Centrif3D): _description_
            nblades (int): _description_
        """
        self.blade=blade
        self.nblades=nblades
        
    
    def add_splitter(self,hub_start:float=0.5,hub_end:float=1,nsplitters:int=1):
        """Add a splitter blade 

        Args:
            hub_start (float): starting location along the hub. Defaults to 0.5
            hub_end (float): ending location along the hub. Defaults to 1
            nsplitters (int): number of splitters in between the blades 
            
        """
        splitter = copy.deepcopy(self.blade)
        splitter.set_blade_position(hub_start,hub_end)
        splitter.build(self.blade)
        self.splitter = splitter
        self.n_splitter = nsplitters
    
    def add_pattern(chord_scaling:List[float],pitch_to_chord_variations:List[float],min_pairs:int=1):

        pass
    
    def __rotate_centrif3D_x__(self,theta:float,blade:Centrif3D):
        
        theta = np.radians(theta)
        
        # Rotate in the y-z axis 
        mat = [[np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)],
               ]
        
        pass