
from typing import List
from .centrif3D import Centrif3D
import copy 
from itertools import combinations
import numpy as np 
from dataclasses import dataclass

@dataclass
class PatternPairCentrif:
    npairs:int
    chord_scaling:float
    pitch_to_chord_scaling:float
    
class Passage3D:
    blade:Centrif3D
    s_c:List[float]
    splitter:Centrif3D 
    
    blades:Centrif3D = []
    splitters:Centrif3D = []
    patterns:List[PatternPairCentrif] = []
    
    def __init__(self,blade:Centrif3D) -> None:
        """Initialize a 3D Passage 

        Args:
            blade (Centrif3D): _description_
        """
        self.blade=blade
        self.patterns.append(PatternPairCentrif(1,1,1)) # adds a default pattern, this is no modification
        
        
    
    def add_splitter(self,hub_start:float=0.5,hub_end:float=1):
        """Add a splitter blade 

        Args:
            hub_start (float): starting location along the hub. Defaults to 0.5
            hub_end (float): ending location along the hub. Defaults to 1
            
        """
        splitter = copy.deepcopy(self.blade)
        splitter.set_blade_position(hub_start,hub_end)
        splitter.build(self.blade)
        self.splitter = splitter
    
    def add_pattern_pair(self,pair:PatternPairCentrif):
        """_summary_

        Args:
            pair (PatternPairCentrif): _description_
        """
        self.patterns.append(pair)
        
    def add_pattern(chord_scaling:List[float],pitch_to_chord_variations:List[float],min_pairs:int=1):
        """Creates a modification pattern. These patterns are randomly distributed along the circumference much like how a car tire is. The intent is to create small variations in the geometry to reduce noise. 

        Args:
            chord_scaling (List[float]): indicates how the chord is changing with respect to some normal pattern. 
            pitch_to_chord_variations (List[float]): _description_
            min_pairs (int, optional): _description_. Defaults to 1.
            
        Example: 
            chord_scaling = [0.98, 0.97]
            pitch_to_chord = [0.05,0.02]
            
        """
        pass
    
    def __rotate_centrif3D_x__(self,theta:float,blade:Centrif3D):
        """Rotate a blade around the x axis

        Args:
            theta (float): angle of rotation
            blade (Centrif3D): blade object
        """
        theta = np.radians(theta)
        
        # Rotate in the y-z axis 
        mat = [[np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)],
               ]
        yz = np.hstack([blade.ss_pts[:,:,1].transpose(),blade.ss_pts[:,:,2].transpose()])
        res = np.matmul(mat,yz)
        blade.ss_pts[:,1] = res[:,0]
        blade.ss_pts[:,2] = res[:,1]
        
        yz = np.hstack([blade.ps_pts[:,:,1].transpose(),blade.ps_pts[:,:,2].transpose()])
        res = np.matmul(mat,yz)
        blade.ps_pts[:,1] = res[:,0]
        blade.ps_pts[:,2] = res[:,1]
    
    def __apply_pattern__(self):
        """Apply patterns modifications to the design. 
        The intent of this is to reduce the peaks associated with a compressor by creating small variations in the geometry. 
        """
        for p in self.patterns:
            
    
    
    def build(self,nblades:int,bSplitter:bool=False):
        """Build the centrif geometry 

        Args:
            nblades (int): number of blades 
            bSplitter (bool): add splitter 
        """
        theta_blade = 360/nblades
        blades = [copy.deepcopy(self.blade) for _ in range(nblades)]
        # Lets rotate the blades 
        theta = 0 
        for b in blades:
            self.__rotate_centrif3D_x__(theta,b)
            theta += theta_blade
        self.blades = blades 
        
        if bSplitter:
            splitters = []
            theta = theta_blade/2
            while theta<=360:
                splitters.append(self.splitter)
                self.__rotate_centrif3D_x__(theta,splitters[-1])
                theta += theta_blade
            self.splitters = splitters
        self.__apply_pattern__()