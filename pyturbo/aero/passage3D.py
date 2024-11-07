
from typing import List
from .centrif3D import Centrif3D
import copy 
from itertools import combinations, combinations_with_replacement
import numpy as np 
from dataclasses import dataclass
import matplotlib.pyplot as plt 

@dataclass
class PatternPairCentrif:
    chord_scaling:float
    rotation_ajustment:float
    
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
        """Patterns are repeated for n number of blades but the goal is to repeat without patterns in a row. 

        Args:
            pair (PatternPairCentrif): Create a pattern pair. 
        """
        self.patterns.append(pair)
        
    
    def __rotate_centrif3D_x__(self,theta:float,blade:Centrif3D):
        """Rotate a blade around the x axis

        Args:
            theta (float): angle of rotation
            blade (Centrif3D): blade object
        """
        theta = np.radians(theta)
        
        # Rotate in the y-z axis 
        mat = [[np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]]
        
        yz = np.hstack([blade.ss_pts[:,:,1].transpose(),blade.ss_pts[:,:,2].transpose()])
        res = np.matmul(mat,yz)
        blade.ss_pts[:,1] = res[:,0]
        blade.ss_pts[:,2] = res[:,1]
        
        yz = np.hstack([blade.ps_pts[:,:,1].transpose(),blade.ps_pts[:,:,2].transpose()])
        res = np.matmul(mat,yz)
        blade.ps_pts[:,1] = res[:,0]
        blade.ps_pts[:,2] = res[:,1]
    
    def __apply_pattern__(self,nblades:int,blades:List[Centrif3D],rotation_angles:List[float]):
        """Apply patterns modifications to the design. The intent of this is to reduce the peaks associated with a compressor by creating small variations in the geometry. 
        
        Args:
            nblades (int): number of blades
            blades (List[Centrif3D]): List of blades 
        """
        total_combinations = []

        if len(self.patterns) == 1:
            total_combinations = [self.patterns[0] for _ in range(nblades)]
        else:
            for i in range(len(self.patterns)-1):
                combos = list(combinations(self.patterns,i))
                temp = [cc for c in combos for cc in c]
            total_combinations.extend(temp)
            assert len(total_combinations)>nblades, "Combinations should be more than number of blades. Please add more patterns."
            total_combinations[:nblades]
        
        for i,pattern in enumerate(total_combinations):
            start_pos = blades[i].blade_position[0]
            end_pos = blades[i].blade_position[1]
            
            end_pos -= pattern.chord_scaling
            blades[i].set_blade_position(start_pos,end_pos)
            pattern.pitch_to_chord_scaling
            rotation_angles[i] += pattern.rotation_ajustment
            
            blades[i].build(blades[i].npts_chord,blades[i].npts_span)
            
            # Rotate the blade 
            self.__rotate_centrif3D_x__(rotation_angles[i],blades[i])

    
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
        self.__apply_pattern__() # Applies the pattern and rotates the blade 
    
    def plot(self):
        """Plots the generated design 
        """
        fig = plt.figure(num=1,dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        for blade in self.blades:
            ax.plot3D(blade.hub_pts[:,0],blade.hub_pts[:,0]*0,blade.hub_pts[:,2],'k')
            ax.plot3D(blade.shroud_pts[:,0],blade.shroud_pts[:,0]*0,blade.shroud_pts[:,2],'k')
            for i in range(blade.ss_pts.shape[0]):
                ax.plot3D(blade.ss_pts[i,:,0],blade.ss_pts[i,:,1],blade.ss_pts[i,:,2],'r')
                ax.plot3D(blade.ps_pts[i,:,0],blade.ps_pts[i,:,1],blade.ps_pts[i,:,2],'b')
        ax.view_init(azim=90, elev=45)
        ax.set_xlabel('x-axial')
        ax.set_ylabel('rth')
        ax.set_zlabel('r-radial')
        plt.show()