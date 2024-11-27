
from typing import List
import numpy.typing as npt
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
    blade:Centrif3D = None
    s_c:List[float]
    splitter:Centrif3D = None 
    
    blades:Centrif3D = []
    splitters:Centrif3D = []
    patterns:List[PatternPairCentrif] = []
    hub_pts: npt.NDArray
    shroud_pts:npt.NDArray
    
    
    def __init__(self,blade:Centrif3D) -> None:
        """Initialize a 3D Passage 

        Args:
            blade (Centrif3D): _description_
        """
        self.blade=blade
        dx = self.blade.ss_pts[0,-1,0] - self.blade.ss_pts[0,0,0]
        drth = self.blade.ss_pts[0,-1,1] 
        self.blade_stagger = np.degrees(np.arctan2(drth,dx))
        self.patterns.append(PatternPairCentrif(chord_scaling=1,rotation_ajustment=0)) # adds a default pattern, this is no modification
        
        
    
    def add_splitter(self,splitter:Centrif3D):
        self.splitter = splitter
    
    def add_pattern_pair(self,pair:PatternPairCentrif):
        """Patterns are repeated for n number of blades but the goal is to repeat without patterns in a row. 

        Args:
            pair (PatternPairCentrif): Create a pattern pair. 
        """
        self.patterns.append(pair)
        
    
    def __rotate_hub_shroud__(self,resolution:int=20):
        """Rotate the hub and shroud 

        Args:
            resolution (int, optional): Number of rotations. Defaults to 20.
        """
        rotations = np.linspace(0,1,resolution)*360
        npts = self.blades[0].hub_pts.shape[0]
        
        xhub = self.blades[0].hub_pts[:,0]
        rhub = self.blades[0].hub_pts[:,2]
        zhub = self.blades[0].hub_pts[:,0]*0
        
        xshroud = self.blades[0].shroud_pts[:,0]
        rshroud = self.blades[0].shroud_pts[:,2]
        zshroud = self.blades[0].shroud_pts[:,0]*0
        
        self.hub_pts = np.zeros((npts,resolution,3))
        self.shroud_pts = np.zeros((npts,resolution,3))
        
        for i in range(len(rotations)):
            theta = np.radians(rotations[i])
            mat = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

            # Rotate the hub and shroud 
            z1,r1 = np.matmul(mat,np.vstack([zhub,rhub]))
            z2,r2 = np.matmul(mat,np.vstack([zshroud,rshroud]))
            
            self.hub_pts[:,i,0] = xhub
            self.hub_pts[:,i,1] = z1
            self.hub_pts[:,i,2] = r1
            
            self.shroud_pts[:,i,0] = xshroud
            self.shroud_pts[:,i,1] = z2
            self.shroud_pts[:,i,2] = r2
    
    def __apply_pattern__(self,):
        """Apply patterns modifications to the design. The intent of this is to reduce the peaks associated with a compressor by creating small variations in the geometry. 
        
        Args:
            nblades (int): number of blades
            blades (List[Centrif3D]): List of blades 
        """
        total_combinations = []
        nblades = len(self.blades)

        if len(self.patterns) == 1:
            total_combinations = [self.patterns[0] for _ in range(nblades)]
        else:
            for i in range(1,len(self.patterns)):
                combos = list(combinations(self.patterns,i))
                temp = [cc for c in combos for cc in c]
                total_combinations.extend(temp)
            assert len(total_combinations)>nblades, "Combinations should be more than number of blades. Please add more patterns."
            total_combinations = total_combinations[:nblades]
        
        for i,pattern in enumerate(total_combinations):
            start_pos = self.blades[i].blade_position[0]
            end_pos = self.blades[i].blade_position[1]
            
            start_pos = end_pos - (end_pos-start_pos)*pattern.chord_scaling
            self.blades[i].set_blade_position(start_pos,end_pos)
            
            self.blades[i].build(self.blades[i].npts_span,self.blades[i].npts_chord)
            self.blades[i].rotate(self.blades[i].rotation_angle+pattern.rotation_ajustment) # Rotate the blade 
                 
    def add_cut_te(self,percent_r:float):
        """Looks in the R-Z Plane and cuts the geometry 

        Args:
            percent_r (float): percentage of ending radius to cut the geometry
        """
        pass 
    
    def build(self,nblades:int,hub_resolution:int=-1):
        """Build the centrif geometry 

        Args:
            nblades (int): number of blades 
            hub_resolution (int): how many rotations of the hub and shroud. Defaults to -1 which is same as number of blades + number of splitters
            
        """
        # Lets check
        theta_blade = 360/nblades
        blades = [copy.deepcopy(self.blade) for _ in range(nblades)]
        # Lets rotate the blades 
        theta = 0
        for b in blades:
            b.rotate(theta)
            theta += theta_blade
        self.blades = blades
        
        theta = theta_blade/2
        if self.splitter is not None:
            splitters = []
            while theta<=360:
                splitters.append(copy.deepcopy(self.splitter))
                splitters[-1].rotate(theta)
                theta += theta_blade
            self.splitters = splitters
        # self.__apply_pattern__() # Applies the pattern and rotates the blade 
        
        if hub_resolution<=0:
            hub_resolution = len(blades) # +len(splitters)
        self.__rotate_hub_shroud__(hub_resolution)
    
    def plot(self,num_blades:int=-1,num_splitters:int=-1):
        """Plots the generated design

        Args:
            num_blades (int, optional): number of blades. Defaults to -1.
            num_splitters (int, optional): number of splitters. Defaults to -1.
        """
        if num_blades<0:
            num_blades = len(self.blades)
        if num_splitters<0:
            num_splitters = len(self.splitters)
        fig = plt.figure(num=1,dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(num_blades):
            blade = self.blades[i]
            for i in range(blade.ss_pts.shape[0]):
                ax.plot3D(blade.ss_pts[i,:,0],blade.ss_pts[i,:,1],blade.ss_pts[i,:,2],'r')
                ax.plot3D(blade.ps_pts[i,:,0],blade.ps_pts[i,:,1],blade.ps_pts[i,:,2],'b')
                
        for i in range(num_splitters):
            splitter = self.splitters[i]
            for i in range(splitter.ss_pts.shape[0]):
                ax.plot3D(splitter.ss_pts[i,:,0],splitter.ss_pts[i,:,1],splitter.ss_pts[i,:,2],'m')
                ax.plot3D(splitter.ps_pts[i,:,0],splitter.ps_pts[i,:,1],splitter.ps_pts[i,:,2],'m')
        
        resolution,npts,_ = self.hub_pts.shape
        for i in range(resolution):
            ax.plot3D(self.hub_pts[i,:,0],self.hub_pts[i,:,1],self.hub_pts[i,:,2],'k',alpha=0.1)
            ax.plot3D(self.shroud_pts[i,:,0],self.shroud_pts[i,:,1],self.shroud_pts[i,:,2],'k',alpha=0.1)
        
        for j in range(npts):
            ax.plot3D(self.hub_pts[:,j,0],self.hub_pts[:,j,1],self.hub_pts[:,j,2],'k',alpha=0.1)
            ax.plot3D(self.shroud_pts[i,:,0],self.shroud_pts[i,:,1],self.shroud_pts[i,:,2],'k',alpha=0.1)
            
        ax.view_init(azim=90, elev=45)
        ax.set_xlabel('x-axial')
        ax.set_ylabel('rth')
        ax.set_zlabel('r-radial')
        
        fig = plt.figure(num=2,dpi=150)
        ax = fig.add_subplot(111)
        
        for i in range(num_blades):
            blade = self.blades[i]
            for i in range(blade.ss_pts.shape[0]):
                ax.plot(blade.ss_pts[i,:,1],blade.ss_pts[i,:,2],'r')
                ax.plot(blade.ps_pts[i,:,1],blade.ps_pts[i,:,2],'b')
                
        for i in range(num_splitters):
            splitter = self.splitters[i]
            for i in range(splitter.ss_pts.shape[0]):
                ax.plot(splitter.ss_pts[i,:,1],splitter.ss_pts[i,:,2],'m')
                ax.plot(splitter.ps_pts[i,:,1],splitter.ps_pts[i,:,2],'m')
        
        ax.set_xlabel('rth')
        ax.set_ylabel('r-radial')
        plt.axis('scaled')
        plt.show()