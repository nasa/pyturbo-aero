from typing import List, Tuple
from .centrif2D import Centrif2D
from ..helper import bezier,convert_to_ndarray, csapi
import numpy as np 
import numpy.typing as npt 
from scipy.interpolate import PchipInterpolator

class Centrif3D():
    """Generates the 3D blade 
    """
    profiles:List[Centrif2D]
    leans:List[bezier]
    lean_cambers:List[float]
    splitters:List[np.ndarray]
    
    hub:npt.NDArray
    shroud:npt.NDArray
    blade_position:Tuple[float,float]
    
    ss_hub_fillet_loc:List[float]
    ss_hub_fillet:List[bezier]
    ps_hub_fillet_loc:List[float]
    ps_hub_fillet:List[bezier]
    fillet_r:float 
    
    ss_pts:npt.NDArray
    ps_pts:npt.NDArray
    
    def __init__(self,profiles:List[Centrif2D]):
        self.profiles = profiles
        self.leans = list()
        self.lean_cambers = list()
        
    def add_lean(self,lean_pts:List[float],percent_camber:float):
        """Adds lean to the 3D blade. Lean goes from hub to shroud

        Args:
            lean_pts (List[float]): points defining the lean. Example [-0.4, 0, 0.4] this is at the hub, mid, tip
            percent_camber (float): Where the lean is applied 
        """
        self.lean_cambers.append(percent_camber)
        self.leans.append(bezier(lean_pts,np.linspace(0,1,len(lean_pts))))
        
    def set_blade_position(self,t_start:float,t_end:float):
        """Sets the starting location of blade along the hub. 

        Args:
            t_start (float): starting percentage along the hub. 
            t_end (float): ending percentage along the hub
        """
        self.blade_position = (t_start,t_end)
    
    
    def add_hub(self,x:List[float,npt.NDArray],r:List[float,npt.NDArray]):
        """Adds Data for the hub 

        Args:
            x (List[float,npt.NDArray]): x coordinates for the hub 
            r (List[float,npt.NDArray]): radial coordinates for the hub 
        """
        self.hub = np.vstack([convert_to_ndarray(x),convert_to_ndarray(r)])
        
    
    def add_shroud(self,x:List[float,npt.NDArray],r:List[float,npt.NDArray]):
        """_summary_

        Args:
            x (List[float,npt.NDArray]): x coordinates for the hub 
            r (List[float,npt.NDArray]): radial coordinates for the hub 
        """
        self.shroud = np.vstack([convert_to_ndarray(x),convert_to_ndarray(r)])
    
    def add_hub_bezier_fillet(self,ps:bezier=None,ps_loc:float=0,ss:bezier=None,ss_loc:float=0,r:float=0):
        """Add hub bezier fillet

        Args:
            ps (List[bezier]): fillet for pressure side.
            ps_loc (List[float]): location of fillets on pressure side as a percentage of the camberline
            ss (List[bezier]): fillet for suction side 
            ss_loc (List[float]): location of fillets on suction side as a percentage of the camberline
            
        Example:
            x = [0,0.4,1] # Normalized by radius
            r = [0,0.4,1] # Normalized by radius 
            fillet=bezier(x,r)
            
        """
        if ps is not None:
            self.ps_hub_fillet.append(ps)
            self.ps_hub_fillet_loc.append(ps_loc)
        if ss is not None:
            self.ss_hub_fillet.append(ss)
            self.ss_hub_fillet_loc.append(ss_loc)
        if r>0:    
            self.fillet_r = r 
            
    def __apply_fillets__(self,npts_chord:int):
        """Apply fillets 

        Args:
            npts_chord (int): Number of points in chord
        """
        max_profile_indx_fillet = np.argmax(self.ss_pts[:,0,2] < self.fillet_r)
            
        # if the pressure side fillet is defined at leading edge
        if self.ps_hub_fillet_loc[0] == 0 and self.ss_hub_fillet_loc[0]>0:
            self.ss_hub_fillet.insert(0,self.ps_hub_fillet[0])
            self.ss_hub_fillet_loc.insert(0,self.ps_hub_fillet_loc[0])
        # if the suction side fillet is defined at leading edge
        elif self.ss_hub_fillet_loc[0] == 0 and self.ps_hub_fillet_loc[0]>0:
            self.ps_hub_fillet.insert(0,self.ss_hub_fillet[0])
            self.ss_hub_fillet_loc.insert(0,self.ss_hub_fillet_loc[0])
        
        if self.ps_hub_fillet_loc[-1] == 1 and self.ss_hub_fillet_loc[0]<1:
            self.ss_hub_fillet.insert(0,self.ps_hub_fillet[0])
            self.ss_hub_fillet_loc.insert(0,self.ps_hub_fillet_loc[0])
        # if the suction side fillet is defined at leading edge
        elif self.ss_hub_fillet_loc[-1] == 1 and self.ps_hub_fillet_loc[0]<1:
            self.ps_hub_fillet.insert(0,self.ss_hub_fillet[0])
            self.ss_hub_fillet_loc.insert(0,self.ss_hub_fillet_loc[0])
        # Not all cases were taken into account. Users should define something at leading edge and trailing edge and have fillets be equal. 
            
        # Extract the fillet from bezier curve 
        ss_fillet_shifts_temp = np.zeros(shape=(max_profile_indx_fillet,len(self.ss_hub_fillet),2))
        self.ss_hub_fillet_loc = convert_to_ndarray(self.ss_hub_fillet_loc); i = 0 
        for _,fillet in zip(self.ss_hub_fillet_loc,self.ss_hub_fillet):
            x,r = fillet.get_point()
            ss_fillet_shifts_temp[:,i,0] = x*self.fillet_r
            ss_fillet_shifts_temp[:,i,1] = r*self.fillet_r
            i+=1
            
        ps_fillet_shifts_temp = np.zeros(shape=(max_profile_indx_fillet,len(self.ps_hub_fillet),2))
        self.ps_hub_fillet_loc = convert_to_ndarray(self.ps_hub_fillet_loc); i = 0 
        for _,fillet in zip(self.ps_hub_fillet_loc,self.ps_hub_fillet):
            x,r = fillet.get_point(np.linspace(0,1,max_profile_indx_fillet))
            ps_fillet_shifts_temp[:,i,0] = x*self.fillet_r
            ps_fillet_shifts_temp[:,i,1] = r*self.fillet_r
            i+=1
        
        # Fillets have been defined now time to interpolate and add to the profiles
        t = np.linspace(0,1,npts_chord)
        ss_fillet = np.zeros(shape=(max_profile_indx_fillet, npts_chord, 2))
        self.ss_hub_fillet_loc = convert_to_ndarray(self.ss_hub_fillet_loc)
        for i in range(len(self.ss_hub_fillet)):
            ss_fillet[i,:,0] = csapi(self.ss_hub_fillet_loc, ss_fillet_shifts_temp[i,:,0])(t)
            ss_fillet[i,:,1] = csapi(self.ss_hub_fillet_loc, ss_fillet_shifts_temp[i,:,1])(t)
            # [span,fillets,(x,y)]
            
        ps_fillet = np.zeros(shape=(max_profile_indx_fillet, npts_chord, 2))
        self.ps_hub_fillet_loc = convert_to_ndarray(self.ps_hub_fillet_loc)
        for i in range(len(self.ps_hub_fillet)):
            ps_fillet[i,:,0] = csapi(self.ps_hub_fillet_loc, ps_fillet_shifts_temp[i,:,0])(t)
            ps_fillet[i,:,1] = csapi(self.ps_hub_fillet_loc, ps_fillet_shifts_temp[i,:,1])(t)
        
        # Apply fillets 
        for i in range(max_profile_indx_fillet):
            self.ss_pts[i,:,0] += ss_fillet[i,:,0]
            self.ss_pts[i,:,1] += ss_fillet[i,:,1]
            self.ps_pts[i,:,0] += ps_fillet[i,:,0]
            self.ps_pts[i,:,1] += ps_fillet[i,:,1]
                
    def build(self,npts_span:int=100,npts_chord:int=100):
        """Build the 3D Blade

        Args:
            npts_span (int, optional): number of points defining the span. Defaults to 100.
            npts_chord (int, optional): number of points defining the chord. Defaults to 100.
        """
        ss_pts_temp = np.zeros((len(self.profiles),npts_chord,3))
        ps_pts_temp = np.zeros((len(self.profiles),npts_chord,3))
        # Build and interpolate the blade 
        for i in range(len(self.profiles)):
            self.profiles[i].build(npts_chord)
            ss_pts_temp[i,:,:] = self.profiles[i].ss_pts
            ps_pts_temp[i,:,:] = self.profiles[i].ps_pts
        
        # Construct the new denser ss and ps 
        ss_pts = np.zeros((npts_span,npts_chord,3))
        ps_pts = np.zeros((npts_span,npts_chord,3))
        
        t_temp = np.linspace(0,1,len(self.profiles))
        t = np.linspace(0,1,npts_span)
        
        for i in range(npts_chord):
            ss_pts[:,i,0] = csapi(t_temp,ss_pts_temp[:,i,0])(t)
            ss_pts[:,i,1] = csapi(t_temp,ss_pts_temp[:,i,1])(t)
            ss_pts[:,i,2] = csapi(t_temp,ss_pts_temp[:,i,2])(t)
            
            ps_pts[:,i,0] = csapi(t_temp,ps_pts_temp[:,i,0])(t)
            ps_pts[:,i,1] = csapi(t_temp,ps_pts_temp[:,i,1])(t)
            ps_pts[:,i,2] = csapi(t_temp,ps_pts_temp[:,i,2])(t)
        
        # Apply Fillet radius to hub 
        if self.fillet_r>0:
            self.__apply_fillets__(npts_chord)
        
        # Scale to match hub and shroud curves 
        t = np.linspace(0,1,npts_chord)
        xhub = PchipInterpolator(t,self.hub[:,0])
        rhub = PchipInterpolator(t,self.hub[:,1])
        xhub = PchipInterpolator(t,self.shroud[:,0])
        rhub = PchipInterpolator(t,self.shroud[:,1])
        
        