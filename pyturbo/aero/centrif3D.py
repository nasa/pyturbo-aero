from typing import List, Tuple, Union
from .centrif2D import Centrif2D
from ..helper import bezier,convert_to_ndarray, csapi, line2D
import numpy as np 
import numpy.typing as npt 
from scipy.interpolate import PchipInterpolator
from pyturbo.aero.airfoil3D import StackType
import matplotlib.pyplot as plt 

class Centrif3D():
    """Generates the 3D blade 
    """
    profiles:List[Centrif2D]
    stacktype:StackType 
    leans:List[bezier]
    lean_percent_spans:List[float]
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

    hub_pts:npt.NDArray
    shroud_pts:npt.NDArray
    
    __tip_clearance:float = 0 
    
    func_xhub:PchipInterpolator
    func_rhub:PchipInterpolator
    func_xshroud:PchipInterpolator
    func_rshroud:PchipInterpolator
    
    @property
    def tip_clearance(self):
        return self.__tip_clearance
    
    @tip_clearance.setter
    def tip_clearance(self,val:float=0):
        """Set the tip clearance 

        Args:
            val (float, optional): tip clearance as a percentage of the hub to shroud. Defaults to 0.
        """
        self.__tip_clearance = val
    
    def __init__(self,profiles:List[Centrif2D],stacking:StackType=StackType.leading_edge):
        self.profiles = profiles
        self.leans = list()
        self.lean_cambers = list()
        self.stacktype = stacking
        self.lean_cambers = list()
        self.leans = list() 
        self.fillet_r = 0
        
    def add_lean(self,lean_pts:List[float],percent_span:float):
        """Adds lean to the 3D blade. Lean goes from hub to shroud

        Args:
            lean_pts (List[float]): points defining the lean. Example [-0.4, 0, 0.4] this is at the hub, mid, tip
            percent_span (float): Where the lean is applied 
        """
        self.lean_percent_spans.append(percent_span)
        self.leans.append(bezier(lean_pts,np.linspace(0,1,len(lean_pts))))
        
    def set_blade_position(self,t_start:float,t_end:float):
        """Sets the starting location of blade along the hub. 

        Args:
            t_start (float): starting percentage along the hub. 
            t_end (float): ending percentage along the hub
        """
        self.blade_position = (t_start,t_end)
      
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
           
    def __apply_stacking__(self):
        if self.stacktype == StackType.centroid:
            c_x = list()
            c_rtheta = list()
            for p in self.profiles:
                c_x.append(0.5*(np.mean(p.ps_pts[:,0]) + np.mean(p.ss_pts[:,0])))
                c_rtheta.append(0.5*(np.mean(p.ps_pts[:,1]) + np.mean(p.ss_pts[:,1])))

            # Relocate centroids to line up
            i = 1
            for p in self.profiles[1:]:
                p.ps_pts[:,0]+=c_x[0]-c_x[i]
                p.ps_pts[:,1]+=c_rtheta[0]-c_rtheta[i]
                
                p.ss_pts[:,0]+=c_x[0]-c_x[i]
                p.ss_pts[:,1]+=c_rtheta[0]-c_rtheta[i]
            
        elif self.stacktype == StackType.trailing_edge:
            te_x = list()
            te_rtheta = list()
            for p in self.profiles:
                te_x.append(p.ps_pts[:,-1])
                te_rtheta.append(p.ps_pts[:,-1])

            # Relocate centroids to line up
            i = 1
            for p in self.profiles[1:]:
                p.ps_pts[:,0]+=te_x[0]-te_x[i]
                p.ps_pts[:,1]+=te_rtheta[0]-te_rtheta[i]
                
                p.ss_pts[:,0]+=te_x[0]-te_x[i]
                p.ss_pts[:,1]+=te_rtheta[0]-te_rtheta[i]
    
    def __stretch_profiles__(self,npts_span:int,npts_chord:int):
        """Stretch the profiles in the x and y direction to match camber

        Args:
            npts_span (int): number of points defining the span
            npts_chord (int): number of points defining the chord 
        """
        # Lets get the length from start to finish
        t = np.linspace(self.blade_position[0],self.blade_position[1],npts_chord)
        xh = self.func_xhub(t)
        rh = self.func_rhub(t)
        
        hub_length_of_blade = np.sum(np.sqrt(np.diff(xh)**2+np.diff(rh)**2))
        _,cambers = self.__percent_camber__(npts_span,npts_chord)
        for i in range(cambers.shape[0]):
            self.ss_pts[i,:]*=hub_length_of_blade/cambers[-1]  # Scale the blade profiles to the hub length 
            self.ps_pts[i,:]*=hub_length_of_blade/cambers[-1]

    def __apply_tip_gap__(self):
        """Apply tip gap and construct new functions that define the hub and shroud 
        """
        # Scale to match hub and shroud curves 
        t = np.linspace(0,1,self.hub.shape[0])
        # Implement Tip gap
        hub = self.hub.copy()       # create a copy 
        shroud = self.shroud.copy()
        if self.tip_clearance>0:
            for i in self.hub.shape[0]:
                xhub = self.hub[i,0]
                rhub = self.hub[i,1]
                xshroud = self.shroud[i,0]
                rshroud = self.shroud[i,1]
                l = line2D([xhub,rhub],[xshroud,rshroud])
                x,r = l.get_point(1-self.tip_clearance)
                shroud[i,0] = x
                shroud[i,1] = r
        
        self.func_xhub = PchipInterpolator(t,hub[:,0])
        self.func_rhub = PchipInterpolator(t,hub[:,1])
        self.func_xshroud = PchipInterpolator(t,shroud[:,0])
        self.func_rshroud = PchipInterpolator(t,shroud[:,1])
        
    def __scale_profiles__(self,npts_span:int,npts_chord:int):
        """scale the profiles to fit into the hub and shroud 

        Args:
            npts_span (int): number of points in the spanwise direction 
            npts_chord (int): number of points in the chordwise direction
        """
        thub_to_shroud = np.linspace(0,1,npts_span)
        percent_camber,_ = self.__percent_camber__(npts_span,npts_chord)
        
        t = np.zeros((npts_span,npts_chord))
        for i in range(npts_span):
            t[i,:] = self.blade_position[0]+(self.blade_position[1]-self.blade_position[0])*percent_camber[i,:]
        
        for j in range(npts_chord):
            xhub = self.func_xhub(t[:,j])
            rhub = self.func_rhub(t[:,j])
            
            xshroud = self.func_xshroud(t[:,j])
            rshroud = self.func_rshroud(t[:,j])
            l = line2D([xhub,rhub],[xshroud,rshroud])
            x,r = l.get_point(np.linspace(0,1,npts_span))
            for i in range(npts_span):
                self.ps_pts[i,j,0]=x[i]
                self.ps_pts[i,j,2]=r[i]
                
                self.ss_pts[i,j,0]=x[i]
                self.ss_pts[i,j,2]=r[i]
            
            # # Shift all profiles
            # for j in range(npts_chord):
            #     l = line2D([xhub,rhub],[xshroud,rshroud])
            #     xhub_to_shroud, rhub_to_shroud = l.get_point(np.linspace(0,1,npts_span))
                
            #     self.ps_pts[:,j,0]=xhub_to_shroud
            #     self.ps_pts[:,j,2]=rhub_to_shroud
                
            #     self.ss_pts[:,j,0]=xhub_to_shroud
            #     self.ss_pts[:,j,2]=rhub_to_shroud
        
        self.hub_pts = np.vstack([
                self.func_xhub(np.linspace(0,1,npts_chord*2)),
                self.func_xhub(np.linspace(0,1,npts_chord*2))*0, 
                self.func_rhub(np.linspace(0,1,npts_chord*2))]).transpose()
        self.shroud_pts = np.vstack([
            self.func_xshroud(np.linspace(0,1,npts_chord*2)),
            self.func_xshroud(np.linspace(0,1,npts_chord*2))*0, 
            self.func_rshroud(np.linspace(0,1,npts_chord*2))]).transpose()

    def __interpolate__(self,npts_span:int,npts_chord:int):
        """Interpolate the geometry to make it denser

        Args:
            npts_span (int): number of points in the spanwise direction 
            npts_chord (int): number of points in the chordwise direction
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
            ss_pts[:,i,0] = csapi(t_temp,ss_pts_temp[:,i,0],t)
            ss_pts[:,i,1] = csapi(t_temp,ss_pts_temp[:,i,1],t)
            ss_pts[:,i,2] = csapi(t_temp,ss_pts_temp[:,i,2],t)
            
            ps_pts[:,i,0] = csapi(t_temp,ps_pts_temp[:,i,0],t)
            ps_pts[:,i,1] = csapi(t_temp,ps_pts_temp[:,i,1],t)
            ps_pts[:,i,2] = csapi(t_temp,ps_pts_temp[:,i,2],t)
        
        self.ps_pts = ps_pts
        self.ss_pts = ss_pts
    
    def __percent_camber__(self,npts_span:int,npts_chord:int) -> npt.NDArray:
        """Gets the percent along the camber line for each profile and interpolates that to fill the interpolated blade. 

        Args:
            npts_span (int): number of points in the span
            npts_chord (int): number of points in the chord 

        Returns:

            Tuple containing:
                **percent_camber** (npt.NDArray): matrix shape [npts_span,npts_chord] of percent camber 
                **camber** (npt.NDArray): [nspan,1] camber of each profile
        """
        percent_distance_along_camber_for_each_profile = np.zeros((npts_span,npts_chord))
        camber_temp = np.zeros(shape=(npts_span,npts_chord,2))
        camber_lengths = list()
        for i in range(npts_span):
            camber_temp[i,:,:] = np.vstack([(
                    self.ss_pts[i,:,0]+self.ps_pts[i,:,0])/2, 
                    (self.ss_pts[i,:,1]+self.ps_pts[i,:,1])/2]).transpose()
            diff_camber = np.vstack([
                    [0,0],
                    np.vstack([np.diff(camber_temp[i,:,0]),np.diff(camber_temp[i,:,1])]).transpose()
            ])
            camber_len = np.cumsum(np.sqrt(diff_camber[:,0]**2 + diff_camber[:,1]**2))
            percent_distance_along_camber = [camber_len[i]/camber_len[-1] for i in range(len(camber_len))]
            percent_distance_along_camber_for_each_profile[i,:] = percent_distance_along_camber
            camber_lengths.append(camber_len)
            
        percent_distance = percent_distance_along_camber_for_each_profile
        camber = np.array(camber_lengths)[:,-1] # camber for each profile
        
        return percent_distance, camber
    
    def __apply_lean__(self,npts_span:int,npts_chord:int):
        """Lean is a shift in the profiles in the y-direction
            
        Args:
            npts_span (int): number of points in the spanwise direction 
            npts_chord (int): number of points in the chordwise direction 
        """
        if len(self.lean_cambers) != 0: 
            percent_camber,_ = self.__percent_camber__(npts_span,npts_chord)
            lean_y_temp = np.zeros((npts_span,len(self.leans)))  # rth
        
            # Insert zero lean at LE and TE if lean isn't specified there
            if self.lean_cambers[0] != 0:
                b = bezier([0 for _ in self.lean_cambers],[0 for _ in self.lean_cambers])
                self.leans.insert(0,b)
                self.lean_cambers.insert(0,0)
            if self.lean_cambers[-1] != 1:
                b = bezier([0 for _ in self.lean_cambers],[0 for _ in self.lean_cambers])
                self.leans.append(b)
                self.lean_cambers.append(1)
            # for each lean and location 
            i = 0 
            for lean,lean_loc in zip(self.leans,self.lean_cambers):
                lean_y_temp[:,i] = lean.get_point(np.linspace(0,1,npts_span))
                i+=1 
            
            # Apply lean 
            for i in range(npts_span):
                self.ss_pts[i,:,1] += csapi(lean_loc,lean_y_temp[i,:])(percent_camber[i,:])
                self.ps_pts[i,:,1] += csapi(lean_loc,lean_y_temp[i,:])(percent_camber[i,:])
            
    def build(self,npts_span:int=100,npts_chord:int=100):
        """Build the 3D Blade

        Args:
            npts_span (int, optional): number of points defining the span. Defaults to 100.
            npts_chord (int, optional): number of points defining the chord. Defaults to 100.
        """
        self.__apply_stacking__()
        
        # interpolate the geometry
        self.__interpolate__(npts_span,npts_chord)
        self.__apply_tip_gap__()
        
        self.__apply_lean__(npts_span,npts_chord)
        self.__stretch_profiles__(npts_span,npts_chord)
        
        # Apply Fillet radius to hub 
        if self.fillet_r>0:
            self.__apply_fillets__(npts_chord)
        
        # Scale the profiles for the passage 
        self.__scale_profiles__(npts_span,npts_chord)
        
        
    def plot(self):
        """Plots the generated design 
        """
        fig = plt.figure(num=1,dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(self.hub_pts[:,0],self.hub_pts[:,0]*0,self.hub_pts[:,2],'k')
        ax.plot3D(self.shroud_pts[:,0],self.shroud_pts[:,0]*0,self.shroud_pts[:,2],'k')
        for i in range(self.ss_pts.shape[0]):
            ax.plot3D(self.ss_pts[i,:,0],self.ss_pts[i,:,1],self.ss_pts[i,:,2],'r')
            ax.plot3D(self.ps_pts[i,:,0],self.ps_pts[i,:,1],self.ps_pts[i,:,2],'b')
        ax.view_init(azim=90, elev=45)
        ax.set_xlabel('x-axial')
        ax.set_ylabel('rth')
        ax.set_zlabel('r-radial')
        plt.show()
        