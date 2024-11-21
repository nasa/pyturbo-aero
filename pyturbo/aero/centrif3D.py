from typing import List, Tuple, Union
from .centrif2D import Centrif2D
from ..helper import bezier,convert_to_ndarray, csapi, line2D, exp_ratio
import numpy as np 
import numpy.typing as npt 
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
from pyturbo.aero.airfoil3D import StackType
import matplotlib.pyplot as plt 
from plot3d import Block
from scipy.interpolate import interp1d
from geomdl import NURBS, knotvector

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
    
    ss_hub_fillet_loc:List[float] = []
    ss_hub_fillet:List[bezier] = []
    ps_hub_fillet_loc:List[float] = []
    ps_hub_fillet:List[bezier] = []
    fillet_r:float 
    
    ss_profile_pts:npt.NDArray
    ps_profile_pts:npt.NDArray
    
    ss_pts:npt.NDArray
    ps_pts:npt.NDArray

    hub_pts:npt.NDArray
    shroud_pts:npt.NDArray
    hub_shroud_thickness:npt.NDArray # Hub to shroud stretch ratio relative to hub. 
    
    __tip_clearance:float = 0 
    
    func_xhub:PchipInterpolator
    func_rhub:PchipInterpolator
    func_xshroud:PchipInterpolator
    func_rshroud:PchipInterpolator
    
    npts_span:int = 100
    npts_chord:int = 100
    t_span:npt.NDArray          # Not used yet but might be in the future
    t_chord:npt.NDArray
    
    __rotation_angle:float = 0 
    
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
    
    
    @property
    def rotation_angle(self) -> float:
        return self.__rotation_angle
    
    def rotate(self,theta:float):
        """Rotate about x axis

        Args:
            theta (float): rotation angle in degrees
        """
        self.__rotation_angle = theta 
        
        theta = np.radians(theta)
        
        # Rotate in the y-z axis 
        mat = np.array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]])
        
        for j in range(self.ss_pts.shape[1]):
            rth_r = np.vstack([self.ss_pts[:,j,1].transpose(),self.ss_pts[:,j,2].transpose()])
            res = np.matmul(mat,rth_r)
            self.ss_pts[:,j,1] = res[0,:]
            self.ss_pts[:,j,2] = res[1,:]
            
            rth_r = np.vstack([self.ps_pts[:,j,1].transpose(),self.ps_pts[:,j,2].transpose()])
            res = np.matmul(mat,rth_r)
            self.ps_pts[:,j,1] = res[0,:]
            self.ps_pts[:,j,2] = res[1,:]

    
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
    
    def __cleanup_fillet_inputs__(self):
        """Cleans up the fillets input so that there is always a fillet defined at the leading edge and trailing edge
        """
        # if the pressure side fillet is defined at leading edge but suction side is not
        if self.ps_hub_fillet_loc[0] == 0 and self.ss_hub_fillet_loc[0]>0:
            self.ss_hub_fillet.insert(0,self.ps_hub_fillet[0])
            self.ss_hub_fillet_loc.insert(0,self.ps_hub_fillet_loc[0])
        # if the suction side is defined at leading edge but pressure side is not 
        elif self.ss_hub_fillet_loc[0] == 0 and self.ps_hub_fillet_loc[0]>0:
            self.ps_hub_fillet.insert(0,self.ss_hub_fillet[0])
            self.ps_hub_fillet_loc.insert(0,self.ss_hub_fillet_loc[0])
        
        # If the pressure side is defined at the trailing edge but suction side is not
        if self.ps_hub_fillet_loc[-1] == 1 and self.ss_hub_fillet_loc[0]!=1:
            self.ss_hub_fillet.append(self.ps_hub_fillet[-1])
            self.ss_hub_fillet_loc.append(self.ps_hub_fillet_loc[-1])
        # if the suction side fillet is defined at leading edge
        elif self.ss_hub_fillet_loc[-1] == 1 and self.ps_hub_fillet_loc[0]!=1:
            self.ps_hub_fillet.append(self.ss_hub_fillet[-1])
            self.ps_hub_fillet_loc.append(self.ss_hub_fillet_loc[-1])
        # Not all cases were taken into account. Users should define something at leading edge and trailing edge and have fillets be equal. 
        
    @staticmethod
    def __fillet_shift__(t_ss:float,height:float,fillet_r:float,fillet_array:List[bezier],fillet_array_loc:List[float]):
        """Gets the fillet shift on the suction side given a height along an airfoil
            Fillets shift must be applied perpendicular to the blade. 

        Args:
            t_ss (float): percent along the suction side 
            height (float): height of the profile
            fillet_r (float): fillet radius 
            fillet_array (List[bezier]): Array of fillets
            fillet_array_loc (List[float]): Location of fillets 
        
        Returns:
            (float): Amount to shift the point at t_ss in the normal direction
        """
        if height > fillet_r:
            return 0
        else:
            t = height/fillet_r
            shifts = np.zeros((len(fillet_array),1))
            for i in range(len(fillet_array)):            # X Values are the shift; Get the shift for each fillet 
                if fillet_array[i].get_point(0)[1]==1:    # Depends how fillets are defined 
                    temp,_ = fillet_array[i].get_point(1-t)
                else:
                    temp,_ = fillet_array[i].get_point(t)
                shifts[i] = temp[0]
            # shift = csapi(fillet_array_loc,shifts,t_ss)[0]*fillet_r
            shift = fillet_array[0].get_point(1-t)[0]*fillet_r
         
            return float(shift)
    
    @staticmethod
    def __get_normal__(pts:npt.NDArray,span_indx:int,chord_indx:int):
        """Get the outward normal for any index for a given set of points 

        Args:
            pts (npt.NDArray): Array of points NxMx3
            span_indx (int): index of the array in N axis
            chord_indx (int): index of the array in M axis

        Returns:
            3x1: Normal Vector 
        """
        max_span,max_pts,_ = pts.shape
        
        if span_indx == 0 and chord_indx>0 and chord_indx<max_pts-1: # If bottom use the one up
            span_indx+=1
            
        if span_indx == 0 and chord_indx==0: # bottom left 
            span_indx+=1
            chord_indx+=1    

        # Bottom right 
        if span_indx==0 and chord_indx==max_pts-1:
            span_indx+=1
            chord_indx-=1
        
        
        if chord_indx==0 and span_indx>0 and span_indx<max_span-1: # Left
            chord_indx+=1
        
        if chord_indx == max_pts-1 and span_indx>0 and span_indx<max_span-1: # Right
            chord_indx-=1
            
        # # Left
        # if chord_indx==0 and span_indx>0 and span_indx<max_span-1:
        #     P = 0.5*(pts[span_indx-1,chord_indx]+pts[span_indx,chord_indx])
        #     Q = 0.5*(pts[span_indx,chord_indx+1]+pts[span_indx,chord_indx])
        #     R = 0.5*(pts[span_indx+1,chord_indx]+pts[span_indx,chord_indx])
        # # Right
        # if chord_indx == max_pts-1 and span_indx>0 and span_indx<max_span-1:
        #     P = 0.5*(pts[span_indx+1,chord_indx]+pts[span_indx,chord_indx])
        #     Q = 0.5*(pts[span_indx,chord_indx-1]+pts[span_indx,chord_indx])
        #     R = 0.5*(pts[span_indx-1,chord_indx]+pts[span_indx,chord_indx])
        if span_indx>0 and span_indx<max_span and chord_indx>0 and chord_indx<max_pts:
            # Interior
            P=0.5*(pts[span_indx+1,chord_indx] + pts[span_indx,chord_indx])
            Q=0.25*(
                        pts[span_indx-1,chord_indx-1] + 
                        pts[span_indx-1,chord_indx] +
                        pts[span_indx,chord_indx] +
                        pts[span_indx,chord_indx-1]
                    )
            R=0.25*(
                        pts[span_indx-1,chord_indx+1] + 
                        pts[span_indx-1,chord_indx] +
                        pts[span_indx,chord_indx] +
                        pts[span_indx,chord_indx+1]
                    )
        n = np.cross(Q-P,R-P)
        return n/np.linalg.norm(n,ord=1) # Normal 
    
    def __apply_fillets__(self):
        """Apply fillets 
        """
        self.__cleanup_fillet_inputs__()
        
        t = self.t_chord
        ss_shifts = np.zeros((self.npts_span, self.npts_chord,3))
        ps_shifts = np.zeros((self.npts_span, self.npts_chord,3))
        for j in range(self.npts_chord):
            # Look along the span to get distance 
            dx = np.diff(self.ss_pts[:,j,0])
            dy = np.diff(self.ss_pts[:,j,1])
            dr = np.diff(self.ss_pts[:,j,2])
            dist = np.sqrt(dx**2+dy**2+dr**2)   # Distance from point to point 
            dist_cumsum = np.cumsum(dist)       # Cumulative distance from the hub 

            # find indices where where less than fillet radius
            for i in range(len(dist_cumsum <= self.fillet_r)): # looking up the span 
                if dist_cumsum[i] > self.fillet_r:
                    break                
                magnitude_of_shift = self.__fillet_shift__(t[j],dist_cumsum[i],self.fillet_r,self.ss_hub_fillet,self.ss_hub_fillet_loc)
                n = self.__get_normal__(self.ss_pts,i,j)
                ss_shifts[i,j,:] = n*magnitude_of_shift
                
                magnitude_of_shift = self.__fillet_shift__(t[j],dist_cumsum[i],self.fillet_r,self.ps_hub_fillet,self.ps_hub_fillet_loc)
                n = -self.__get_normal__(self.ps_pts,i,j)
                ps_shifts[i,j,:] = n*magnitude_of_shift    
            
        self.ss_pts+=ss_shifts
        self.ps_pts+=ps_shifts
        LE = 0.5*(self.ss_pts[:,0,:] + self.ps_pts[:,0,:])
        self.ss_pts[:,0,:] = LE
        self.ps_pts[:,0,:] = LE
        
        TE = 0.5*(self.ss_pts[:,-1,:] + self.ps_pts[:,-1,:])
        self.ss_pts[:,-1,:] = TE
        self.ps_pts[:,-1,:] = TE
        
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
        t = self.t_chord * (self.blade_position[1]-self.blade_position[0]) + self.blade_position[0]
        # t = np.linspace(self.blade_position[0],self.blade_position[1],npts_chord)
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
            Note: This only affects x and r and not r_theta
        Args:
            npts_span (int): number of points in the spanwise direction 
            npts_chord (int): number of points in the chordwise direction
        """
        percent_camber,_ = self.__percent_camber__(npts_span,npts_chord)
        hub_shroud_thickness = np.zeros((npts_chord))
        t = self.blade_position[0]+(self.blade_position[1]-self.blade_position[0])*percent_camber[0,:]
        
        for j in range(npts_chord):
            xhub = self.func_xhub(t[j])
            rhub = self.func_rhub(t[j])
            
            xshroud = self.func_xshroud(t[j])
            rshroud = self.func_rshroud(t[j])
            l = line2D([xhub,rhub],[xshroud,rshroud])
            hub_shroud_thickness[j]=l.length
            x,r = l.get_point(self.t_span)
            for i in range(npts_span):
                self.ps_pts[i,j,0]=x[i] 
                self.ps_pts[i,j,2]=r[i]
                
                self.ss_pts[i,j,0]=x[i]
                self.ss_pts[i,j,2]=r[i]
        self.hub_shroud_thickness = hub_shroud_thickness
        # Build the hub and shroud 
        self.hub_pts = np.vstack([
                self.func_xhub(np.linspace(0,1,npts_chord*2)),
                self.func_xhub(np.linspace(0,1,npts_chord*2))*0, 
                self.func_rhub(np.linspace(0,1,npts_chord*2))]).transpose()
        self.shroud_pts = np.vstack([
            self.func_xshroud(np.linspace(0,1,npts_chord*2)),
            self.func_xshroud(np.linspace(0,1,npts_chord*2))*0, 
            self.func_rshroud(np.linspace(0,1,npts_chord*2))]).transpose()

    def __flatten__(self,pts:npt.NDArray):
        """Each of the profile in ss_pts and ps_pts follow the hub and shroud.
        This code flattens it at constant radius but keeps the length 

        Returns:
            Tuple: containing
            
                *flatten_pts* (npt.NDArray): flatten points
                *dx* (npt.NDArray): matrix of dx between points 
        """
        flatten_pts = pts.copy()*0
        dx_list = np.zeros((self.npts_span,self.npts_chord))
        dr_list = np.zeros((self.npts_span,self.npts_chord))
        for i in range(self.npts_span):
            dx = np.diff(pts[i,:,0])
            dr = np.diff(pts[i,:,2])
            dh = np.hstack([[0], np.cumsum(np.sqrt(dx**2+dr**2))])
            dx = np.hstack([[0], np.cumsum(dx)])
            flatten_pts[i,:,0] = pts[i,0,0]+dh
            flatten_pts[i,:,2] = pts[i,0,2]
            dx_list[i,:] = dx
            dr_list[i,:] = dr
        return flatten_pts,dx_list,dr_list
    
    def __unflatten__(self,flatten_pts:npt.NDArray,dx_list:npt.NDArray,dr_list:npt.NDArray):
        """unflatten the geometry 

        Args:
            flatten_pts (npt.NDArray): _description_
            dx (npt.NDArray): _description_
        """
        pts = np.zeros((self.npts_span,self.npts_chord))
        for i in range(self.npts_span):
            pts[i,:,0] = flatten_pts[i,0,0]+dx_list[i,:]
            pts[i,:,2] = flatten_pts[i,0,2]+dr_list[i,:]
    
    def build_splitter(self,nose_thickness:float=0.1,
                       splitter_start:float=0.5,
                       wall_start:float=0.54):
        """Build splitter

        Args:
            nose_thickness (float, optional): nose thickness as a percent of thickness. Defaults to 0.1.
            splitter_start (float, optional): splitter starting position. Defaults to 0.5.
            wall_start (float, optional): wall start position. Defaults to 0.54.

        Returns:
            Centrif3D: splitter object 
        """
        sp_ss_pts = self.ss_pts.copy()*0
        sp_ps_pts = self.ss_pts.copy()*0
        
        ss_pts = self.ss_pts
        ps_pts = self.ps_pts
        for i in range(self.npts_span):
            ss,ps = self.__splitter_build_profile__(ss_pts=ss_pts[i,:,:],
                                            ps_pts=ps_pts[i,:,:],
                                            nose_thickness=nose_thickness,
                                            t_splitter_start=splitter_start,
                                            t_wall_start=wall_start,
                                            t_span=self.t_span[i])
            sp_ss_pts[i,:,:] = ss
            sp_ps_pts[i,:,:] = ps
        splitter = Centrif3D(self.profiles,self.stacktype)
        splitter.ss_pts = sp_ss_pts
        splitter.ps_pts = sp_ps_pts
        splitter.npts_chord = self.npts_chord
        splitter.npts_span = self.npts_span
        splitter.t_chord = self.t_chord
        splitter.t_span = self.t_span
        return splitter
            
    def __splitter_build_profile__(self,ss_pts:npt.NDArray,
                                    ps_pts:npt.NDArray,
                                    nose_thickness:float,
                                    t_splitter_start:float,
                                    t_wall_start:float,
                                    t_span:float):
        """Builds a splitter profile
        

        Args:
            t_pts (npt.NDArray): percentage along the hub for each point
            ss_pts (npt.NDArray): suction side points 
            ps_pts (npt.NDArray): pressure side points 
            thickness (float): nose thickness
            t_splitter_start (float): percentage along hub splitter starts
        """
        camber_pts = 0.5*(ss_pts + ps_pts)

        def get_thickness(p:float):
            """Get the thickness 

            Args:
                p (float): Percent between splitter start and wall start 

            Returns:
                _type_: _description_
            """
            
            ss_nose_pt = np.array([csapi(self.t_chord,ss_pts[:,0],t_splitter_start+p*(t_wall_start-t_splitter_start)),
                                    csapi(self.t_chord,ss_pts[:,1],t_splitter_start+p*(t_wall_start-t_splitter_start))])

            ps_nose_pt = np.array([csapi(self.t_chord,ps_pts[:,0],t_splitter_start+p*(t_wall_start-t_splitter_start)),
                                    csapi(self.t_chord,ps_pts[:,1],t_splitter_start+p*(t_wall_start-t_splitter_start))])
        
            camber_pt = np.array([csapi(self.t_chord,camber_pts[:,0], t_splitter_start+p*(t_wall_start-t_splitter_start)), 
                                csapi(self.t_chord,camber_pts[:,1], t_splitter_start+p*(t_wall_start-t_splitter_start))])
            ss_thck = ss_nose_pt - camber_pt # dx,dy basically 
            ps_thck = ps_nose_pt - camber_pt
            return ss_thck, ps_thck, camber_pt
        
        # Cut the blade based on a percentage of the hub  
        nose_start = np.array([csapi(self.t_chord,camber_pts[:,0], t_splitter_start), 
                                csapi(self.t_chord,camber_pts[:,1], t_splitter_start)])
        t_splitter = np.linspace(t_splitter_start,self.blade_position[1],self.npts_chord)
        ss_thck_1, ps_thck_1,c1 = get_thickness(0)
        ss_thck_2, ps_thck_2,c2 = get_thickness(0.5)
        ss_thck_3, ps_thck_3,c3 = get_thickness(0.8)
        
        t_wall = np.linspace(t_wall_start,1,self.npts_chord)
        ss_pts_int = np.array([csapi(self.t_chord,ss_pts[:,0],t_wall),
                            csapi(self.t_chord,ss_pts[:,1],t_wall)]).transpose()
        ps_pts_int = np.array([csapi(self.t_chord,ps_pts[:,0],t_wall),
                            csapi(self.t_chord,ps_pts[:,1],t_wall)]).transpose()
        
        nose_ss = np.vstack([nose_start,
                             ss_thck_1*nose_thickness+c1, # need to add nose_start[2] to this
                             ss_thck_2*0.5*(1-nose_thickness)+c2,
                             ss_thck_3*0.9*(1-nose_thickness)+c3])
        nose_ss = np.vstack([nose_ss,ss_pts_int])
        
        nose_ps = np.vstack([nose_start,
                        ps_thck_1*nose_thickness+c1, # need to add nose_start[2] to this
                        ps_thck_2*0.5*(1-nose_thickness)+c2,
                        ps_thck_3*0.9*(1-nose_thickness)+c3])
        nose_ps = np.vstack([nose_ps,ps_pts_int])
        
        # lets get the radius 
        radius = np.zeros((self.npts_chord))
        for i,t in enumerate(t_splitter):
            l = line2D([self.func_xhub(t),self.func_rhub(t)],
                    [self.func_xshroud(t),self.func_rhub(t)])
            _,r = l.get_point(t_span)
            radius[i] = r
            
        # Create the nurbs
        ss = NURBS.Curve()
        ss.degree = 3 # Cubic
        ctrlpts = np.column_stack([nose_ss, nose_ss[:,1]*0]) # Add empty column for z axis
        ss.ctrlpts = ctrlpts
        ss.knotvector = knotvector.generate(ss.degree,ctrlpts.shape[0])
        ss.delta = 1/self.npts_chord
        
        ps = NURBS.Curve()
        ps.degree = 3 # Cubic
        ctrlpts = np.column_stack([nose_ps, nose_ps[:,1]*0]) # Add empty column for z axis
        ps.ctrlpts = ctrlpts
        ps.knotvector = knotvector.generate(ps.degree,ctrlpts.shape[0])
        ps.delta = 1/self.npts_chord
        
        ss_pts = np.hstack([np.array(ss.evalpts),r])
        ps_pts = np.hstack([np.array(ps.evalpts),r])
        
        plt.plot(ss_pts[:,0],ss_pts[:,1])
        plt.show()
        return ss_pts,ps_pts
                             
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
        
        self.ss_profile_pts = ss_pts_temp
        self.ps_profile_pts = ps_pts_temp
        
        # Construct the new denser ss and ps 
        ss_pts = np.zeros((npts_span,npts_chord,3))
        ps_pts = np.zeros((npts_span,npts_chord,3))
        
        t_temp = np.linspace(0,1,len(self.profiles))
        
        for i in range(npts_chord):
            ss_pts[:,i,0] = csapi(t_temp,ss_pts_temp[:,i,0],self.t_span)
            ss_pts[:,i,1] = csapi(t_temp,ss_pts_temp[:,i,1],self.t_span)
            ss_pts[:,i,2] = csapi(t_temp,ss_pts_temp[:,i,2],self.t_span)
            
            ps_pts[:,i,0] = csapi(t_temp,ps_pts_temp[:,i,0],self.t_span)
            ps_pts[:,i,1] = csapi(t_temp,ps_pts_temp[:,i,1],self.t_span)
            ps_pts[:,i,2] = csapi(t_temp,ps_pts_temp[:,i,2],self.t_span)
        
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
                lean_y_temp[:,i] = lean.get_point(self.t_span)
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
        self.npts_span = npts_span
        self.npts_chord = npts_chord
        
        if self.fillet_r>0:
            # Lets get a better resolution of the fillet 
            a = 0.1 # Percent span where expansion ratio stops
            b = 0.1 # Percent chord when expansion ratio starts and stops 
            h1 = exp_ratio(1.2,50)*a
            h2 = np.linspace(0,1,npts_span-50)*(1-a)+a
            self.t_span = np.hstack([h1,h2[1:]])
            self.t_chord = np.linspace(0,1,npts_chord)
        else:
            self.t_span = np.linspace(0,1,npts_span)
            self.t_chord = np.linspace(0,1,npts_chord)

        
        self.__apply_stacking__()
        
        # interpolate the geometry
        self.__interpolate__(npts_span,npts_chord)
        self.__apply_tip_gap__()
        
        self.__apply_lean__(npts_span,npts_chord)
        self.__stretch_profiles__(npts_span,npts_chord)
        
        # Scale the profiles for the passage 
        self.__scale_profiles__(npts_span,npts_chord)
        
        # Apply Fillet radius to hub 
        if self.fillet_r>0:
            self.__apply_fillets__()
    
    
        
    def plot_x_slice(self,j:int):
        fig = plt.figure(num=2,dpi=150)
        ax = fig.add_subplot(111)
        ax.plot(self.ss_pts[:,j,1],self.ss_pts[:,j,2],'.r',label='suction side')
        ax.plot(self.ps_pts[:,j,1],self.ps_pts[:,j,2],'.b',label='pressure side')
        ax.set_xlabel('rth')
        ax.set_ylabel('r')
        plt.axis('equal')
        plt.show()
        
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
        
        