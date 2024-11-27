from typing import List, Tuple, Union
from .centrif2D import Centrif2D
from ..helper import bezier,convert_to_ndarray, csapi, line2D, exp_ratio,interpcurve
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
    """Generates the 3D blade starting from LE to TE. Leading edge is all lined up 
    """
    profiles:List[Centrif2D]
    stacktype:StackType 
    leans:List[bezier]
    lean_percent_spans:List[float]
    
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
    camber_pts:npt.NDArray

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
    scales:npt.NDArray
    centroids:npt.NDArray
    
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
        def rotate_via_numpy(xy, radians):
            """Use numpy to build a rotation matrix and take the dot product."""
            x, y = xy
            c, s = np.cos(radians), np.sin(radians)
            j = np.matrix([[c, s], [-s, c]])
            m = np.dot(j, [x, y])
            return m[0,:],m[1,:]
        
        self.__rotation_angle = theta 
        theta = np.radians(theta)
        
        # Rotate in the y-z axis 
        for j in range(self.ss_pts.shape[1]):            
            rth_r = np.vstack([self.ss_pts[:,j,1].transpose(),self.ss_pts[:,j,2].transpose()])
            res = rotate_via_numpy(rth_r,theta)
            self.ss_pts[:,j,1] = res[0]
            self.ss_pts[:,j,2] = res[1]
            
            rth_r = np.vstack([self.ps_pts[:,j,1].transpose(),self.ps_pts[:,j,2].transpose()])
            res = rotate_via_numpy(rth_r,theta)
            self.ps_pts[:,j,1] = res[0]
            self.ps_pts[:,j,2] = res[1]
            
            rth_r = np.vstack([self.camber_pts[:,j,1].transpose(),self.camber_pts[:,j,2].transpose()])
            res = rotate_via_numpy(rth_r,theta)
            self.camber_pts[:,j,1] = res[0]
            self.camber_pts[:,j,2] = res[1]
    
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

    def __match_aspect_ratio__(self):
        """Stretch the profiles in the x and y direction to match the height of the hub and shroud
        """
        def solve_t(t,val:float,func:PchipInterpolator):
            return np.abs(val-func(t))
        
        # Lets get the length from start to finish
        splitter_start = self.profiles[0].splitter_camber_start
        if splitter_start == 0:
            t = self.t_chord * (self.blade_position[1]-self.blade_position[0]) + self.blade_position[0]
        else:
            t = self.t_chord * (self.blade_position[1]-splitter_start) + splitter_start
        
        # Build a database of xr for the profiles in the spanwise direction 
        xh = self.func_xhub(t); xsh = self.func_xshroud(t)
        rh = self.func_rhub(t); rsh = self.func_rshroud(t)
        x_r = np.zeros((self.npts_span,self.npts_chord,2))
        
        for j in range(len(self.t_chord)):
            x,r = line2D([xh[j],rh[j]],[xsh[j],rsh[j]]).get_point(self.t_span[:,j])
            x_r[:,j,0] = x
            x_r[:,j,1] = r
        
        # Setup
        scale = np.zeros((self.npts_span,1)); centroid = np.zeros((self.npts_span,2))
        for i in range(self.npts_span):
            x_scale = x_r[i,-1,0]-x_r[i,0,0]
            xmax = max(self.ss_pts[i,:,0].max(),
                    self.ps_pts[i,:,0].max()); 
            xmin = min(self.ss_pts[i,:,0].min(),
                    self.ps_pts[i,:,0].min())
                
            rth_max = max(self.ss_pts[i,:,1].max(),
                        self.ps_pts[i,:,1].max()); 
            rth_min = min(self.ss_pts[i,:,1].min(),
                        self.ps_pts[i,:,1].min())
            dx = xmax - xmin
            centroid[i,0] = (xmax+xmin)/2
            centroid[i,1] = (rth_max+rth_min)/2
            scale[i] = x_scale/dx
            
        # Get the aspect ratio
        for i in range(self.npts_span):
            if self.stacktype == StackType.leading_edge:
                x_start = x_r[i,0,0]
                # Shift the x coordinates to start at hub starting location
                self.ss_pts[i,:,0] += x_start-self.ss_pts[i,0,0]
                self.ps_pts[i,:,0] += x_start-self.ps_pts[i,0,0]
                self.camber_pts[i,:,0] += x_start-self.camber_pts[i,0,0]
            else:
                x_end = x_r[i,-1,0]
                # Shift the x coordinates to end at hub ending location
                self.ss_pts[i,:,0] += x_end-self.ss_pts[i,-1,0]
                self.ps_pts[i,:,0] += x_end-self.ps_pts[i,-1,0]
                self.camber_pts[i,:,0] += x_end-self.camber_pts[i,-1,0]
            # Scale the blade up by moving geometry to the center. Calculate Aspect Ratio using Camber            
            self.ss_pts[i,:,0] -= centroid[i,0]
            self.ss_pts[i,:,1] -= centroid[i,1]
            
            self.ps_pts[i,:,0] -= centroid[i,0]
            self.ps_pts[i,:,1] -= centroid[i,1]
            
            self.camber_pts[i,:,0] -= centroid[i,0]
            self.camber_pts[i,:,1] -= centroid[i,1]
            
            self.ss_pts[i,:,0] *= scale[i]
            self.ss_pts[i,:,1] *= scale[i]
            
            self.ps_pts[i,:,0] *= scale[i]
            self.ps_pts[i,:,1] *= scale[i]
            
            self.camber_pts[i,:,0] *= scale[i]
            self.camber_pts[i,:,1] *= scale[i]
            
            if self.stacktype == StackType.leading_edge:
                self.ss_pts[i,:,0] += x_start-self.ss_pts[i,0,0]
                self.ss_pts[i,:,1] -= self.ss_pts[i,0,1]
            
                self.ps_pts[i,:,0] += x_start-self.ps_pts[i,0,0]
                self.ps_pts[i,:,1] -= self.ps_pts[i,0,1]
            
                self.camber_pts[i,:,0] += x_start-self.camber_pts[i,0,0]
                self.camber_pts[i,:,1] -= self.camber_pts[i,0,1]
            else:
                self.ss_pts[i,:,0] += x_end-self.ss_pts[i,-1,0]
                self.ss_pts[i,:,1] -= self.ss_pts[i,-1,1]
            
                self.ps_pts[i,:,0] += x_end-self.ps_pts[i,-1,0]
                self.ps_pts[i,:,1] -= self.ps_pts[i,-1,1]
            
                self.camber_pts[i,:,0] += x_end-self.camber_pts[i,-1,0]
                self.camber_pts[i,:,1] -= self.camber_pts[i,-1,1]
                
            # need to work on this function 
            func = PchipInterpolator(np.linspace(0,1,self.npts_chord),x_r[i,:,0])
            func_r = PchipInterpolator(np.linspace(0,1,self.npts_chord),x_r[i,:,1])
            for j in range(self.npts_chord):
                res = minimize_scalar(solve_t,bounds=[0,1],args=(self.ss_pts[i,j,0],func))
                t = res.x 
                r = func_r(t)
                self.ss_pts[i,j,2] = r
            
            for j in range(self.npts_chord):
                res = minimize_scalar(solve_t,bounds=[0,1],args=(self.ps_pts[i,j,0],func))
                t = res.x 
                r = func_r(t)
                self.ps_pts[i,j,2] = r
        
        # Shift leading edge to rth = 0 
        if self.stacktype == StackType.trailing_edge:
            LE_rth = self.ss_pts[0,0,1]
            for i in range(self.npts_span):
                self.ss_pts[i,:,1] -= LE_rth
                self.ps_pts[i,:,1] -= LE_rth
        # i = 10
        # plt.figure(clear=True)
        # # plt.plot(x_r[i,:,0],x_r[i,:,1],'k')
        # plt.plot(self.ss_pts[i,:,0],self.ss_pts[i,:,1],'r')
        # plt.plot(self.ps_pts[i,:,0],self.ps_pts[i,:,1],'b')
        # plt.axis('scaled')
        # plt.show()
        # self.plot(True)
        self.scales = scale
        self.centroids = centroid
         
    def __interpolate__(self):
        """Interpolate the geometry to make it denser
        """
        ss_pts_temp = np.zeros((len(self.profiles),self.npts_chord,3))
        ps_pts_temp = np.zeros((len(self.profiles),self.npts_chord,3))
        camber_pts_temp = np.zeros((len(self.profiles),self.npts_chord,3))
        h = np.linspace(0,1,self.npts_span)

        # Build and interpolate the blade 
        for i in range(len(self.profiles)):
            self.profiles[i].build(self.npts_chord)
            ss_pts_temp[i,:,:] = self.profiles[i].ss_pts
            ps_pts_temp[i,:,:] = self.profiles[i].ps_pts
            camber_pts_temp[i,:,:] = self.profiles[i].camber_pts
            ss_pts_temp[i,:,2] = h[i]
            ps_pts_temp[i,:,2] = h[i]
        
        self.ss_profile_pts = ss_pts_temp
        self.ps_profile_pts = ps_pts_temp
        
        # Construct the new denser ss and ps 
        ss_pts = np.zeros((self.npts_span,self.npts_chord,3))
        ps_pts = np.zeros((self.npts_span,self.npts_chord,3))
        camber_pts = np.zeros((self.npts_span,self.npts_chord,3))
        
        t_temp = np.linspace(0,1,len(self.profiles))
        
        t = (self.t_span-self.t_span[0,:])/(self.t_span[-1,:]-self.t_span[0,:])
        for j in range(self.npts_chord):
            ss_pts[:,j,0] = csapi(t_temp,ss_pts_temp[:,j,0],t[:,j])
            ss_pts[:,j,1] = csapi(t_temp,ss_pts_temp[:,j,1],t[:,j])
            ss_pts[:,j,2] = csapi(t_temp,ss_pts_temp[:,j,2],t[:,j])
            
            ps_pts[:,j,0] = csapi(t_temp,ps_pts_temp[:,j,0],t[:,j])
            ps_pts[:,j,1] = csapi(t_temp,ps_pts_temp[:,j,1],t[:,j])
            ps_pts[:,j,2] = csapi(t_temp,ps_pts_temp[:,j,2],t[:,j])
            
            camber_pts[:,j,0] = csapi(t_temp,camber_pts_temp[:,j,0],t[:,j])
            camber_pts[:,j,1] = csapi(t_temp,camber_pts_temp[:,j,1],t[:,j])
            camber_pts[:,j,2] = csapi(t_temp,camber_pts_temp[:,j,2],t[:,j])
        self.ps_pts = ps_pts
        self.ss_pts = ss_pts
        self.camber_pts = camber_pts
    
    def __tip_clearance__(self):
        """Build the tspan matrix such that tip clearance is maintained
        """
        self.t_span = np.zeros((self.npts_span,self.npts_chord))
        self.t_chord = np.linspace(0,1,self.npts_chord)
        t = self.t_chord * (self.blade_position[1]-self.blade_position[0]) + self.blade_position[0]
        
        xh = self.func_xhub(t); xsh = self.func_xshroud(t)
        rh = self.func_rhub(t); rsh = self.func_rshroud(t)
                
        for j in range(len(self.t_chord)):
            l = line2D([xh[j],rh[j]],[xsh[j],rsh[j]])
            t2 = l.get_t(l.length-self.tip_clearance)

            if self.fillet_r>0:
                # Lets get a better resolution of the fillet 
                a = 0.1 # Percent span where expansion ratio stops
                h1 = exp_ratio(1.2,50)*a
                h2 = np.linspace(0,t2,self.npts_span-50)*(t2-a)+a
                self.t_span[:,j] = np.hstack([h1,h2[1:]])
            else:
                self.t_span[:,j] = np.linspace(0,t2,self.npts_span)
        
    def __rth_shift__(self,main_blade=None):
        if main_blade is not None:
            drth = self.ss_pts[0,-1,1] - main_blade.ss_pts[0,-1,1]
            self.ss_pts[:,:,1] -= drth
            self.ps_pts[:,:,1] -= drth
            self.camber_pts[:,:,1] -= drth
        
    def build(self,npts_span:int=100,npts_chord:int=100,main_blade=None):
        """Build the 3D Blade

        Args:
            npts_span (int, optional): number of points defining the span. Defaults to 100.
            npts_chord (int, optional): number of points defining the chord. Defaults to 100.
            main_blade (Centrif3D): main_blade. Include this only if you are using a splitter.
        """
        self.npts_span = npts_span
        self.npts_chord = npts_chord
        
        # Scale to match hub and shroud curves 
        t = np.linspace(0,1,self.hub.shape[0])
        self.func_xhub = PchipInterpolator(t,self.hub[:,0])
        self.func_rhub = PchipInterpolator(t,self.hub[:,1])
        self.func_xshroud = PchipInterpolator(t,self.shroud[:,0])
        self.func_rshroud = PchipInterpolator(t,self.shroud[:,1])
        
        self.__tip_clearance__()
        self.__apply_stacking__()
        self.__interpolate__()
        # self.plot(True)
        self.__match_aspect_ratio__()
        self.__rth_shift__(main_blade)
        # self.__apply_lean__(npts_span,npts_chord)
        # Build the hub and shroud 
        self.hub_pts = np.vstack([
                self.func_xhub(np.linspace(0,1,npts_chord*2)),
                self.func_xhub(np.linspace(0,1,npts_chord*2))*0, 
                self.func_rhub(np.linspace(0,1,npts_chord*2))]).transpose()
        self.shroud_pts = np.vstack([
            self.func_xshroud(np.linspace(0,1,npts_chord*2)),
            self.func_xshroud(np.linspace(0,1,npts_chord*2))*0, 
            self.func_rshroud(np.linspace(0,1,npts_chord*2))]).transpose()
        
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
        
    def plot(self,blade_only:bool=False):
        """Plots the generated design 
        """
        fig = plt.figure(num=1,dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        if not blade_only:
            ax.plot3D(self.hub_pts[:,0],self.hub_pts[:,0]*0,self.hub_pts[:,2],'k')
            ax.plot3D(self.shroud_pts[:,0],self.shroud_pts[:,0]*0,self.shroud_pts[:,2],'k')
            
        for i in range(self.ss_pts.shape[0]):
            ax.plot3D(self.ss_pts[i,:,0],self.ss_pts[i,:,1],self.ss_pts[i,:,2],'r')
            ax.plot3D(self.ps_pts[i,:,0],self.ps_pts[i,:,1],self.ps_pts[i,:,2],'b')
        ax.view_init(azim=90, elev=45)
        ax.set_xlabel('x-axial')
        ax.set_ylabel('rth')
        ax.set_zlabel('r-radial')
        plt.axis('scaled')
        plt.show()
    
        
        