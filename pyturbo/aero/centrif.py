from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union
from pyturbo.helper import convert_to_ndarray, line2D, bezier, exp_ratio, arc, ellispe, csapi
import numpy.typing as npt 
import numpy as np 
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from geomdl import NURBS, knotvector
from scipy.optimize import minimize_scalar
from scipy.integrate import cumulative_trapezoid,cumulative_simpson
from scipy.integrate import trapezoid
import copy
from findiff import Diff, coefficients

class WaveDirection(Enum):
    x:int = 0
    r:int = 1

@dataclass 
class TrailingEdgeProperties:
    # Flat Trailing Edge 
    TE_Cut:bool = False                     # Defaults to no cut TE
    TE_cut_strength:int = 2                 # Number of control points to flatten TE
    # Circular Trailing Edge
    TE_Radius:float = 0.005                 # In theta

@dataclass 
class WavyBladeProperties:
    LE_Wave:npt.NDArray
    TE_wave:npt.NDArray
    SS_Wave:npt.NDArray
    PS_Wave:npt.NDArray
    LE_Wave_angle:Optional[List[float]]=None
    TE_Wave_angle:Optional[List[float]]=None

@dataclass
class SplitterProperties:    
    position:Tuple[float,float] = (0.5,1)
    
@dataclass
class CentrifProfile:
    """Properties used to build a 2D Centrif Profile
    """
    percent_span:float                      
    LE_Thickness:float                      # In theta
    
    LE_Metal_Angle:float
    TE_Metal_Angle:float
    
    LE_Metal_Angle_Loc:float
    TE_Metal_Angle_Loc:float
    
    ss_thickness:List[float]
    ps_thickness:List[float]
    
    wrap_angle:float                        # angle of wrap/theta
    wrap_displacements:List[float]          # percent of wrap_angle
    wrap_displacement_locs:List[float]      # percent chord
    
    trailing_edge_properties:TrailingEdgeProperties
    
@dataclass
class CentrifProfileDebug:
    """Class representing the construction of a single 2D Centrif Profile in the meridional direction. 
    contains debugging information 
    """
    SS:npt.NDArray                  # ControlPoints mp-theta
    PS:npt.NDArray  
    camber_mp_th:npt.NDArray        # Camber points mp-theta mp(0), x(1), r(2), th(3), t-camber(4), t-span(5)
    ss_mp_pts:npt.NDArray           # Generated ss points mp(0),theta(1),r(2),x(3),t_camber,t_span
    ps_mp_pts:npt.NDArray           # Generated ps points mp-theta
    camber_mp_func:PchipInterpolator

@dataclass
class CentrifBlade:
    ss_cyl_pts: npt.NDArray # span_indx, x,r,theta
    ps_cyl_pts: npt.NDArray 
    
    ss_cart_pts: npt.NDArray # span_indx, x, y, z
    ps_cart_pts: npt.NDArray 

    ss_mp_pts: npt.NDArray # span_indx, mp, theta
    ps_mp_pts: npt.NDArray 
    
    tspan:npt.NDArray   # span_indx, span_percent

class Centrif:
    hub:npt.NDArray
    shroud:npt.NDArray
    profiles:List[CentrifProfile]
    profiles_debug:List[CentrifProfileDebug]
    splitter_profiles:List[CentrifProfile]
    splitter_debug:List[CentrifProfileDebug] = None

    mainblade:CentrifBlade = None
    splitterblade:CentrifBlade = None
    fullwheel = Tuple[List[CentrifBlade],List[CentrifBlade]] # Tuple containing (mainblade) and (splitterblade)
    blade_position:Tuple[float,float] # Start and End positions
    splitter_start:List[float]
    
    func_xhub:PchipInterpolator
    func_rhub:PchipInterpolator
    func_xshroud:PchipInterpolator
    func_rshroud:PchipInterpolator
    
    camber_t_th:List[bezier]
    
    __tip_clearance_percent:float = 0
    __tip_clearance:float = 0
    
    t_hub:npt.NDArray                   # This is the t from hub to shroud 
    t_span:npt.NDArray
    t_blade:npt.NDArray
    npts_chord:int              
    npts_span:int
    
    hub_pts_cyl:npt.NDArray
    shroud_pts_cyl:npt.NDArray
    
    def __init__(self,blade_position:Tuple[float,float]=(0.1,0.9)):
        """Initializes a centrif

        Args:
            blade_position (Tuple[float,float]): start and end position of the centrif blade 
        """
        self.profiles = list()
        self.blade_position = blade_position
        self.splitter_profiles = list()
        self.splitter_start = list()
    
    def set_blade_position(self,t_start:float,t_end:float):
        """Sets the starting location of blade along the hub. 

        Args:
            t_start (float): starting percentage along the hub. 
            t_end (float): ending percentage along the hub
        """
        self.blade_position = (t_start,t_end)
    
    def add_splitter(self,splitter_profiles:List[CentrifProfile],splitter_starts:List[float]=[0.5]):
        """Call this to construct the splitter

        Args:
            splitter_profiles (List[CentrifProfile]): Splitter Profiles with thicknesses etc. 
            splitter_starts (List[float], optional): Splitter start position as a percentage of the camberline. Defaults to [0.5].
        """
        self.splitter_profiles = splitter_profiles
        self.splitter_start = splitter_starts
        
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
    
    def add_profile(self,profile:CentrifProfile):
        """Add wrap and adjustment

        Args:
            wrap_angle (float): _description_
            wrap_adjustment (List[wrapAdjustment]): 
        """
        self.profiles.append(profile)
        
    @property
    def tip_clearance(self):
        return self.__tip_clearance_percent
    
    @tip_clearance.setter
    def tip_clearance(self,val:float):
        self.__tip_clearance_percent = val
            
    def add_LE_Wave(self,wave:Union[List[float],npt.NDArray],direction:WaveDirection):
        self.le_wave = convert_to_ndarray(wave)
    
    def add_SS_Wave(self,wave:Union[List[float],npt.NDArray]):
        self.ss_wave = convert_to_ndarray[wave]
    
    def __get_camber_xr_point__(self,t_span:float,t_hub:float) -> npt.NDArray:
        # Returns the x,r point. Doesn't require vertical line test 
        shroud_pts_cyl = np.hstack([self.func_xshroud(t_hub),self.func_rshroud(t_hub)])
        hub_pts_cyl = np.hstack([self.func_xhub(t_hub),self.func_rhub(t_hub)])    
        l = line2D(hub_pts_cyl,shroud_pts_cyl)
        x,r = l.get_point(t_span)
        return np.array([x,r])
    
    def __get_rx_slice__(self,t_span:float,t_hub:npt.NDArray) -> npt.NDArray:
        """Get the x and r coordinates between hub and shroud

        Args:
            t_span (float): percent span
            t_hub (npt.NDArray): percent along the hub from 0 to 1. Defaults to None.
        
        Returns:
            npt.NDArray: Array nx2 containing xr 
        """
        # Returns xr for the camber line. Doesn't require vertical line test 
        # t_hub is the percent along the hub 
       
        shroud_pts_cyl = np.vstack([self.func_xshroud(t_hub),self.func_rshroud(t_hub)]).transpose()
        hub_pts_cyl = np.vstack([self.func_xhub(t_hub),self.func_rhub(t_hub)]).transpose()
        n = len(t_hub)
            
        xr = np.zeros((n,2))
        for j in range(n):
            l = line2D(hub_pts_cyl[j,:],shroud_pts_cyl[j,:])
            xr[j,0],xr[j,1] = l.get_point(t_span)
        return xr
    
    def get_camber_points(self,i:int,t_hub:npt.NDArray,t_camber:npt.NDArray):
        """Get the camber in cylindrical coordinates x,r,th

        Args:
            i (int): camber index
            t_hub (npt.NDArray): new t_hub distribution, t-chord a percentage along the hub. Defaults to None
            t_camber (npt.NDArray): new t_blade distribution, t-blade is percent camber of the blade. Defaults to None

        Returns:
            (npt.NDArray): x, r, theta 
            
        """
 
        xr = self.__get_rx_slice__(self.profiles[i].percent_span,t_hub)
        _,th = self.camber_t_th[i].get_point(t_camber) # using t_camber matches up with t_hub from get_camber_xr
        xrth = np.hstack([xr,th.reshape(-1,1)])
        return xrth
    
    def get_camber_length(self,i:int):
        xrth = self.get_camber_points(i,self.t_camber)
        dr = np.diff(xrth[:,1])
        dx = np.diff(xrth[:,0])
        dth = np.diff(xrth[:,2])
        return np.sum(np.sqrt(dr**2+xrth[:,1]**2 * dth**2 + dx**2))
        

    def __build_camber__(self,profiles:CentrifProfile):
        """Builds the camber lines for all of the profiles
        creates x,r,theta as functions of t. e.g. t-theta,t-x,t-r 
        """
        t = np.linspace(0,1,self.hub.shape[0])
        self.func_xhub = PchipInterpolator(t,self.hub[:,0])
        self.func_rhub = PchipInterpolator(t,self.hub[:,1])
        self.func_xshroud = PchipInterpolator(t,self.shroud[:,0])
        self.func_rshroud = PchipInterpolator(t,self.shroud[:,1])
        
        # Build camber_xr        
        self.camber_t_th = list()       # Camber t-theta control points 
        # Profiles are defined as t-theta, t-r, t-x where t is from 0 to 1
        for profile in profiles:
            xr = self.__get_rx_slice__(profile.percent_span,np.linspace(self.blade_position[0],self.blade_position[1],self.npts_chord))
            dx = np.diff(xr[:,0])
            dr = np.diff(xr[:,1])
            mp = [2/(xr[i,1]+xr[i-1,1])*np.sqrt(dr[i-1]**2 + dx[i-1]**2) for i in range(1,len(xr[:,1]))]
            mp = np.hstack([[0],np.cumsum(mp)])
            camb_len = mp[-1]

            # LE Metal Angle dth
            dth_LE = mp[-1]*profile.LE_Metal_Angle_Loc * np.tan(np.radians(profile.LE_Metal_Angle))
            dth_wrap = mp[-1] * np.tan(np.radians(profile.wrap_angle))
            dth_TE = dth_wrap - mp[-1]*(1-profile.TE_Metal_Angle_Loc)*np.tan(np.radians(profile.TE_Metal_Angle))

            # r1 = starting radius, r2 = ending radius             
            # wrap_displacement_locs: percent chord
            # wrap displacement: percent of wrap_angle
            camber_bezier_t_th = np.zeros(shape=(4+len(profile.wrap_displacements),2)) # Bezier Control points in the t,theta plane
            camber_bezier_t_th[0,:] = [0, 0]
            camber_bezier_t_th[1,:] = [profile.LE_Metal_Angle_Loc, dth_LE]
            camber_bezier_t_th[-2,:] = [profile.TE_Metal_Angle_Loc, dth_TE]
            camber_bezier_t_th[-1,:] = [1, dth_wrap]
            x = np.hstack([camber_bezier_t_th[0:2,0], camber_bezier_t_th[-2:,0]])
            y = np.hstack([camber_bezier_t_th[0:2,1], camber_bezier_t_th[-2:,1]])
            camber_bezier = bezier(x, y)
            
            if np.any(np.abs(profile.wrap_displacements)>0): # If there are displacements factor it in
                # # Distance formula in cylindrical coordinates https://math.stackexchange.com/questions/3612484/how-do-you-calculate-distance-between-two-cylindrical-coordinates
                # camb_len = np.sqrt(xr0[1]**2+xr1[1]**2 -2*xr0[1]*xr1[1]*np.cos(np.radians(profile.wrap_angle)) + (xr1[0]-xr0[0])**2)
                j = 2
                dl = profile.TE_Metal_Angle_Loc - profile.LE_Metal_Angle_Loc
                for loc,displacement in zip(profile.wrap_displacement_locs, profile.wrap_displacements):
                    l = profile.LE_Metal_Angle_Loc + loc*dl
                    nx,ny = camber_bezier.get_point_dt(l)
                    x1,y1 = camber_bezier.get_point(l)
                    x2 = -ny*displacement*camb_len + x1
                    y2 = nx*displacement*camb_len + y1
                    camber_bezier_t_th[j,0] = x2
                    camber_bezier_t_th[j,1] = y2
                    j+=1
                self.camber_t_th.append(bezier(camber_bezier_t_th[:,0],camber_bezier_t_th[:,1]))    # Camber line is constructed as a bezier curve
            else: # if there are no displacements then use a simple bezier curve
                self.camber_t_th.append(camber_bezier)
    
    def __build_hub_shroud__(self,hub_rotation_resolution:int=20,hub_axial_npts:int=100):
        """Construct the hub and shroud

        Args:
            hub_rotation_resolution (int, optional): Resolution in number of hub rotations. Defaults to 20.
        """
        self.hub_pts_cyl = np.zeros(shape=(hub_rotation_resolution,hub_axial_npts,3))       # x,r,th
        self.shroud_pts_cyl = np.zeros(shape=(hub_rotation_resolution,hub_axial_npts,3))

        rotations = np.linspace(0,1,hub_rotation_resolution)*360
        
        self.hub_pts = np.zeros((hub_rotation_resolution,hub_axial_npts,3))
        self.shroud_pts = np.zeros((hub_rotation_resolution,hub_axial_npts,3))
        
        xhub = self.func_xhub(np.linspace(0,1,hub_axial_npts))
        rhub = self.func_rhub(np.linspace(0,1,hub_axial_npts))
        xshroud = self.func_xshroud(np.linspace(0,1,hub_axial_npts))
        rshroud = self.func_rshroud(np.linspace(0,1,hub_axial_npts))
        for i in range(len(rotations)):
            theta = np.radians(rotations[i])
            
            self.hub_pts_cyl[i,:,0] = xhub
            self.hub_pts_cyl[i,:,1] = rhub
            self.hub_pts_cyl[i,:,2] = theta

            self.shroud_pts_cyl[i,:,0] = xshroud
            self.shroud_pts_cyl[i,:,1] = rshroud
            self.shroud_pts_cyl[i,:,2] = theta

            self.hub_pts[i,:,0] = xhub
            self.hub_pts[i,:,1] = rhub*np.sin(theta)    # y
            self.hub_pts[i,:,2] = rhub*np.cos(theta) # z

            self.shroud_pts[i,:,0] = xshroud
            self.shroud_pts[i,:,1] = rshroud*np.sin(theta)     # y
            self.shroud_pts[i,:,2] = rshroud*np.cos(theta)  # z 
  
    def __create_fullwheel__(self,nblades:int,nsplitters:int=0):
        """Create fullwheel by copying the blades 

        Args:
            nblades (int): number of blades 
            nsplitters (int, optional): number of splitters. Defaults to 0.
        """
        def rotate(blade:CentrifBlade, theta:float):  # blade passed by ref
            for i in range(self.npts_span):
                blade.ps_cyl_pts[i,:,2] += theta
                blade.ss_cyl_pts[i,:,2] += theta
                
                blade.ps_mp_pts[i,:,1] += theta
                blade.ss_mp_pts[i,:,1] += theta
                r = blade.ss_cyl_pts[i,:,1]
                blade.ss_cart_pts[i,:,1]= r*np.sin(blade.ss_cyl_pts[i,:,2])  # y
                blade.ss_cart_pts[i,:,2]= r*np.cos(blade.ss_cyl_pts[i,:,2])  # z
                
                r = blade.ps_cyl_pts[i,:,1]
                blade.ps_cart_pts[i,:,1]= r*np.sin(blade.ps_cyl_pts[i,:,2])  # y
                blade.ps_cart_pts[i,:,2]= r*np.cos(blade.ps_cyl_pts[i,:,2])  # z
            return blade 
        
        # Lets check
        theta_blade = 360/nblades
        mainblades = [copy.deepcopy(self.mainblade) for _ in range(nblades)]
        splitters = []
        # Lets rotate the blades 
        theta = 0
        for b in mainblades:
            b = rotate(b, np.radians(theta))
            theta += theta_blade

        theta = theta_blade/2
        if nsplitters>0 and (self.splitterblade is not None):
            while theta<=360:
                splitters.append(copy.deepcopy(self.splitterblade))
                splitters[-1] = rotate(splitters[-1],np.radians(theta))
                theta += theta_blade

        self.fullwheel = (mainblades,splitters)

        # self.__apply_pattern__() # Applies the pattern and rotates the blade 

    @staticmethod
    def __NURBS_interpolate__(points,npts):
        curve = NURBS.Curve(); # knots = # control points + order of curve
        curve.degree = 3 # cubic
        ctrlpts = points
        # ctrlpts = np.column_stack([ctrlpts, ctrlpts[:,1]*0]) # Add empty column for z axis
        curve.ctrlpts = ctrlpts
        curve.delta = 1/npts
        curve.knotvector = knotvector.generate(curve.degree,ctrlpts.shape[0])
        return np.array(curve.evalpts)
  
    
    def __apply_thickness__(self,profiles:List[CentrifProfile],camber_starts:List[float]=[],npts_chord:int=100) -> Tuple[List[CentrifProfileDebug],npt.NDArray,npt.NDArray]:
        """Apply thickness to the cambers. Thickness cannot be applied in the t-x,t-theta,t-r domain. We must first flatten the geometry to be a function of mprime (distance along both r and x)/r. Thickness is then applied to theta but as a function of mprime. 

        Args:
            profiles (List[CentrifProfile]): Centrif Profiles or splitter profiles 
            camber_start (List[float], optional): Starting percentage along camberline. Defaults to [].
            npts_chord (int, optional): number of points along the chord . Defaults to 100.

        Returns:
            Tuple containing:
            
                **ss_mp_pts** (npt.NDArray)
                **ps_mp_pts** (npt.NDArray)
                **camb_mp_func** (List[PchipInterpolator]): List of spline fitted camber in meridional plane
                **profiles_debug** (CentrifProfileDebug): Data for plotting and debugging
        """
        ss_mp_pts = np.zeros(shape=(len(profiles),npts_chord,6)) # mp(0),theta(1),r(2),x(3),t_camber,t_span
        ps_mp_pts = np.zeros(shape=(len(profiles),npts_chord,6))
        camber_mp_th = np.zeros(shape=(len(profiles),npts_chord,6)) # mp, x, r, th, t-camber, t-span

        profiles_debug = list()
        camb_mp_func = list()
        
        def solve_t(t,val:float,func:PchipInterpolator):
            return np.abs(val-func(t))
        
        if len(camber_starts) == 0:
            [camber_starts.append(0) for _ in range(len(profiles))]
        
        # Apply thickness in theta direction 
        for i,p in enumerate(zip(profiles,camber_starts)):
            profile = p[0]; camber_start = p[1]
            te_radius = profile.trailing_edge_properties.TE_Radius
            # mp-full from slice start to end. Slice is between hub and shroud. 
            xr_full = self.__get_rx_slice__(profile.percent_span,np.linspace(0,1,self.npts_chord*2))
            dx = np.diff(xr_full[:,0])
            dr = np.diff(xr_full[:,1])

            dist_mp_full = [2/(xr_full[i,1]+xr_full[i-1,1])*np.sqrt(dr[i-1]**2 + dx[i-1]**2) for i in range(1,len(xr_full[:,1]))]
            dist_mp_full = np.hstack([[0],np.cumsum(dist_mp_full)])
            func_mp_full = PchipInterpolator(np.linspace(0,1,self.npts_chord*2),dist_mp_full)
            
            # Flatten in xr only for camber line
            xrth = self.get_camber_points(i,self.t_hub,self.t_camber)
            dx = np.diff(xrth[:,0])
            dr = np.diff(xrth[:,1])
            
            mp = [2/(xrth[i,1]+xrth[i-1,1])*np.sqrt(dr[i-1]**2 + dx[i-1]**2) for i in range(1,len(xrth[:,1]))]
            mp = np.hstack([[0],np.cumsum(mp)])

            dist_mp = mp + func_mp_full(self.t_hub[0])    # Include the offset
            func_mp = PchipInterpolator(self.t_camber,dist_mp)
            camb_mp_func.append(func_mp)
            dfunc_mp = func_mp.derivative()
            
            # Apply Thickness in theta
            npoint_ss = len(profile.ss_thickness)+4 # LE Start, LE Thickness, SS_Thickness, TE_Radius, TE_end
            npoint_ps = len(profile.ps_thickness)+4 # LE Start, LE Thickness, PS_Thickness, TE_Radius, TE_End
            if profile.trailing_edge_properties.TE_Cut:
                # Add extra point to make straight
                TE_cut_strength = profile.trailing_edge_properties.TE_cut_strength
                SS = np.zeros((npoint_ss+TE_cut_strength,3))
                PS = np.zeros((npoint_ps+TE_cut_strength,3))  # mp, theta 
            else:
                SS = np.zeros((npoint_ss,3)); PS = np.zeros((npoint_ps,3))  # mp, theta 

            camber = self.camber_t_th[i]
            _,th_start = camber.get_point(camber_start)                      # Add LE Start
            SS[0,0] = func_mp(camber_start)
            SS[0,1] = th_start
            PS[0,0] = func_mp(camber_start)
            PS[0,1] = th_start

            # Add LE Thickness
            mp_start = func_mp(camber_start)
            _,dth_dt = camber.get_point_dt(camber_start)
            dmp_dt = dfunc_mp(camber_start)
            angle = np.degrees(np.arctan2(dth_dt,dmp_dt))
            PS[1,0] = mp_start + profile.LE_Thickness * np.cos(np.radians(angle+90))   # mp
            PS[1,1] = th_start + profile.LE_Thickness * np.sin(np.radians(angle+90))   # theta
            
            SS[1,0] = mp_start + profile.LE_Thickness * np.cos(np.radians(angle-90))   # mp
            SS[1,1] = th_start + profile.LE_Thickness * np.sin(np.radians(angle-90))   # theta
            if profile.trailing_edge_properties.TE_Cut:
                SS[:,2] = np.hstack([[camber_start],np.linspace(camber_start,1,len(profile.ss_thickness)+2),np.ones((TE_cut_strength+1,))])
                PS[:,2] = np.hstack([[camber_start],np.linspace(camber_start,1,len(profile.ps_thickness)+2),np.ones((TE_cut_strength+1,))])
            else:
                SS[:,2] = np.hstack([[camber_start],np.linspace(camber_start,1,len(profile.ss_thickness)+2),[1]])
                PS[:,2] = np.hstack([[camber_start],np.linspace(camber_start,1,len(profile.ps_thickness)+2),[1]])
                            
            # Pressure side thicknesses
            profile.ps_thickness.append(te_radius)
            for j in range(2,PS.shape[0]-1):  
                _,th_start = camber(PS[j,2])
                mp_start = func_mp(PS[j,2])
                _,dth_dt = camber.get_point_dt(PS[j,2])
                dmp_dt = dfunc_mp(PS[j,2])
                angle = np.degrees(np.arctan2(dth_dt,dmp_dt))
                if profile.trailing_edge_properties.TE_Cut and j == PS.shape[0]-2-TE_cut_strength:
                    PS[j:j+TE_cut_strength+1,0] = mp_start
                    PS[j:j+TE_cut_strength+1,1] = np.linspace(th_start + profile.ps_thickness[j-2],th_start,TE_cut_strength+2).flatten()[:-1]
                    break
                else:
                    PS[j,0] = mp_start + profile.ps_thickness[j-2] * np.cos(np.radians(angle+90))
                    PS[j,1] = th_start + profile.ps_thickness[j-2] * np.sin(np.radians(angle+90))
            
            profile.ss_thickness.append(te_radius)
            for j in range(2,SS.shape[0]-1):  
                _,th_start = camber(SS[j,2])
                mp_start = func_mp(SS[j,2])
                _,dth_dt = camber.get_point_dt(SS[j,2])
                dmp_dt = dfunc_mp(SS[j,2])
                angle = np.degrees(np.arctan2(dth_dt,dmp_dt))
                if profile.trailing_edge_properties.TE_Cut and j == SS.shape[0]-2-TE_cut_strength:
                    SS[j:j+TE_cut_strength+1,0] = mp_start
                    SS[j:j+TE_cut_strength+1,1] = np.linspace(th_start - profile.ss_thickness[j-2],th_start,TE_cut_strength+2).flatten()[:-1]
                    break
                else:
                    SS[j,0] = mp_start + profile.ss_thickness[j-2] * np.cos(np.radians(angle-90))
                    SS[j,1] = th_start + profile.ss_thickness[j-2] * np.sin(np.radians(angle-90))
            
            _, th_end = camber.get_point(1)                             # Add LE Start
            mp_end = func_mp(1)
            PS[-1,0] = mp_end; PS[-1,1] = th_end
            SS[-1,0] = mp_end; SS[-1,1] = th_end

            ps_nurbs_ctrl_pts = PS[:,:2] # mp,theta
            ss_nurbs_ctrl_pts = SS[:,:2] # mp,theta

            ps_mp_pts[i,:,:2] = self.__NURBS_interpolate__(ps_nurbs_ctrl_pts,npts_chord) # mp,theta    
            ss_mp_pts[i,:,:2] = self.__NURBS_interpolate__(ss_nurbs_ctrl_pts,npts_chord)

            m = 1/(self.t_hub.max()-self.t_hub.min())
            # Inversely solve for t_camber for each mp value 
            for j in range(npts_chord):
                mp = ss_mp_pts[i,j,0]
                res = minimize_scalar(solve_t,bounds=[0,1],args=(mp,func_mp_full))
                ss_mp_pts[i,j,4] = res.x # new t_camber for mp value 
                thub = 1/m * res.x + self.t_hub.min()
                xrth = self.__get_rx_slice__(profile.percent_span,[ss_mp_pts[i,j,4]])[0]
                ss_mp_pts[i,j,2] = xrth[1] # r
                ss_mp_pts[i,j,3] = xrth[0] # x

                mp = ps_mp_pts[i,j,0]
                res = minimize_scalar(solve_t,bounds=[0,1],args=(mp,func_mp_full))
                ps_mp_pts[i,j,4] = res.x
                thub = 1/m * res.x + self.t_hub.min()   # Converting t-camber to thub
                xrth = self.__get_rx_slice__(profile.percent_span,[ps_mp_pts[i,j,4]])[0]
                ps_mp_pts[i,j,2] = xrth[1] # r
                ps_mp_pts[i,j,3] = xrth[0] # x

            ss_mp_pts[i,:,5] = self.t_span[i,:] # tspan 
            ps_mp_pts[i,:,5] = self.t_span[i,:]    

            xrth = self.get_camber_points(i,self.t_hub,self.t_camber)
            camber_mp_th = np.vstack([func_mp(self.t_camber),xrth[:,0],xrth[:,1],xrth[:,2],self.t_camber,self.t_span[i,:]]).transpose() # mp, x, r, th, t-camber, t-span
            
            profiles_debug.append(
                CentrifProfileDebug(SS=SS,
                                    PS=PS,
                                    camber_mp_th=camber_mp_th,
                                    ss_mp_pts=ss_mp_pts[i,:,:],
                                    ps_mp_pts=ps_mp_pts[i,:,:],
                                    camber_mp_func=camb_mp_func))
            
        return profiles_debug

    def __interpolate__(self,profiles:List[CentrifProfileDebug]) -> CentrifBlade:    
        """Interpolate the profiles over the full span 
        
        Args:
            profiles (List[CentrifProfileDebug]): List of profiles either for blade or splitter

        Returns:
            (CentrifBlade) containing: 

                **ss_cyl_pts** (npt.NDArray): Suction side cylindrical coordinates (x,r,th)
                **ps_cyl_pts** (npt.NDArray): Pressure side cylindrical coordinates (x,r,th)
                **ss_cart_pts** (npt.NDArray): Suction side cartesian coordinates (x,r,th)
                **ps_cart_pts** (npt.NDArray): Pressure side coordinates (x,r,th)
                **ss_mp_pts** (npt.NDArray): Suction side meridional coordinates (x,r,th)
                **ps_mp_pts** (npt.NDArray): Pressure side meridional coordinates (x,r,th)
        """
        npts_chord, ndata = profiles[0].ss_mp_pts.shape
        temp_ss_mp_pts = np.zeros((len(profiles),npts_chord,ndata))
        temp_ps_mp_pts = np.zeros((len(profiles),npts_chord,ndata))
        ss_cyl_pts = np.zeros((self.npts_span,npts_chord,3))    # x,r,th
        ps_cyl_pts = np.zeros((self.npts_span,npts_chord,3))    
        ss_cart_pts = np.zeros((self.npts_span,npts_chord,3))    # x,y,z
        ps_cart_pts = np.zeros((self.npts_span,npts_chord,3))    
        ss_mp_pts = np.zeros((self.npts_span,npts_chord,2))     # mp,theta
        ps_mp_pts = np.zeros((self.npts_span,npts_chord,2))     
        
        # Lets get all the data as a big matrix
        for i in range(len(profiles)):
            for j in range(ndata):  # mp(0),theta(1),r(2),x(3),t_camber,t_span
                temp_ss_mp_pts[i,:,j] = profiles[i].ss_mp_pts[:,j] 
                temp_ps_mp_pts[i,:,j] = profiles[i].ps_mp_pts[:,j] 

        for j in range(self.npts_chord):
            tspan_profiles = np.linspace(0, self.t_span[-1,j], len(profiles))
            ss_cyl_pts[:,j,0] = csapi(tspan_profiles,temp_ss_mp_pts[:,j,3],self.t_span[:,j])   # x
            ss_cyl_pts[:,j,1] = csapi(tspan_profiles,temp_ss_mp_pts[:,j,2],self.t_span[:,j])   # r
            ss_cyl_pts[:,j,2] = csapi(tspan_profiles,temp_ss_mp_pts[:,j,1],self.t_span[:,j])   # th
            
            ps_cyl_pts[:,j,0] = csapi(tspan_profiles,temp_ps_mp_pts[:,j,3],self.t_span[:,j])   # x
            ps_cyl_pts[:,j,1] = csapi(tspan_profiles,temp_ps_mp_pts[:,j,2],self.t_span[:,j])   # r
            ps_cyl_pts[:,j,2] = csapi(tspan_profiles,temp_ps_mp_pts[:,j,1],self.t_span[:,j])   # th

            ss_mp_pts[:,j,0] = csapi(tspan_profiles,temp_ss_mp_pts[:,j,0],self.t_span[:,j])   # mp
            ss_mp_pts[:,j,1] = csapi(tspan_profiles,temp_ss_mp_pts[:,j,1],self.t_span[:,j])   # theta

        for i in range(self.npts_span):
            theta = ss_cyl_pts[i,:,2]
            r = ss_cyl_pts[i,:,1]

            ss_cart_pts[i,:,0]=ss_cyl_pts[i,:,0]  # x
            ss_cart_pts[i,:,1]=r*np.sin(theta)  # y
            ss_cart_pts[i,:,2]=r*np.cos(theta)  # z
            
            theta =ps_cyl_pts[i,:,2]
            r = ps_cyl_pts[i,:,1]

            ps_cart_pts[i,:,0]=ps_cyl_pts[i,:,0]  # x            
            ps_cart_pts[i,:,1]=r*np.sin(theta)  # y
            ps_cart_pts[i,:,2]=r*np.cos(theta)  # z

        return CentrifBlade(ss_cyl_pts,ps_cyl_pts,ss_cart_pts,ps_cart_pts,ss_mp_pts,ps_mp_pts,tspan=self.t_span)
    
    def __tip_clearance__(self):
        """Build the tspan matrix such that tip clearance is maintained
        """
        self.t_span = np.zeros((self.npts_span,self.npts_chord))
        
        xh = self.func_xhub(self.t_hub); xsh = self.func_xshroud(self.t_hub)
        rh = self.func_rhub(self.t_hub); rsh = self.func_rshroud(self.t_hub)
        
        for j in range(len(self.t_hub)):
            cut = line2D([xh[j],rh[j]],[xsh[j],rsh[j]])
            t2 = cut.get_t(cut.length-self.tip_clearance)
            self.t_span[:,j] = np.linspace(0,t2,self.npts_span) # exp_ratio(1.2,self.npts_span,maxvalue=t2)
     

    def build(self,npts_span:int=100, npts_chord:int=100,nblades:int=3,nsplitters:int=1):
        """Build the centrif blade 

        Args:
            npts_span (int, optional): number of points in spanwise direction. Defaults to 100.
            npts_chord (int, optional): number of points in chord. Defaults to 100.
            nblades (int, optional): number of blades. Defaults to 3
            nsplitters (int, optional): splitters in between blades. Defaults to 1
            
        """
        self.npts_chord = npts_chord; self.npts_span = npts_span
        self.t_span = np.linspace(0,1,npts_span)       
        self.t_hub = np.linspace(self.blade_position[0],self.blade_position[1],npts_chord)
        
        self.t_camber = (self.t_hub - self.t_hub.min())/(self.t_hub.max()-self.t_hub.min()) # non-dimensional t for camber line.
        self.__build_camber__(self.profiles)
        self.__build_hub_shroud__()
        self.__tip_clearance__()
        
        self.profiles_debug = self.__apply_thickness__(self.profiles,camber_starts=[],npts_chord=npts_chord)  # Creates the flattened profiles
        self.mainblade = self.__interpolate__(self.profiles_debug)
        if self.splitter_start != 0 and len(self.splitter_profiles)>0:
            self.splitter_debug = self.__apply_thickness__(self.splitter_profiles,camber_starts=self.splitter_start,npts_chord=npts_chord)  # Creates the flattened profiles
            self.splitterblade = self.__interpolate__(self.splitter_debug)
        
        self.__create_fullwheel__(nblades,nsplitters)
        
    def plot_camber(self,plot_hub_shroud:bool=True):
        """Plot the camber line
        """
        
        t = self.t_camber
        for i,b in enumerate(self.camber_t_th):
            fig = plt.figure(num=1,dpi=150,clear=True)
            [x,y] = b.get_point(t)        
            plt.plot(x, y,'-b',label=f"curve {i}")
            plt.plot(b.x, b.y,'or',label=f"curve {i}")
            plt.xlabel('percent camber')
            plt.ylabel('theta')
            plt.title(f'Profile {i}')
            plt.savefig(f'profile camber {i:2d}')

        fig = plt.figure(num=2,dpi=150,clear=True)
        ax = fig.add_subplot(111, projection='3d')
        # Plots the camber and control points 
        k = 0
        for camber,profile in zip(self.camber_t_th,self.profiles):
            xc = np.zeros(shape=(len(camber.x),1))      # Bezier control points x,r,th
            rc = np.zeros(shape=(len(camber.x),1))
            thc = np.zeros(shape=(len(camber.x),1))
            for i in range(len(camber.x)):              # Camber x is really t 
                xc[i],rc[i] = self.__get_camber_xr_point__(profile.percent_span,camber.x[i])
                _,thc[i] = camber.get_point(camber.x[i])
            
            ax.plot3D(xc[1:-1],thc[1:-1],rc[1:-1],'or',markersize=4)
            ax.plot3D(xc[0],thc[0],rc[0],'ok',markersize=4)
            ax.plot3D(xc[-1],thc[-1],rc[-1],'ok',markersize=4)
            
            xrth = self.get_camber_points(k,self.t_hub,self.t_camber)
            ax.plot3D(xrth[:,0],xrth[:,2],xrth[:,1],'-b',linewidth=2)   # x, theta, r
            k+=1
        # Plots the hub and shroud 
        if plot_hub_shroud:
            ax.plot3D(self.hub_pts_cyl[0,:,0],self.hub_pts_cyl[0,:,2],self.hub_pts_cyl[0,:,1],'k',linewidth=2.5)
            ax.plot3D(self.shroud_pts_cyl[0,:,0],self.shroud_pts_cyl[0,:,2],self.shroud_pts_cyl[0,:,1],'k',linewidth=2.5)
        ax.view_init(68,-174)
        plt.axis('equal')
        # plt.show()
        
    def plot_mp_profile(self):
        """Plot the control profiles in the rx-theta plane
        """
        def plot_data(profiles_debug:CentrifProfileDebug,prefix:str):
            for i in range(len(profiles_debug)):
                p = profiles_debug[i]
                plt.figure(num=i,clear=True)    # mp view
                plt.plot(p.camber_mp_th[:,0],p.camber_mp_th[:,3], color='black', linestyle='dashed',linewidth=2,label='camber')
                plt.plot(p.SS[:,0],p.SS[:,1],'ro', label='suction') 
                plt.plot(p.PS[:,0],p.PS[:,1],'bo',label='pressure')
                plt.plot(p.ss_mp_pts[:,0],p.ss_mp_pts[:,1],'r-',label='ss')
                plt.plot(p.ps_mp_pts[:,0],p.ps_mp_pts[:,1],'b-',label='ps')
                plt.legend()
                plt.xlabel('mprime')
                plt.ylabel('theta')
                plt.title(f'mprime profile-{i}')
                plt.axis('equal')
                # plt.show()
                plt.savefig(f'{prefix} mp-theta {i:02d}.png',dpi=150)
                
                plt.figure(num=i,clear=True)    # x-theta view
                xrth = self.get_camber_points(i,self.t_hub,self.t_camber)
                plt.plot(xrth[:,0],xrth[:,2], color='black', linestyle='dashed',linewidth=2,label='camber')
                plt.plot(p.ss_mp_pts[:,3],p.ss_mp_pts[:,1],'r-',label='ss')
                plt.plot(p.ps_mp_pts[:,3],p.ps_mp_pts[:,1],'b-',label='ps')
                plt.legend()
                plt.xlabel('X')
                plt.ylabel('Theta')
                plt.title(f'X-Theta Profile-{i}')
                plt.axis('equal')
                plt.savefig(f'{prefix} x-theta {i:02d}.png',dpi=150)
                
                plt.figure(num=i,clear=True)    # theta-r view
                xrth = self.get_camber_points(i,self.t_hub,self.t_camber)
                plt.plot(np.degrees(xrth[:,2]),xrth[:,1], color='black', linestyle='dashed',linewidth=2,label='camber')
                plt.plot(np.degrees(p.ss_mp_pts[:,1]),p.ss_mp_pts[:,2],'r-',label='ss')
                plt.plot(np.degrees(p.ps_mp_pts[:,1]),p.ps_mp_pts[:,2],'b-',label='ps')
                plt.legend()
                plt.xlabel('Theta')
                plt.ylabel('R')
                plt.axis('equal')
                plt.title(f'Theta-r Profile-{i}')
                plt.savefig(f'{prefix} theta-r {i:02d}.png',dpi=150)
                
                plt.figure(num=i+50,clear=True)    # x-r view
                xrth = self.get_camber_points(i,self.t_hub,self.t_camber)
                plt.plot(xrth[:,0],xrth[:,1], color='black', linestyle='dashed',linewidth=2,label='camber')
                plt.plot(p.ss_mp_pts[:,3],p.ss_mp_pts[:,2],'r-',label='ss')
                plt.plot(p.ps_mp_pts[:,3],p.ps_mp_pts[:,2],'b-',label='ps')
                plt.plot(self.hub_pts_cyl[0,:,0],self.hub_pts_cyl[0,:,1],'k',label='hub',alpha=0.2)
                plt.plot(self.shroud_pts_cyl[0,:,0],self.shroud_pts_cyl[0,:,1],'k',label='shroud',alpha=0.2)
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('r')
                plt.title(f'x-r Profile-{i}')
                plt.axis('equal')
                plt.savefig(f'{prefix} x-r {i:02d}.png',dpi=150)
        plot_data(self.profiles_debug,'profile')
        if self.splitter_debug is not None:
            plot_data(self.splitter_debug,'splitter')
    
    def plot(self):
        """3D Cartesian Plot
        """
        p = self.mainblade
        plt.figure(num=100,clear=True)    # x-r view
        for i in range(p.ps_cyl_pts.shape[0]):
            plt.plot(p.ss_cyl_pts[i,:,0],p.ss_cyl_pts[i,:,1],'r-',label='ss')
            plt.plot(p.ss_cyl_pts[i,:,0],p.ss_cyl_pts[i,:,1],'b-',label='ps')
        plt.plot(self.hub_pts_cyl[0,:,0],self.hub_pts_cyl[0,:,1],'k',label='hub',alpha=0.2)
        plt.plot(self.shroud_pts_cyl[0,:,0],self.shroud_pts_cyl[0,:,1],'k',label='shroud',alpha=0.2)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('r')
        plt.title(f'Meridional Plot x-r')
        plt.axis('equal')
        
        # Cartesian
        fig = plt.figure(num=101,clear=True,dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot3D(self.hub_pts_cyl[:,0],self.hub_pts_cyl[:,0]*0,self.hub_pts_cyl[:,2],'k')
        # ax.plot3D(self.shroud_pts_cyl[:,0],self.shroud_pts_cyl[:,0]*0,self.shroud_pts_cyl[:,2],'k')
        for i in range(p.ss_cart_pts.shape[0]):
            ax.plot3D(p.ss_cart_pts[i,:,0],p.ss_cart_pts[i,:,1],p.ss_cart_pts[i,:,2],'r',label='suction') # x,y,z
            ax.plot3D(p.ps_cart_pts[i,:,0],p.ps_cart_pts[i,:,1],p.ps_cart_pts[i,:,2],'b',label='pressure')
        ax.set_xlabel('x-axial')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('3D Plot - Cartesian')
        ax.view_init(azim=90, elev=45)
        plt.axis('equal')
        plt.show()
              
    def plot_front_view(self):
        p = self.mainblade
        plt.figure(num=1,clear=True) # Front view theta-r
        for i in range(self.npts_span):
            plt.plot(p.ss_cyl_pts[i,:,1],p.ss_cyl_pts[i,:,2])         # mp,theta,r,x,t_camber,t_span
            plt.plot(p.ps_cyl_pts[i,:,1],p.ps_cyl_pts[i,:,2])
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title(f'Front view Theta-r')
        plt.axis('equal')
        plt.savefig(f'front view.png',dpi=150)

    def plot_fullwheel(self):
        blades = self.fullwheel[0]
        splitters = self.fullwheel[1]
        plt.close('all')
        fig = plt.figure(num=1,clear=True)
        ax = fig.add_subplot(111, projection='3d')
        for k,b in enumerate(blades):   # Plot the Blades 
            for i in range(self.npts_span):
                ax.plot3D(b.ss_cart_pts[i,:,0],b.ss_cart_pts[i,:,1],b.ss_cart_pts[i,:,2],'r',label='suction') # x,y,z
                ax.plot3D(b.ps_cart_pts[i,:,0],b.ps_cart_pts[i,:,1],b.ps_cart_pts[i,:,2],'b',label='pressure') # x,y,z

        for k,s in enumerate(splitters): # Plot the splitters 
            for i in range(self.npts_span):
                ax.plot3D(s.ss_cart_pts[i,:,0],s.ss_cart_pts[i,:,1],s.ss_cart_pts[i,:,2],'g',label='splitter_suction') # x,y,z
                ax.plot3D(s.ps_cart_pts[i,:,0],s.ps_cart_pts[i,:,1],s.ps_cart_pts[i,:,2],'m',label='splitter_pressure') # x,y,z
        
        for i in range(self.hub_pts.shape[0]):
            ax.plot3D(self.hub_pts[i,:,0],self.hub_pts[i,:,1],self.hub_pts[i,:,2],'k',label='hub',alpha=0.1)
            ax.plot3D(self.shroud_pts[i,:,0],self.shroud_pts[i,:,1],self.shroud_pts[i,:,2],'k',label='shroud',alpha=0.1)
                
        for j in range(self.hub_pts.shape[1]):
            ax.plot3D(self.hub_pts[:,j,0],self.hub_pts[:,j,1],self.hub_pts[:,j,2],'k',label='hub',alpha=0.1)
            ax.plot3D(self.shroud_pts[:,j,0],self.shroud_pts[:,j,1],self.shroud_pts[:,j,2],'k',label='shroud',alpha=0.1)

        ax.set_xlabel('x-axial')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title(f'Full Wheel')
        plt.axis('equal')
        plt.show()
     
    def pitch_to_chord(self) -> Tuple[npt.NDArray,npt.NDArray]:
        """Computes the pitch to chord ratio at the inlet and exit
        

        Returns:
            Tuple[npt.NDArray,npt.NDArray]: containing
            
                **s_c_b2b** (npt.ndarray): pitch to chord blade to blade from (hub to shroud)
                **s_c_b2s** (npt.ndarray): pitch to chord blade to splitter from (hub to shroud)
        """
        
        assert len(self.fullwheel)>0,"Blades must be generated"
        mainblades = self.fullwheel[0]
        splitters = self.fullwheel[1]
        # Inlet Blade to Blade
        assert len(mainblades)>0, "main blades must be generated"
        x1 = mainblades[0].ps_cyl_pts[:,:,0] 
        r1 = mainblades[0].ps_cyl_pts[:,:,1] 
        th1 = mainblades[0].ps_cyl_pts[:,:,2] 
        
        x2 = mainblades[1].ps_cyl_pts[:,:,0] 
        r2 = mainblades[1].ps_cyl_pts[:,:,1] 
        th2 = mainblades[1].ps_cyl_pts[:,:,2] 
        
        d_b2b = r1**2+r2**2-2*r1*r2*np.cos(th1-th2)+(x1-x2)**2 # distance blade to blade 
        
        # Blade chord or mprime*r
        s_c_b2b = np.zeros(shape=(mainblades[0].ps_cyl_pts.shape[0],2)) # Pitch to chord blade to blade
        s_c_b2s = np.zeros(shape=(mainblades[0].ps_cyl_pts.shape[0],2)) # Pitch to chord blade to splitter 

        for i in range(mainblades[0].ps_cyl_pts.shape[0]):
            dx = np.diff(x1[i,:])
            dr = np.diff(r1[i,:])
            dth = np.diff(th1[i,:])
            c = 0 
            for j in range(1,len(dx)):
                c += np.sqrt(dr[j-1]**2+dr[j]**2 -2*dr[j-1]*dr[j]*np.cos(dth[j-1]-dth[j]) +  (dx[j-1]-dx[j])**2)
            s_c_b2b[i,0] = d_b2b[i,0]/c
            s_c_b2b[i,-1] = d_b2b[i,-1]/c
            
        # Inlet Blade to Splitter
        if len(splitters)>0: 
            x3 = splitters[0].ps_cyl_pts[:,:,0]
            r3 = splitters[0].ps_cyl_pts[:,:,0]
            th3 = splitters[0].ps_cyl_pts[:,:,0]
            
            d_b2s = r1**2+r3**2-2*r1*r3*np.cos(th1-th3)+(x1-x3)**2 # distance blade to stator
        
            for i in range(mainblades[0].ps_cyl_pts.shape[0]):
                dx = np.diff(x1[i,:])
                dr = np.diff(r1[i,:])
                dth = np.diff(th1[i,:])
                c = 0 
                for j in range(1,len(dx)):
                    c += dr[j-1]**2+dr[j]**2 -2*dr[j-1]*dr[j]*np.cos(dth[j-1]-dth[j]) +  (dx[j-1]-dx[j])**2
                s_c_b2s[i,0] = d_b2s[i,0]/c
                s_c_b2s[i,-1] = d_b2s[i,-1]/c
                
        return s_c_b2b,s_c_b2s
