from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union
from pyturbo.helper import convert_to_ndarray, line2D, bezier, csapi, ray2D, ray2D_intersection, arc, ellispe, xr_to_mprime
import numpy.typing as npt 
import numpy as np 
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from geomdl import NURBS, knotvector
from scipy.optimize import minimize_scalar
import copy
import scipy.interpolate as spi
from itertools import combinations, combinations_with_replacement

class WaveDirection(Enum):
    x:int = 0
    r:int = 1

@dataclass 
class TrailingEdgeProperties:
    # Flat Trailing Edge 
    TE_Cut:bool = False                     # Defaults to no cut TE
    TE_cut_strength:int = 2                 # Number of control points to flatten TE
    # Circular Trailing Edge
    TE_Radius:float = 0.005                 # In theta radians
    npts:int=30                             # number of trailing edge points 

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
class PatternPairCentrif:
    chord_scaling:float
    rotation_ajustment:float
    
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
    thickness_start:float=0.01
    thickness_end:float=0.95
    use_bezier_thickness:bool = False       # False: use nurbs. True: use bezier curve
    
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
    ss_arc:bezier
    ps_arc:bezier
    
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
    hub_arc_len:float 
    
    camber_mp_th:List[bezier]
    le_theta_shifts:List[float] = []
    te_theta_shifts:List[float] = []
    
    __tip_clearance_percent:float = 0
    
    t_hub:npt.NDArray                   # This is the t from hub to shroud 
    t_span:npt.NDArray
    t_blade:npt.NDArray
    npts_chord:int              
    npts_span:int
    
    hub_pts_cyl:npt.NDArray
    shroud_pts_cyl:npt.NDArray
    use_mid_wrap_angle:bool
    use_ray_camber:bool
    
    patterns:List[PatternPairCentrif] = []

    def __init__(self,blade_position:Tuple[float,float]=(0.1,0.9),
                 use_mid_wrap_angle:bool=True,
                 use_ray_camber:bool=False):
        """Initializes a centrif

        Args:
            blade_position (Tuple[float,float]): start and end position of the centrif blade 
            use_mid_wrap_angle (bool): If true, wrap angle from mid is applied to rest of profiles. Defaults to True
            use_ray_camber (bool): use ray intersection to construct the bezier curve defining camber line 
        """
        self.profiles = list()
        self.blade_position = blade_position
        self.splitter_profiles = list()
        self.splitter_start = list()
        
        self.use_ray_camber = use_ray_camber
        self.use_mid_wrap_angle = use_mid_wrap_angle
        self.camber_mp_th = list()
        self.patterns.append(PatternPairCentrif(chord_scaling=1,rotation_ajustment=0)) # adds a default pattern, this is no modification

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
     
    def add_le_theta_shifts(self,theta_shifts:List[float]):
        """Add leading edge theta shifts

        Args:
            theta_shifts (List[float]): how much to shift the leading edge in theta direction
        """
        self.le_theta_shifts = theta_shifts
    
    def add_te_theta_shifts(self,theta_shifts:List[float]):
        """Add trailing edge theta shifts

        Args:
            theta_shifts (List[float]): how much to shift the trailing edge in theta direction
        """
        self.te_theta_shifts = theta_shifts

    def add_pattern_pair(self,pair:PatternPairCentrif):
        """Patterns are repeated for n number of blades but the goal is to repeat without patterns in a row. 

        Args:
            pair (PatternPairCentrif): Create a pattern pair. 
        """
        self.patterns.append(pair)
        
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
        t_hub = convert_to_ndarray(t_hub)
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
            i (int): profile index
            t_hub (npt.NDArray): new t_hub distribution, t-chord a percentage along the hub. Defaults to None
            t_camber (npt.NDArray): new t_blade distribution, t-blade is percent camber of the blade. Defaults to None

        Returns:
            (npt.NDArray): x, r, theta 
            
        """
 
        xr = self.__get_rx_slice__(self.profiles[i].percent_span,t_hub)
        _,th = self.camber_mp_th[i].get_point(t_camber) # using t_camber matches up with t_hub from get_camber_xr
        xrth = np.hstack([xr,th.reshape(-1,1)])
        return xrth
    
    def get_camber_length(self,i:int):
        xrth = self.get_camber_points(i,self.t_camber)
        dr = np.diff(xrth[:,1])
        dx = np.diff(xrth[:,0])
        dth = np.diff(xrth[:,2])
        return np.sum(np.sqrt(dr**2+xrth[:,1]**2 * dth**2 + dx**2))
 
    def __build_camber__(self,profile:CentrifProfile,theta_wrap:float=None,use_ray_intersection:bool=False) -> Tuple[bezier,float]:
        """Builds the camber for a profile

        Args:
            profile (CentrifProfile): Centrif Profile Definition
            theta_wrap (float, optional): Manually specify the ending theta wrap. It can be useful if you wanted to keep the wrap angle consistent. Defaults to None.
            use_ray_intersection (bool, optional): Use ray intersection to specify camber. Defaults to False.

        Returns:
            Tuple[bezier,float] containing:
            
                *camber* (bezier): bezier curve containing the camberline
                *theta_wrap* (float): wrap angle
        """
        profile_loc = profile.percent_span
        xr = self.__get_rx_slice__(profile_loc,np.linspace(self.blade_position[0],self.blade_position[1],self.npts_chord))
        mp = xr_to_mprime(xr)[0]
        camb_len = mp[-1]
        n_wrap_displacements = np.count_nonzero(profile.wrap_displacement_locs)
        camber_bezier_mp_th = np.zeros(shape=(4+n_wrap_displacements,2)) # Bezier Control points in the mp,theta plane

        # LE Metal Angle dth
        dth_LE = mp[-1]*profile.LE_Metal_Angle_Loc * np.tan(np.radians(profile.LE_Metal_Angle))
        dth_wrap = mp[-1] * np.tan(np.radians(profile.wrap_angle))
        if theta_wrap:
            dth_wrap = theta_wrap
        dth_TE = dth_wrap - mp[-1]*(1-profile.TE_Metal_Angle_Loc)*np.tan(np.radians(profile.TE_Metal_Angle))
        
        # r1 = starting radius, r2 = ending radius             
        # wrap_displacement_locs: percent chord
        # wrap displacement: percent of wrap_angle
        if use_ray_intersection:
            camber_bezier_mp_th = np.zeros(shape=(3,2)) # Bezier Control points in the t,theta plane
            camber_bezier_mp_th[0,:] = [0, 0]
            r1 = ray2D(0,0,profile.LE_Metal_Angle_Loc,dth_LE)
            r2 = ray2D(camb_len,dth_wrap,-(1-profile.TE_Metal_Angle_Loc),-dth_TE)
            intersect_x,intersect_y,_,_ = ray2D_intersection(r1,r2)
            camber_bezier_mp_th[1,:] = [intersect_x,intersect_y]
            camber_bezier_mp_th[-1,:] = [camb_len, dth_wrap]
            return bezier(camber_bezier_mp_th[:,0], camber_bezier_mp_th[:,1]),dth_wrap
        else:
            camber_bezier_mp_th = np.zeros(shape=(4+len(profile.wrap_displacements),2)) # Bezier Control points in the t,theta plane
            camber_bezier_mp_th[0,:] = [0, 0]
            camber_bezier_mp_th[1,:] = [profile.LE_Metal_Angle_Loc*camb_len, dth_LE]
            camber_bezier_mp_th[-2,:] = [profile.TE_Metal_Angle_Loc*camb_len, dth_TE]
            camber_bezier_mp_th[-1,:] = [camb_len, dth_wrap]
            x = np.hstack([camber_bezier_mp_th[0:2,0], camber_bezier_mp_th[-2:,0]])
            y = np.hstack([camber_bezier_mp_th[0:2,1], camber_bezier_mp_th[-2:,1]])
            camber_bezier = bezier(x, y)

        if n_wrap_displacements>0: 
            # If there are displacements factor it in
            # # Distance formula in cylindrical coordinates https://math.stackexchange.com/questions/3612484/how-do-you-calculate-distance-between-two-cylindrical-coordinates
            # camb_len = np.sqrt(xr0[1]**2+xr1[1]**2 -2*xr0[1]*xr1[1]*np.cos(np.radians(profile.wrap_angle)) + (xr1[0]-xr0[0])**2)
            j = 2
            dl = profile.TE_Metal_Angle_Loc - profile.LE_Metal_Angle_Loc
            for loc,displacement in zip(profile.wrap_displacement_locs, profile.wrap_displacements):
                l = profile.LE_Metal_Angle_Loc + loc*dl
                nx,ny = camber_bezier.get_point_dt(l)
                x1,y1 = camber_bezier.get_point(l)
                x2 = x1
                y2 = -displacement*dth_wrap
                camber_bezier_mp_th[j,0] = x2
                camber_bezier_mp_th[j,1] = y2
                j+=1
        camber_bezier = bezier(camber_bezier_mp_th[:,0],camber_bezier_mp_th[:,1])
        return camber_bezier,dth_wrap

    def apply_camber_shifts(self,le_theta_shifts:List[float]=[],te_theta_shifts:List[float]=[]):
        """Shifts the camber in the theta direction at the leading and trailing edges. 

        Args:
            le_theta_shifts (List[float], optional): _description_. Defaults to [].
            te_theta_shifts (List[float], optional): _description_. Defaults to [].
        """
        n_profiles = len(self.profiles)
        # Shift the leading edge 
        if len(le_theta_shifts)==0:
            le_th_shifts = [0 for _ in range(n_profiles)]
        else:
            le_th_shifts = spi.PchipInterpolator(np.linspace(0,1,len(le_theta_shifts)),le_theta_shifts)(np.linspace(0,1,n_profiles))            
            for i in range(len(le_th_shifts)):
                x = self.camber_mp_th[i].x
                y = self.camber_mp_th[i].y
                y[0] += le_th_shifts[i]
                y[1] += le_th_shifts[i] # Done to maintain flow angle
                self.camber_mp_th[i] = bezier(x,y)
        
        if len(te_theta_shifts)==0:
            te_th_shifts = [0 for _ in range(n_profiles)]
        else:
            te_th_shifts = spi.PchipInterpolator(np.linspace(0,1,len(te_theta_shifts)),te_theta_shifts)(np.linspace(0,1,n_profiles))
            for i in range(len(te_th_shifts)):
                x = self.camber_mp_th[i].x
                y = self.camber_mp_th[i].y
                y[-1] += te_th_shifts[i]
                y[-2] += te_th_shifts[i]
                self.camber_mp_th[i] = bezier(x,y)
    
    
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
    
    
    
    def __apply_thickness__(self,profile:CentrifProfile,profile_indx:int,camber_start:float=0,npts_chord:int=100) -> CentrifProfileDebug:
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
        ss_mp_pts = np.zeros(shape=(npts_chord,6)) # mp(0),theta(1),r(2),x(3),t_camber,t_span
        ps_mp_pts = np.zeros(shape=(npts_chord,6))
        camber_mp_th = np.zeros(shape=(npts_chord,6)) # mp, x, r, th, t-camber, t-span

        camb_mp_func = list()
        
        def solve_t(t,val:float,func:PchipInterpolator):
            return np.abs(val-func(t))
        
        te_props = profile.trailing_edge_properties
        te_radius = te_props.TE_Radius

        # mp-full from slice start to end. Slice is between hub and shroud. 
        xr_full = self.__get_rx_slice__(profile.percent_span,np.linspace(0,1,self.npts_chord*4))
        dist_mp_full = xr_to_mprime(xr_full)[0]
        func_mp_full = PchipInterpolator(np.linspace(0,1,self.npts_chord*4),dist_mp_full)
        
        xr = self.__get_rx_slice__(profile.percent_span,np.linspace(0,self.blade_position[0],self.npts_chord))
        mp = xr_to_mprime(xr)[0]
        mp_offset = mp[-1]
        
        # Apply Thickness in theta
        npoint_ss = len(profile.ss_thickness)+3 # LE Start, LE Thickness, SS_Thickness, TE_Radius
        npoint_ps = len(profile.ps_thickness)+3 # LE Start, LE Thickness, PS_Thickness, TE_Radius
        if te_props.TE_Cut:
            # Add extra point to make straight
            TE_cut_strength = te_props.TE_cut_strength
            SS = np.zeros((npoint_ss+TE_cut_strength,3))
            PS = np.zeros((npoint_ps+TE_cut_strength,3))  # mp, theta 
        else:
            SS = np.zeros((npoint_ss,3)); PS = np.zeros((npoint_ps,3))  # mp, theta , t 

        camber = self.camber_mp_th[profile_indx]
        mp_start,th_start = camber.get_point(camber_start)                      # Add LE Start
        SS[0,0] = mp_start
        SS[0,1] = th_start
        PS[0,0] = mp_start
        PS[0,1] = th_start

        # Add LE Thickness
        dmp_dt,dth_dt = camber.get_point_dt(camber_start)
        mp,th = camber.get_point(np.linspace(0,1,1000))
        mp_len = mp[-1]
        
        profile.LE_Thickness *= mp_len
        profile.ss_thickness = [th*mp_len for th in profile.ss_thickness]  # Scale it as a percentage of the max mprime
        profile.ps_thickness = [th*mp_len for th in profile.ps_thickness]
        profile.trailing_edge_properties.TE_Radius *= mp_len
        
        angle = np.degrees(np.arctan2(dth_dt,dmp_dt))
        PS[1,0] = mp_start + np.round(profile.LE_Thickness * np.cos(np.radians(angle-90)),decimals=4)   # mp
        PS[1,1] = th_start + np.round(profile.LE_Thickness * np.sin(np.radians(angle-90)),decimals=4)   # theta
        
        SS[1,0] = mp_start + np.round(profile.LE_Thickness * np.cos(np.radians(angle+90)),decimals=4)   # mp
        SS[1,1] = th_start + np.round(profile.LE_Thickness * np.sin(np.radians(angle+90)),decimals=4)  # theta
        
        tss = np.hstack([[camber_start], np.linspace(profile.thickness_start,profile.thickness_end,len(profile.ss_thickness)+1)])
        # tss = camber_start + exp_ratio(1.3,len(profile.ss_thickness)+2,1-camber_start)
        tps = np.hstack([[camber_start], np.linspace(profile.thickness_start,profile.thickness_end,len(profile.ps_thickness)+1)])
        # tps = camber_start + exp_ratio(1.3,len(profile.ps_thickness)+2,1-camber_start)

        if te_props.TE_Cut:
            SS[:,2] = np.hstack([[camber_start],tss,np.ones((TE_cut_strength+1,))])
            PS[:,2] = np.hstack([[camber_start],tps,np.ones((TE_cut_strength+1,))])
        else:   
            SS[:,2] = np.hstack([[camber_start],tss])
            PS[:,2] = np.hstack([[camber_start],tps])
        
        for _ in range(2):
            # Pressure side thicknesses
            for j in range(2,PS.shape[0]):  
                mp_start,th_start = camber(PS[j,2])
                dmp_dt,dth_dt = camber.get_point_dt(PS[j,2])
                # dmp_dt = dfunc_mp(PS[j,2])
                angle = np.degrees(np.arctan2(dth_dt,dmp_dt))
                if te_props.TE_Cut and j == PS.shape[0]-2-TE_cut_strength:
                    PS[j:j+TE_cut_strength+1,0] = mp_start
                    PS[j:j+TE_cut_strength+1,1] = np.linspace(th_start - profile.ps_thickness[j-2],th_start,TE_cut_strength+2).flatten()[:-1]
                    break
                else:
                    PS[j,0] = mp_start + profile.ps_thickness[j-3] * np.cos(np.radians(angle-90))
                    PS[j,1] = th_start + profile.ps_thickness[j-3] * np.sin(np.radians(angle-90))
            
            for j in range(2,SS.shape[0]):  
                mp_start,th_start = camber(SS[j,2])
                dmp_dt,dth_dt = camber.get_point_dt(SS[j,2])
                # dmp_dt = dfunc_mp(SS[j,2])
                angle = np.degrees(np.arctan2(dth_dt,dmp_dt))
                if te_props.TE_Cut and j == SS.shape[0]-2-TE_cut_strength:
                    SS[j:j+TE_cut_strength+1,0] = mp_start
                    SS[j:j+TE_cut_strength+1,1] = np.linspace(th_start + profile.ss_thickness[j-2],th_start,TE_cut_strength+2).flatten()[:-1]
                    break
                else:
                    SS[j,0] = mp_start + profile.ss_thickness[j-3] * np.cos(np.radians(angle+90))
                    SS[j,1] = th_start + profile.ss_thickness[j-3] * np.sin(np.radians(angle+90))
        
        # Add TE
        npts_te = 30
        
        if profile.use_bezier_thickness:            
            ps_bezier = bezier(PS[:,0],PS[:,1]); ss_bezier = bezier(SS[:,0],SS[:,1])
            ps_pts = np.vstack(ps_bezier.get_point(np.linspace(0,1,npts_chord-npts_te+1))).transpose()
            ss_pts = np.vstack(ss_bezier.get_point(np.linspace(0,1,npts_chord-npts_te+1))).transpose()
            ps_mp_pts[:(npts_chord-npts_te),:2] = ps_pts[:-1,:]
            ss_mp_pts[:(npts_chord-npts_te),:2] = ss_pts[:-1,:]
            ss_arc_pts, ps_arc_pts, ss_arc, ps_arc = centrif_create_te(SS=ss_pts,PS=ps_pts,camber=camber,radius=te_radius,n_te_pts=te_props.npts)

            ps_mp_pts[(npts_chord-npts_te):,:2] = ps_arc_pts
            ss_mp_pts[(npts_chord-npts_te):,:2] = ss_arc_pts
        else:
            ps_pts = self.__NURBS_interpolate__(PS[:,:2],npts_chord-npts_te+1)
            ss_pts = self.__NURBS_interpolate__(SS[:,:2],npts_chord-npts_te+1)
            ps_mp_pts[:(npts_chord-npts_te),:2] = ps_pts[:-1,:]
            ss_mp_pts[:(npts_chord-npts_te),:2] = ss_pts[:-1,:]
            ss_arc_pts, ps_arc_pts, ss_arc, ps_arc = centrif_create_te(SS=ss_pts,PS=ps_pts,camber=camber,radius=te_radius,n_te_pts=te_props.npts)

            ps_mp_pts[(npts_chord-npts_te):,:2] = ps_arc_pts
            ss_mp_pts[(npts_chord-npts_te):,:2] = ss_arc_pts
            
        m = 1/(self.t_hub.max()-self.t_hub.min())
        # Inversely solve for t_camber for each mp value
        for j in range(npts_chord):
            mp = ss_mp_pts[j,0] + mp_offset
            res = minimize_scalar(solve_t,bounds=[0,1],args=(mp,func_mp_full),tol=1e-12)
            ss_mp_pts[j,4] = res.x # new t_camber for mp value 
            xrth = self.__get_rx_slice__(profile.percent_span,[res.x])[0]
            ss_mp_pts[j,2] = xrth[1] # r
            ss_mp_pts[j,3] = xrth[0] # x

            mp = ps_mp_pts[j,0] + mp_offset
            res = minimize_scalar(solve_t,bounds=[0,1],args=(mp,func_mp_full),tol=1e-12)
            ps_mp_pts[j,4] = res.x
            xrth = self.__get_rx_slice__(profile.percent_span,[res.x])[0]
            ps_mp_pts[j,2] = xrth[1] # r
            ps_mp_pts[j,3] = xrth[0] # x
        # # At leading edge 
        # mean_vals = 0.5*(ss_mp_pts[0,1:4] + ps_mp_pts[0,1:4]) # th,r,x
        # ss_mp_pts[0,1:4] = mean_vals; ps_mp_pts[0,1:4] = mean_vals
    
        # # At trailing edge 
        # mean_vals = 0.5*(ss_mp_pts[-1,1:4] + ps_mp_pts[-1,1:4]) # th,r,x
        # ss_mp_pts[-1,1:4] = mean_vals; ps_mp_pts[-1,1:4] = mean_vals
        
        ss_mp_pts[:,5] = profile.percent_span # percent span location of each profile
        ps_mp_pts[:,5] = profile.percent_span  

        xrth = self.get_camber_points(profile_indx,self.t_hub,self.t_camber)
        camber_mp_th = np.vstack([camber.get_point(self.t_camber)[0],xrth[:,0],xrth[:,1],xrth[:,2],self.t_camber]).transpose() # mp, x, r, th, t-camber, t-span
        
        return CentrifProfileDebug(SS=SS,
                                PS=PS,
                                camber_mp_th=camber_mp_th,
                                ss_mp_pts=ss_mp_pts[:,:],
                                ps_mp_pts=ps_mp_pts[:,:],
                                camber_mp_func=camb_mp_func,
                                ss_arc=ss_arc,
                                ps_arc=ps_arc)
    
    def __build_hub_shroud__(self,hub_rotation_resolution:int=20,hub_axial_npts:int=100):
        """Construct the hub and shroud

        Args:
            hub_rotation_resolution (int, optional): Resolution in number of hub rotations. Defaults to 20.
        """
        hub_arc_len = xr_to_mprime(self.hub)[1]
        self.hub_arc_len = hub_arc_len[-1]
        
        self.func_xhub = PchipInterpolator(hub_arc_len/hub_arc_len[-1],self.hub[:,0])
        self.func_rhub = PchipInterpolator(hub_arc_len/hub_arc_len[-1],self.hub[:,1])
        self.func_xshroud = PchipInterpolator(hub_arc_len/hub_arc_len[-1],self.shroud[:,0])
        self.func_rshroud = PchipInterpolator(hub_arc_len/hub_arc_len[-1],self.shroud[:,1])
        
        self.hub_pts_cyl = np.zeros(shape=(hub_rotation_resolution,hub_axial_npts,3))       # x,r,th
        self.shroud_pts_cyl = np.zeros(shape=(hub_rotation_resolution,hub_axial_npts,3))

        rotations = np.linspace(0,1,hub_rotation_resolution)*360
        
        self.hub_pts = np.zeros((hub_rotation_resolution,hub_axial_npts,3))
        self.shroud_pts = np.zeros((hub_rotation_resolution,hub_axial_npts,3))
        
        thub = np.linspace(0,1,hub_axial_npts)
        xhub = self.func_xhub(thub)
        rhub = self.func_rhub(thub)
        xshroud = self.func_xshroud(thub)
        rshroud = self.func_rshroud(thub)
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
            self.hub_pts[i,:,2] = rhub*np.cos(theta)    # z

            self.shroud_pts[i,:,0] = xshroud
            self.shroud_pts[i,:,1] = rshroud*np.sin(theta)      # y
            self.shroud_pts[i,:,2] = rshroud*np.cos(theta)      # z 
  
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
            ps_mp_pts[:,j,0] = csapi(tspan_profiles,temp_ps_mp_pts[:,j,0],self.t_span[:,j])   # mp
            ps_mp_pts[:,j,1] = csapi(tspan_profiles,temp_ps_mp_pts[:,j,1],self.t_span[:,j])   # 

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
        
        self.__build_hub_shroud__()
        self.__tip_clearance__()
        self.profiles_debug = list() 
        self.splitter_debug = list()
        
        if self.use_mid_wrap_angle:
            _,theta_wrap = self.__build_camber__(self.profiles[1],theta_wrap=None,use_ray_intersection=self.use_ray_camber) # use wrap angle from mid profile
        else:
            theta_wrap = None
            
        for i,profile in enumerate(self.profiles):
            self.camber_mp_th.append(self.__build_camber__(profile,theta_wrap,use_ray_intersection=self.use_ray_camber)[0])
            self.apply_camber_shifts(self.le_theta_shifts,self.te_theta_shifts)
            self.profiles_debug.append(self.__apply_thickness__(profile,i,camber_start=0,npts_chord=npts_chord))  # Creates the flattened profiles
        if self.splitter_start != 0 and len(self.splitter_profiles)>0:
            for s,splitter_profile in enumerate(self.splitter_profiles):
                self.splitter_debug.append(self.__apply_thickness__(splitter_profile,profile_indx=s,camber_start=self.splitter_start,npts_chord=npts_chord))  # Creates the flattened profiles
        self.mainblade = self.__interpolate__(self.profiles_debug)
        if self.splitter_start != 0 and len(self.splitter_profiles)>0:
            self.splitterblade = self.__interpolate__(self.splitter_debug)

        self.__create_fullwheel__(nblades,nsplitters)
    
    def __apply_pattern__(self,nblades:int):
        """Apply patterns modifications to the design. The intent of this is to reduce the peaks associated with a compressor by creating small variations in the geometry. 
        
        Args:
            nblades (int): number of blades
            blades (List[Centrif3D]): List of blades 
        """
        total_combinations = []

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
            
    def plot_camber(self,plot_hub_shroud:bool=True):
        """Plot the camber line
        """
        
        t = self.t_camber
        for i,b in enumerate(self.camber_mp_th):
            fig = plt.figure(num=1,dpi=150,clear=True)
            [x,y] = b.get_point(t)        
            plt.plot(x, y,'-b',label=f"curve {i}")
            plt.plot(b.x, b.y,'or',label=f"curve {i}")
            plt.xlabel('mprime')
            plt.ylabel('theta')
            plt.title(f'Profile {i}')
            plt.axis('equal')
            plt.savefig(f'profile camber {i:2d}')

        fig = plt.figure(num=2,dpi=150,clear=True)
        ax = fig.add_subplot(111, projection='3d')
        # Plots the camber and control points 
        k = 0
        for camber,profile in zip(self.camber_mp_th,self.profiles):
            xc = np.zeros(shape=(len(camber.x),1))      # Bezier control points x,r,th
            rc = np.zeros(shape=(len(camber.x),1))
            thc = np.zeros(shape=(len(camber.x),1))     # theta control points
            mpc = np.zeros(shape=(len(camber.x),1))     # mprime control points
            
            for i in range(len(camber.x)):              # Camber x is really mprime
                t = camber.x[i]/camber.x[-1]            # Percent along mprime
                xc[i],rc[i] = self.__get_camber_xr_point__(profile.percent_span,t)
                mpc[i],thc[i] = camber.get_point(t)
            
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
    
    def plot_profile_debug_3D(self,nblades:int=1,total_blades:int=1):
        fig = plt.figure(num=2,clear=True,dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        dtheta = np.radians(360/total_blades)

        for _ in range(nblades):
            theta = 0 
            for p in self.profiles_debug:
                th = p.ss_mp_pts[:,1] + theta
                r = p.ss_mp_pts[:,2]
                x = p.ss_mp_pts[:,3]; rth = r*th
                ax.plot3D(x,rth,r,'r',label='suction') # x,rth,r
                th = p.ps_mp_pts[:,1] + theta
                r = p.ps_mp_pts[:,2]
                x = p.ps_mp_pts[:,3]; rth = r*th
                ax.plot3D(x,rth,r,'b',label='pressure') # x,rth,r
            theta += dtheta
        ax.set_xlabel('x-axial')
        ax.set_ylabel('rth')
        ax.set_zlabel('r')
        plt.title('3D Plot - Cartesian')
        ax.view_init(azim=90, elev=45)
        plt.axis('equal')
        plt.show()
        
    def plot_mp_profile(self,nblades:int=1,total_blades:int=1):
        """Plot the control profiles in the rx-theta plane
        
        Args:
            nblades (int, optional): number of blades. Defaults to 1.
            total_blades (int, optional): total blades, used to calculate dtheta. Defaults to 1.
        """
        def plot_data(profiles_debug:CentrifProfileDebug,prefix:str):
            dtheta = np.radians(360/total_blades)
            for i in range(len(profiles_debug)):
                theta = 0; p = profiles_debug[i]
                plt.figure(num=i*4,clear=True)          # mp view
                for _ in range(nblades):
                    plt.plot(p.camber_mp_th[:,0],p.camber_mp_th[:,3]+theta, color='black', linestyle='dashed',linewidth=2,label='camber')
                    plt.plot(p.SS[:,0],p.SS[:,1],'ro', label='suction') 
                    plt.plot(p.PS[:,0],p.PS[:,1],'bo',label='pressure')
                    plt.plot(p.ss_mp_pts[:,0],p.ss_mp_pts[:,1]+theta,'r-',label='ss')
                    plt.plot(p.ps_mp_pts[:,0],p.ps_mp_pts[:,1]+theta,'b-',label='ps')
                    plt.plot(p.ss_arc.x,p.ss_arc.y+theta,'ro',markerfacecolor='none',label='suction')
                    plt.plot(p.ps_arc.x,p.ps_arc.y+theta,'bo',markerfacecolor='none',label='pressure')
                    theta += dtheta
                plt.legend()
                plt.xlabel('mprime')
                plt.ylabel('theta')
                plt.title(f'mprime profile-{i}')
                plt.axis('equal')
                plt.savefig(f'{prefix} mp-theta {i:02d}.png',dpi=150)
                
            # for i in range(len(profiles_debug)):
            #     theta = 0; p = profiles_debug[i]
            #     plt.figure(num=i*4+1,clear=True)        # x-theta view
            #     for n in range(nblades):
            #         xrth = self.get_camber_points(i,self.t_hub,self.t_camber)
            #         plt.plot(xrth[:,0],xrth[:,2]+theta, color='black', linestyle='dashed',linewidth=2,label='camber')
            #         plt.plot(p.ss_mp_pts[:,3],p.ss_mp_pts[:,1]+theta,'r-',label='ss')
            #         plt.plot(p.ps_mp_pts[:,3],p.ps_mp_pts[:,1]+theta,'b-',label='ps')
            #         theta += dtheta
            #     plt.legend()
            #     plt.xlabel('X')
            #     plt.ylabel('Theta')
            #     plt.title(f'X-Theta Profile-{i}')
            #     plt.axis('equal')
            #     plt.savefig(f'{prefix} x-theta {i:02d}.png',dpi=150)
                
            # for i in range(len(profiles_debug)):
            #     theta = 0; p = profiles_debug[i]
            #     plt.figure(num=i*4+2,clear=True)        # theta-r view
            #     for n in range(nblades):
            #         xrth = self.get_camber_points(i,self.t_hub,self.t_camber)
            #         plt.plot(np.degrees(xrth[:,2]),xrth[:,1]+theta, color='black', linestyle='dashed',linewidth=2,label='camber')
            #         plt.plot(np.degrees(p.ss_mp_pts[:,1]),p.ss_mp_pts[:,2]+theta,'r-',label='ss')
            #         plt.plot(np.degrees(p.ps_mp_pts[:,1]),p.ps_mp_pts[:,2]+theta,'b-',label='ps')
            #         theta += dtheta
            #     plt.legend()
            #     plt.xlabel('Theta')
            #     plt.ylabel('R')
            #     plt.axis('equal')
            #     plt.title(f'Theta-r Profile-{i}')
            #     plt.savefig(f'{prefix} theta-r {i:02d}.png',dpi=150)
            
            for i in range(len(profiles_debug)):
                p = profiles_debug[i]
                plt.figure(num=i*4+3,clear=True)        # x-r view
                xrth = self.get_camber_points(i,self.t_hub,self.t_camber)
                plt.plot(xrth[:,0],xrth[:,1]+theta, color='black', linestyle='dashed',linewidth=2,label='camber')
                plt.plot(p.ss_mp_pts[:,3],p.ss_mp_pts[:,2]+theta,'r-',label='ss')
                plt.plot(p.ps_mp_pts[:,3],p.ps_mp_pts[:,2]+theta,'b-',label='ps')
                plt.plot(self.hub_pts_cyl[0,:,0],self.hub_pts_cyl[0,:,1]+theta,'k',label='hub',alpha=0.2)
                plt.plot(self.shroud_pts_cyl[0,:,0],self.shroud_pts_cyl[0,:,1]+theta,'k',label='shroud',alpha=0.2)
                
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
        # for i in range(p.ps_cyl_pts.shape[0]):
        i=1
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
    

def centrif_create_te(SS:List[float],PS:List[float],camber:bezier,radius:float=1, n_te_pts:int=15):
    """Add a trailing edge that's rounded

    Args:
        radius_scale (float): 0 to 1 as to how the radius shrinks with respect to spacing between ss and ps last control points
        wedge_ss (float): suction side wedge angle
        wedge_ps (float): pressure side wedge angle 
        elliptical (float): 1=circular, any value >1 controls how it is elliptical
    """
    xn,yn = camber.get_point(1) # End of camber line
    dx_dt,dy_dt = camber.get_point_dt(1)
    r = np.sqrt((PS[-1,0] - SS[-1,0])**2 + (PS[-1,1] - SS[-1,1])**2)
    dx1=(PS[-1,0] - PS[-2,0])*0.02
    dy1=(PS[-1,1] - PS[-2,1])*0.02
    # x = [PS[-1,0],PS[-1,0]+dx1,xn+dy_dt*radius,xn] 
    # y = [PS[-1,1],PS[-1,1]+dy1,yn-dx_dt*radius,yn]
    x = [PS[-2,0],PS[-1,0],xn+dy_dt*radius,xn] 
    y = [PS[-2,1],PS[-1,1],yn-dx_dt*radius,yn]
    
    x = [float(p) for p in x];y = [float(p) for p in y]
    ps_arc = bezier(x, y)
    
    dx1=(SS[-1,0] - SS[-2,0])*0.02
    dy1=(SS[-1,1] - SS[-2,1])*0.02
    # x = [SS[-1,0],SS[-1,0]+dx1,xn-dy_dt*radius,xn]
    # y = [SS[-1,1],SS[-1,1]+dy1,yn+dx_dt*radius,yn]
    x = [SS[-2,0],SS[-1,0],xn-dy_dt*radius,xn]
    y = [SS[-2,1],SS[-1,1],yn+dx_dt*radius,yn]
    
    x = [float(p) for p in x]; y = [float(p) for p in y]
    ss_arc = bezier(x,y)
    
    ps_te_pts = ps_arc.get_point(np.linspace(0,1,n_te_pts))
    ss_te_pts = ss_arc.get_point(np.linspace(0,1,n_te_pts))
    return np.array(ss_te_pts).transpose(), np.array(ps_te_pts).transpose(), ss_arc, ps_arc
 