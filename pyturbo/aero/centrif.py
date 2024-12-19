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
from sklearn.preprocessing import MinMaxScaler

class WaveDirection(Enum):
    x:int = 0
    r:int = 1


@dataclass 
class TrailingEdgeProperties:
    # Flat Trailing Edge 
    TE_Cut:bool = False                     # Defaults to no cut TE
    
    # Circular Trailing Edge
    TE_Radius:float = 0.005                 # In theta
    TE_WedgeAngle_SS:float = 5              # Wedge angle in theta, not used if CUT_TE is selected
    TE_WedgeAngle_PS:float = 3              # Wedge angle in theta, not used if CUT_TE is selected
    
    # Elliptical Trailing Edge
    radius_scale:float = 1                  # 0 to 1 as to how the radius shrinks with respect to spacing between ss and ps last control points

def DefineCircularTE(radius:float,SS_WedgeAngle:float, PS_WedgeAngle:float,major_axis_scale:float=1) -> TrailingEdgeProperties:
    """Create either a circular or elliptical trailing edge 

    Args:
        radius (float): radius of trailing edge 
        SS_WedgeAngle (float): suction side wedge angle
        PS_WedgeAngle (float): pressure side wedge angle
        major_axis_scale (float): 1 - circular; >1 elliptical 

    Returns:
        TrailingEdgeProperties: TE Properties Object containing the settings
    """
    return TrailingEdgeProperties(TE_Cut=False,TE_Radius=radius,TE_WedgeAngle_PS=PS_WedgeAngle,TE_WedgeAngle_SS=SS_WedgeAngle,radius_scale=major_axis_scale)

def DefineCutTE(radius:float) -> TrailingEdgeProperties:
    """Define a simple cut trailing edge. Radius is defined as theta

    Args:
        radius (float): radius of trailing edge

    Returns:
        TrailingEdgeProperties: TE Properties Object containing the settings
    """
    return TrailingEdgeProperties(TE_Cut=True,TE_Radius=radius)
    
@dataclass 
class WavyBladeProperties:
    LE_Wave:npt.NDArray
    TE_wave:npt.NDArray
    SS_Wave:npt.NDArray
    PS_Wave:npt.NDArray
    LE_Wave_angle:Optional[List[float]]=None
    TE_Wave_angle:Optional[List[float]]=None
    

@dataclass
class CentrifProfile:
    percent_span:float                      
    LE_Thickness:float                      # In theta
    
    LE_Metal_Angle:float
    TE_Metal_Angle:float
    
    LE_Metal_Angle_Loc:float
    TE_Metal_Angle_Loc:float
    
    ss_thickness:List[float]
    ps_thickness:List[float]
    
    warp_angle:float                        # angle of warp/theta
    warp_displacements:List[float]          # percent of warp_angle
    warp_displacement_locs:List[float]      # percent chord
    
    trailing_edge_properties:TrailingEdgeProperties
    
    splitter_camber_start:float = 0         # What percentage along the camberline to start the splitter 
    npts_te: int = 10                       # Suction side and pressure side 
    
@dataclass
class CentrifProfileDebug:
    SS:npt.NDArray              # ControlPoints rx-theta
    PS:npt.NDArray  
    camber_rx_th:npt.NDArray    # Camber points rx-theta
    ss_rx_pts:npt.NDArray          # Generated ss points rx-theta
    ps_rx_pts:npt.NDArray          # Generated ps points rx-theta
    ss_te:npt.NDArray
    ps_te:npt.NDArray
    rx_end:float
    th_end:float
    center_rx:float
    center_th:float
    

def cylindrical_to_cartesian(rho, phi, z):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z

class Centrif:
    hub:npt.NDArray
    shroud:npt.NDArray
    profiles:List[CentrifProfile]
    profiles_debug:List[CentrifProfileDebug]
    blade_position:Tuple[float,float] # Start and End positions
    
    func_xhub:PchipInterpolator
    func_rhub:PchipInterpolator
    func_xshroud:PchipInterpolator
    func_rshroud:PchipInterpolator
    
    camber_t_th:List[bezier]
    
    __tip_clearance_percent:float = 0
    __tip_clearance:float = 0
    
    t_hub:npt.ArrayLike                   # This is the t from hub to shroud 
    t_span:npt.ArrayLike
    t_blade:npt.ArrayLike                   
    npts_chord:int              
    npts_span:int               
    
    ss_pts:npt.NDArray          # shape=(nspan, nchord, 3)
    ps_pts:npt.NDArray          # shape=(nspan, nchord, 3)
    ss_profile_pts:npt.NDArray  # shape=(nprofile, nchord, 3)
    ps_profile_pts:npt.NDArray  # shape=(nprofile, nchord, 3)
    ss_rx_pts:npt.NDArray       # shape=(nprofile,nchord,6), 6 = rx,theta,r,x,t_camber,t_span
    ps_rx_pts:npt.NDArray
    camb_rx_func:List[PchipInterpolator]
    
    def __init__(self):
        self.profiles = list()
        self.blade_position = (0,1)
        
    
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
    
    def add_profile(self,profile:CentrifProfile):
        """Add warp and adjustment

        Args:
            warp_angle (float): _description_
            warp_adjustment (List[WarpAdjustment]): 
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
        shroud_pts = np.hstack([self.func_xshroud(t_hub),self.func_rshroud(t_hub)])
        hub_pts = np.hstack([self.func_xhub(t_hub),self.func_rhub(t_hub)])    
        l = line2D(hub_pts,shroud_pts)
        x,r = l.get_point(t_span)
        return np.array([x,r])
    
    ''' 
    '''
    def __get_camber_xr__(self,t_span:float,t_hub:npt.NDArray) -> npt.NDArray:
        """Get the x and r coordinates of the camberline

        Args:
            t_span (float): percent span
            t_hub (npt.NDArray): percent along the hub from 0 to 1. Defaults to None.
        
        Returns:
            npt.NDArray: Array nx2 containing xr 
        """
        # Returns xr for the camber line. Doesn't require vertical line test 
        # t_hub is the percent along the hub 
       
        shroud_pts = np.vstack([self.func_xshroud(t_hub),self.func_rshroud(t_hub)]).transpose()
        hub_pts = np.vstack([self.func_xhub(t_hub),self.func_rhub(t_hub)]).transpose()
        n = len(t_hub)
            
        xr = np.zeros((n,2))
        for j in range(n):
            l = line2D(hub_pts[j,:],shroud_pts[j,:])
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
 
        xr = self.__get_camber_xr__(self.profiles[i].percent_span,t_hub)
        _,th = self.camber_t_th[i].get_point(t_camber) # using t_camber matches up with t_hub from get_camber_xr
        xrth = np.hstack([xr,th.reshape(-1,1)])
        return xrth
    
    def get_camber_length(self,i:int):
        xrth = self.get_camber_points(i,self.t_camber)
        dr = np.diff(xrth[:,1])
        dx = np.diff(xrth[:,0])
        dth = np.diff(xrth[:,2])
        return np.sum(np.sqrt(dr**2+xrth[:,1]**2 * dth**2 + dx**2))
    
    def __build_camber__(self):
        t = np.linspace(0,1,self.hub.shape[0])
        self.func_xhub = PchipInterpolator(t,self.hub[:,0])
        self.func_rhub = PchipInterpolator(t,self.hub[:,1])
        self.func_xshroud = PchipInterpolator(t,self.shroud[:,0])
        self.func_rshroud = PchipInterpolator(t,self.shroud[:,1])
            
        # Build camber_xr        
        self.camber_t_th = list()
        i = 0
        for profile in self.profiles:
            # r1 = starting radius, r2 = ending radius 
            t_start = self.blade_position[0]; t_end = self.blade_position[-1]
            l = line2D([t_start,0],[t_end,np.radians(profile.warp_angle)])
            n = np.array([l.dx,-l.dy])/np.linalg.norm([l.dx,-l.dy])
            l.get_point(profile.LE_Metal_Angle_Loc)[1]
            # warp_displacement_locs: percent chord
            # warp displacement: percent of warp_angle
            camber_bezier_t_th = np.zeros(shape=(4+len(profile.warp_displacements),2))           # Bezier Control points in the t,theta plane
            camber_bezier_t_th[0,:] = [t_start, 0]
            camber_bezier_t_th[1,:] = [profile.LE_Metal_Angle_Loc, profile.LE_Metal_Angle_Loc*np.tan(np.radians(profile.LE_Metal_Angle))]
            camber_bezier_t_th[-2,:] = [profile.TE_Metal_Angle_Loc, (t_end-profile.TE_Metal_Angle_Loc)*np.tan(np.radians(profile.TE_Metal_Angle))+np.radians(profile.warp_angle)]
            camber_bezier_t_th[-1,:] = [t_end, np.radians(profile.warp_angle)]
            
            xr0 = self.__get_camber_xr_point__(profile.percent_span,self.t_hub[-1])
            xr1 = self.__get_camber_xr_point__(profile.percent_span,self.t_hub[0])
            camb_len = np.sqrt(xr0[1]**2+xr1[1]**2 -2*xr0[1]*xr1[1]*np.cos(np.radians(profile.warp_angle)) + (xr0[0]-xr1[0])**2)
            
            j = 2
            for loc,displacement in zip(profile.warp_displacement_locs, profile.warp_displacements):
                displacement = camb_len*displacement
                x1,y1 = l.get_point(loc)
                x2 = n[1]*displacement + x1
                y2 = n[0]*displacement + y1
                camber_bezier_t_th[j,0] = x2
                camber_bezier_t_th[j,1] = y2
                j+=1
            i+=1 
            self.camber_t_th.append(bezier(camber_bezier_t_th[:,0],camber_bezier_t_th[:,1]))
    
    def __build_hub_shroud__(self):
        self.hub_pts = np.vstack([
                self.func_xhub(np.linspace(0,1,self.npts_chord*2)),
                self.func_xhub(np.linspace(0,1,self.npts_chord*2))*0, 
                self.func_rhub(np.linspace(0,1,self.npts_chord*2))]).transpose()
        self.shroud_pts = np.vstack([
            self.func_xshroud(np.linspace(0,1,self.npts_chord*2)),
            self.func_xshroud(np.linspace(0,1,self.npts_chord*2))*0, 
            self.func_rshroud(np.linspace(0,1,self.npts_chord*2))]).transpose()
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
    @staticmethod
    def __apply_te__(center_pt:Tuple[float,float],
                     start_pt:Tuple[float,float],
                     end_pt:Tuple[float,float],
                     end_pt2:Tuple[float,float],
                     te_properties:TrailingEdgeProperties):
        """Apply Trailing Edge in the clockwise direction 
        
        Note: 
            If trailing edge is a circle or ellispe; when defining the pressure side component of the trailing edge, the start point is on the pressure side. However if we are dealing with suction side, the trailing edge starts at the camber end point and rotates to the end of the suction side 
        
        Args:
            center_pt (Tuple[float,float]): center point in terms of rx, theta
            start_pt (Tuple[float,float]): start point eg. pressure side point
            end_pt (Tuple[float,float]): end point eg. trailing edge
            end_pt2 (Tuple[float,float]): end point eg. suction side point
            te_properties (TrailingEdgeProperties): Trailing edge properties 
        """
        te_radius_scale = te_properties.radius_scale     # Ellipitcal TE 
        te_radius = te_properties.TE_Radius
        if te_properties.TE_Cut:
            th = np.linspace(end_pt[1],end_pt[1]+te_radius,10)
            te_pts_upper = np.flipud(np.concat([end_pt[0]+0*th, th],axis=1))
            
            th = np.linspace(end_pt[1]-te_radius,end_pt[1],10)
            te_pts_lower = np.concat([end_pt2[0]+0*th, th],axis=1)
            
        else:
            start_angle = np.degrees(np.arctan2(start_pt[1]-center_pt[1],start_pt[0]-center_pt[0]))
            end_angle = np.degrees(np.arctan2(end_pt[1]-center_pt[1],end_pt[0]-center_pt[0]))
            
            end_angle2 = np.degrees(np.arctan2(end_pt2[1]-center_pt[1],end_pt2[0]-center_pt[0]))
            
            if te_radius_scale == 1:
                te_arc1 = arc(center_pt[0],center_pt[1],te_radius,start_angle, end_angle)                
                te_x, te_y = te_arc1.get_point(np.linspace(0,1,10))
                te_pts_upper = np.column_stack([te_x,te_y])
                
                te_arc2 = arc(center_pt[0],center_pt[1],te_radius,end_angle, end_angle2)                
                te_x, te_y = te_arc2.get_point(np.linspace(0,1,10))
                te_pts_lower = np.column_stack([te_x,te_y])
                
        return te_pts_upper,te_pts_lower            
        
    def __apply_thickness__(self):
        """Apply thickness to the cambers 
        """  
        
        self.ss_rx_pts = np.zeros(shape=(len(self.profiles),self.npts_chord,6)) # rx,theta,r,x,t_camber,t_span
        self.ps_rx_pts = np.zeros(shape=(len(self.profiles),self.npts_chord,6))
        
        self.profiles_debug = list()
        self.camb_rx_func = list()
        
        def solve_t(t,val:float,func:PchipInterpolator):
            return np.abs(val-func(t))
        
        # Apply thickness in theta direction 
        for i,profile in enumerate(self.profiles):  
            # Flatten in xr
            xrth = self.get_camber_points(i,self.t_hub,self.t_camber)
            dx = np.diff(xrth[:,0])
            dr = np.diff(xrth[:,1])
            dist_rx = np.hstack([[0],np.cumsum(np.sqrt(dx**2+dr**2))])
            func_rx = PchipInterpolator(self.t_camber,dist_rx)
            self.camb_rx_func.append(func_rx)
            dfunc_rx = func_rx.derivative()
            
            # Apply Thickness in theta
            npoint_ss = len(profile.ss_thickness)+3 # LE Start, Leading Edge Thickness, SS_Thickness, TE_Radius
            npoint_ps = len(profile.ps_thickness)+3 # LE Start,Leading Edge Thickness, PS_Thickness, TE_Radius
            SS = np.zeros((npoint_ss,2)); PS = np.zeros((npoint_ps,2))  # rx, theta 
            ps_wedge = profile.trailing_edge_properties.TE_WedgeAngle_PS
            ss_wedge = profile.trailing_edge_properties.TE_WedgeAngle_SS
            te_radius = profile.trailing_edge_properties.TE_Radius
            
            dist = [np.sqrt(xrth[-1,1]**2+xrth[j,1]**2 - 2*xrth[-1,1]*xrth[j,1]*np.cos(xrth[-1,2]-xrth[j,2])+(xrth[-1,1]-xrth[j,1])**2) for j in range(len(xrth))]
            # dist = (dist_rx[-1]-dist_rx[j])**2 + (camber_th[-1]-camber_th[j])**2
            dist = np.array(dist)
            indx = np.argmin(np.array(dist) > profile.trailing_edge_properties.TE_Radius)
            t_te_starts = self.t_camber[indx]         

            camber = self.camber_t_th[i]
            t_start,th_start = camber.get_point(0)                      # Add LE Start
            SS[0,0] = func_rx(t_start)
            SS[0,1] = th_start
            PS[0,0] = func_rx(t_start)
            PS[0,1] = th_start
                                                                        # Add LE Thickness
            PS[1,0] = profile.LE_Thickness * np.cos(np.radians(profile.LE_Metal_Angle+90))      # rx
            PS[1,1] = profile.LE_Thickness * np.sin(np.radians(profile.LE_Metal_Angle+90))      # theta
            
            SS[1,0] = profile.LE_Thickness * np.cos(np.radians(profile.LE_Metal_Angle-90))      # rx
            SS[1,1] = profile.LE_Thickness * np.sin(np.radians(profile.LE_Metal_Angle-90))      # theta
            
            t_ps = np.linspace(0,1,npoint_ps)*t_te_starts # exp_ratio(1.2,npoint_ps,maxvalue=t_te_starts)        # Pressure side thicknesses 
            for j in range(len(profile.ps_thickness)):  
                _,th_start = camber(t_ps[j+1])
                rx_start = func_rx(t_ps[j+1])
                _,dth_dt = camber.get_point_dt(t_ps[j+1])
                drx_dt = dfunc_rx(t_ps[j+1])
                angle = np.degrees(np.arctan2(dth_dt,drx_dt))
                PS[j+2,0] = rx_start + profile.ps_thickness[j] * np.cos(np.radians(angle+90))
                PS[j+2,1] = th_start + profile.ps_thickness[j] * np.sin(np.radians(angle+90))
            
            _,th_start = camber(t_ps[-1])
            rx_start = func_rx(t_ps[-1])
            _,dth_dt = camber.get_point_dt(t_ps[-1])
            drx_dt = dfunc_rx(t_ps[-1])
            angle = np.degrees(np.arctan2(dth_dt,drx_dt))
            PS[-1,0] = rx_start + te_radius * np.cos(np.radians(angle+90-ps_wedge))
            PS[-1,1] = th_start + te_radius * np.sin(np.radians(angle+90-ps_wedge))
            
            t_ss = t_ps = np.linspace(0,1,npoint_ps)*t_te_starts # exp_ratio(1.2,npoint_ss,maxvalue=t_te_starts)        # Suction side thicknesses
            for j in range(len(profile.ss_thickness)):
                _,th_start = camber(t_ps[j+1])
                rx_start = func_rx(t_ps[j+1])
                _,dth_dt = camber.get_point_dt(t_ps[j+1])
                drx_dt = dfunc_rx(t_ps[j+1])
                angle = np.degrees(np.arctan2(dth_dt,drx_dt))
                SS[j+2,0] = rx_start + profile.ss_thickness[j] * np.cos(np.radians(angle-90))
                SS[j+2,1] = th_start + profile.ss_thickness[j] * np.sin(np.radians(angle-90))
            _,th_start = camber(t_ss[-1])
            rx_start = func_rx(t_ss[-1])
            _,dth_dt = camber.get_point_dt(t_ss[-1])
            drx_dt = dfunc_rx(t_ss[-1])
            angle = np.degrees(np.arctan2(dth_dt,drx_dt))
            SS[-1,0] = rx_start + te_radius * np.cos(np.radians(angle-90+ss_wedge))
            SS[-1,1] = th_start + te_radius * np.sin(np.radians(angle-90+ss_wedge))
            
            center_rx = func_rx(t_te_starts)
            _,center_th = camber(t_te_starts)
            rx_end = func_rx(1)
            _,th_end = camber(1)
            
            ps_te,ss_te = self.__apply_te__(center_pt=(center_rx,center_th),
                                            start_pt=PS[-1,:],end_pt=(rx_end,th_end),
                                            end_pt2=SS[-1,:],te_properties=profile.trailing_edge_properties)
            ps_nurbs_ctrl_pts = np.vstack([PS[:-1,:],ps_te])            # rx,theta
            ss_nurbs_ctrl_pts = np.vstack([SS[:-1,:],np.flipud(ss_te)]) # rx,theta

            self.ss_rx_pts[i,:,:2] = self.__NURBS_interpolate__(ss_nurbs_ctrl_pts,self.npts_chord) # rx,theta
            self.ps_rx_pts[i,:,:2] = self.__NURBS_interpolate__(ps_nurbs_ctrl_pts,self.npts_chord) # rx,theta
            
            self.profiles_debug.append(CentrifProfileDebug(SS=SS,PS=PS,
                                camber_rx_th=np.vstack([func_rx(self.t_camber),camber(self.t_camber)[1]]).transpose(),
                                ss_rx_pts=self.ss_rx_pts,ps_rx_pts=self.ps_rx_pts,
                                ss_te=ss_te,ps_te=ps_te,rx_end=rx_end,th_end=th_end,center_rx=center_rx,center_th=center_th))
            
            m = 1/(self.t_hub.max()-self.t_hub.min())
            # Inversely solve for t_camber for each rx value 
            for j in range(self.npts_chord):
                rx = self.ss_rx_pts[i,j,0]
                res = minimize_scalar(solve_t,bounds=[0,1],args=(rx,func_rx))
                self.ss_rx_pts[i,j,4] = res.x # new t_camber for rx value 
                thub = 1/m * res.x + self.t_hub.min()
                xrth = self.get_camber_points(i,[thub],self.ss_rx_pts[i,j,4])[0]
                self.ss_rx_pts[i,j,2] = xrth[1] # r
                self.ss_rx_pts[i,j,3] = xrth[0] # x
                

                rx = self.ps_rx_pts[i,j,0]
                res = minimize_scalar(solve_t,bounds=[0,1],args=(rx,func_rx))
                self.ps_rx_pts[i,j,4] = res.x
                thub = 1/m * res.x + self.t_hub.min()   # Converting t-camber to thub
                xrth = self.get_camber_points(i,[thub],self.ps_rx_pts[i,j,4])[0]
                self.ps_rx_pts[i,j,2] = xrth[1] # r
                self.ps_rx_pts[i,j,3] = xrth[0] # x
            
            self.ss_rx_pts[i,:,5] = self.t_span[i,:] # tspan 
            self.ps_rx_pts[i,:,5] = self.t_span[i,:] 
            
            
            
    def __interpolate__(self):    
        """Interpolate the control profiles and get x,r,th
        """
        self.ss_pts = np.zeros(shape=(self.npts_span,self.npts_chord,3)) # 3 = (x,r,theta)
        self.ps_pts = np.zeros(shape=(self.npts_span,self.npts_chord,3)) # 3 = (x,r,theta)
        
        for j in range(self.npts_chord):
            tspan = np.linspace(0,self.t_span[-1,j],self.ss_rx_pts.shape[0])
            self.ss_pts[:,j,0] = csapi(tspan,self.ss_rx_pts[:,j,3],self.t_span[:,j])   # x
            self.ss_pts[:,j,1] = csapi(tspan,self.ss_rx_pts[:,j,2],self.t_span[:,j])   # r
            self.ss_pts[:,j,2] = csapi(tspan,self.ss_rx_pts[:,j,1],self.t_span[:,j])   # th
            
            self.ps_pts[:,j,0] = csapi(tspan,self.ps_rx_pts[:,j,3],self.t_span[:,j])   # x
            self.ps_pts[:,j,1] = csapi(tspan,self.ps_rx_pts[:,j,2],self.t_span[:,j])   # r
            self.ps_pts[:,j,2] = csapi(tspan,self.ps_rx_pts[:,j,1],self.t_span[:,j])   # th
            
            
    
    def __tip_clearance__(self):
        """Build the tspan matrix such that tip clearance is maintained
        """
        self.t_span = np.zeros((self.npts_span,self.npts_chord))
        self.t_hub = np.linspace(0,1,self.npts_chord)
        t = self.t_hub * (self.blade_position[1]-self.blade_position[0]) + self.blade_position[0]
        
        xh = self.func_xhub(t); xsh = self.func_xshroud(t)
        rh = self.func_rhub(t); rsh = self.func_rshroud(t)
                
        for j in range(len(self.t_hub)):
            cut = line2D([xh[j],rh[j]],[xsh[j],rsh[j]])
            t2 = cut.get_t(cut.length-self.tip_clearance)
            self.t_span[:,j] = exp_ratio(1.2,self.npts_span,maxvalue=t2)
    
    def build(self,npts_span:int=100, npts_chord:int=100):
        """Build the centrif blade 

        Args:
            npts_span (int, optional): _description_. Defaults to 100.
            npts_chord (int, optional): _description_. Defaults to 100.
        """
        self.npts_chord = npts_chord; self.npts_span = npts_span
        self.t_hub = np.linspace(0,1,npts_chord)
        self.t_span = np.linspace(0,1,npts_span)
        
        splitter_start = self.profiles[0].splitter_camber_start
        if splitter_start == 0:
            t_splitter = self.t_hub * (self.blade_position[1]-self.blade_position[0]) + self.blade_position[0]
        else:
            self.t_hub = self.t_hub * (self.blade_position[1]-splitter_start) + splitter_start
        self.t_camber = (self.t_hub - self.t_hub.min())/(self.t_hub.max()-self.t_hub.min())
        self.__build_camber__()
        self.__build_hub_shroud__()
        self.__tip_clearance__()
        self.__apply_thickness__()  # Creates the flattened profiles
        self.__interpolate__()
        
    def plot_camber(self,plot_hub_shroud:bool=True):
        """Plot the camber line
        """
        t = self.t_camber
        fig = plt.figure(num=1,dpi=150)
        for i,b in enumerate(self.camber_t_th):
            [x,y] = b.get_point(t)        
            plt.plot(x, y,'-b',label=f"curve {i}")
            plt.plot(b.x, b.y,'or',label=f"curve {i}")
            plt.title(f'Profile {i}')
            plt.savefig(f'Centrif_Profile_{i:2d}')
                
        fig = plt.figure(num=2,dpi=150)
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
            
            xrth = self.get_camber_points(k)
            ax.plot3D(xrth[:,0],xrth[:,2],xrth[:,1],'-b',linewidth=2)
            k+=1
        # Plots the hub and shroud 
        if plot_hub_shroud:
            ax.plot3D(self.hub_pts[:,0],self.hub_pts[:,1],self.hub_pts[:,2],'k',linewidth=2.5)
            ax.plot3D(self.shroud_pts[:,0],self.shroud_pts[:,1],self.shroud_pts[:,2],'k',linewidth=2.5)
        ax.view_init(68,-174)
        plt.axis('equal')
        plt.show()
        
    def plot_rx_profile(self):
        """Plot the control profiles in the rx-theta plane
        """
        for i in range(len(self.profiles_debug)):
            p = self.profiles_debug[i]
            plt.figure(num=i,clear=True)    # rx view
            plt.plot(p.camber_rx_th[:,0],p.camber_rx_th[:,1], color='black', linestyle='dashed',linewidth=2,label='camber')
            plt.plot(p.SS[:,0],p.SS[:,1],'ro', label='suction')
            plt.plot(p.ss_te[0,0],p.ss_te[0,1],'ro',fillstyle='none', label='suction-te') 
            plt.plot(p.ss_te[-1,0],p.ss_te[-1,1],'ro',fillstyle='none', label='suction-te') 
            plt.plot(p.PS[:,0],p.PS[:,1],'bo',label='pressure')
            plt.plot(p.ps_te[0,0],p.ps_te[0,1],'bo',fillstyle='none', label='pressure-te') 
            plt.plot(p.ps_te[-1,0],p.ps_te[-1,1],'bo',fillstyle='none', label='pressure-te') 
            plt.plot(p.rx_end,p.th_end,'go',fillstyle='none', label='end-pt') 
            plt.plot(p.center_rx,p.center_th,'ko',fillstyle='none', label='center-pt')
            plt.plot(p.ss_rx_pts[i,:,0],p.ss_rx_pts[i,:,1],'r-',label='ss')
            plt.plot(p.ps_rx_pts[i,:,0],p.ps_rx_pts[i,:,1],'b-',label='ps')
            plt.legend()
            plt.xlabel('distance along r and x')
            plt.ylabel('theta')
            plt.title(f'RX Profile-{i}')
            plt.axis('equal')
            plt.savefig(f'profile rx-theta {i:02d}.png',dpi=150)
            
            plt.figure(num=i,clear=True)    # x-theta view
            xrth = self.get_camber_points(i,self.t_hub,self.t_camber)
            plt.plot(xrth[:,0],xrth[:,2], color='black', linestyle='dashed',linewidth=2,label='camber')
            plt.plot(p.ss_rx_pts[i,:,3],p.ss_rx_pts[i,:,1],'r-',label='ss')
            plt.plot(p.ps_rx_pts[i,:,3],p.ps_rx_pts[i,:,1],'b-',label='ps')
            plt.legend()
            plt.xlabel('Distance along x')
            plt.ylabel('Theta')
            plt.title(f'X-Theta Profile-{i}')
            plt.axis('equal')
            plt.savefig(f'profile x-theta {i:02d}.png',dpi=150)
            
            
    def plot(self):
        """3D Cartesian Plot
        """
        fig = plt.figure(num=100,clear=True,dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot3D(self.hub_pts[:,0],self.hub_pts[:,0]*0,self.hub_pts[:,2],'k')
        # ax.plot3D(self.shroud_pts[:,0],self.shroud_pts[:,0]*0,self.shroud_pts[:,2],'k')
        for i in range(self.ss_pts.shape[0]):
            ax.plot3D(self.ss_pts[i,:,0],self.ss_pts[i,:,2],self.ss_pts[i,:,1],'r') # x,
            ax.plot3D(self.ps_pts[i,:,0],self.ps_pts[i,:,2],self.ps_pts[i,:,1],'b')
        ax.set_xlabel('x-axial')
        ax.set_ylabel('theta')
        ax.set_zlabel('r-radial')
        ax.view_init(azim=90, elev=45)
        plt.axis('equal')
        plt.show()
        
        
            
    
    def plot_front_view(self):
        plt.figure(num=1,clear=True) # Front view theta-r
        for i in range(self.npts_span):
            plt.plot(self.ss_pts[i,:,1],self.ss_pts[i,:,2])         # rx,theta,r,x,t_camber,t_span
            plt.plot(self.ps_pts[i,:,1],self.ps_pts[i,:,2])
        plt.xlabel('theta')
        plt.ylabel('r')
        plt.title(f'Theta-r Profile-{i}')
        plt.axis('equal')
        plt.savefig(f'profile theta-r {i:02d}.png',dpi=150)