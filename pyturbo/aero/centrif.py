from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union
from pyturbo.helper import convert_to_ndarray, line2D, bezier, exp_ratio, arc, ellispe
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

class Centrif:
    hub:npt.NDArray
    shroud:npt.NDArray
    profiles:List[CentrifProfile]
    blade_position:Tuple[float,float] # Start and End positions
    
    func_xhub:PchipInterpolator
    func_rhub:PchipInterpolator
    func_xshroud:PchipInterpolator
    func_rshroud:PchipInterpolator
    
    camber_t_th:List[bezier]
    
    __tip_clearance_percent:float = 0
    __tip_clearance:float = 0
    
    t_chord:npt.ArrayLike                   # This is the t from hub to shroud 
    t_span:npt.ArrayLike
    t_blade:npt.ArrayLike                   
    npts_chord:int              
    npts_span:int               
    
    ss_pts:npt.NDArray
    ps_pts:npt.NDArray
    
    ss_t_theta = []
    ps_t_theta = []
    
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
    
    def __get_camber_xr_point__(self,t_span:float,t_chord:float) -> npt.NDArray:
        # Returns the x,r point. Doesn't require vertical line test 
        shroud_pts = np.hstack([self.func_xshroud(t_chord),self.func_rshroud(t_chord)])
        hub_pts = np.hstack([self.func_xhub(t_chord),self.func_rhub(t_chord)])    
        l = line2D(hub_pts,shroud_pts)
        x,r = l.get_point(t_span)
        return np.array([x,r])
    
    def __get_camber_xr__(self,t_span:float) -> npt.NDArray:
        # Returns xr for the camber line. Doesn't require vertical line test 
        # t_chord is the percent along the hub 
        shroud_pts = np.vstack([self.func_xshroud(self.t_chord),self.func_rshroud(self.t_chord)]).transpose()
        hub_pts = np.vstack([self.func_xhub(self.t_chord),self.func_rhub(self.t_chord)]).transpose()
        xr = np.zeros((self.npts_chord,2))
        for j in range(self.npts_chord):
            l = line2D(hub_pts[j,:],shroud_pts[j,:])
            xr[j,0],xr[j,1] = l.get_point(t_span)
        return xr
    
    def get_camber_points(self,i:int):
        """Get the camber in cylindrical coordinates x,r,th

        Args:
            i (int): camber index

        Returns:
            (npt.NDArray): x, r, theta 
            
        """
        xr = self.__get_camber_xr__(self.profiles[i].percent_span)
        _,th = self.camber_t_th[i].get_point(self.t_blade)      # using t_blade matches up with t_chord from get_camber_xr
        xrth = np.hstack([xr,np.asmatrix(th).transpose()])
        return xrth
    
    def get_camber_length(self,i:int):
        xrth = self.get_camber_points(i)
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
            
            dx = self.__get_camber_xr_point__(profile.percent_span,self.t_chord[-1])[0] - self.__get_camber_xr_point__(profile.percent_span,self.t_chord[0])[0]
            dr = self.__get_camber_xr_point__(profile.percent_span,self.t_chord[-1])[1] - self.__get_camber_xr_point__(profile.percent_span,self.t_chord[0])[1]
            dth = np.radians(profile.warp_angle)
            camb_len = np.sum(np.sqrt(dr**2+dth**2 * dth**2 + dx**2))
            
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
    
    def __apply_thickness__(self):
        """Apply thickness to the cambers 
        """  
        # Apply thickness in theta direction 
        for i,profile in enumerate(self.profiles):            
            xrth = self.get_camber_points(i)
            dx = np.diff(xrth[:,0])
            dr = np.diff(xrth[:,1])
            dth = np.diff(xrth[:,2])
            indx = np.argmax(np.cumsum(np.flip(np.sqrt(dr**2+xrth[:,1]**2 * dth**2 + dx**2))) < profile.trailing_edge_properties.TE_Radius)
            t_te_starts = self.t_chord[indx]
            
            camber = self.camber_t_th[i]
            
            # Add SS and PS Points 
            npoint_ss = len(profile.ss_thickness)+2 # Leading Edge Thickness, SS_Thickness, TE_Radius
            npoint_ps = len(profile.ps_thickness)+2 # Leading Edge Thickness, PS_Thickness, TE_Radius
            SS = np.zeros((npoint_ss,2)); PS = np.zeros((npoint_ps,2))  # t, theta 
            ps_wedge = profile.trailing_edge_properties.TE_WedgeAngle_PS
            ss_wedge = profile.trailing_edge_properties.TE_WedgeAngle_SS
            te_radius = profile.trailing_edge_properties.TE_Radius
            te_radius_scale = profile.trailing_edge_properties.radius_scale
            
            PS[0,0] = profile.LE_Thickness * np.cos(np.radians(profile.LE_Metal_Angle+90))
            PS[0,1] = profile.LE_Thickness * np.sin(np.radians(profile.LE_Metal_Angle+90))
            SS[0,0] = profile.LE_Thickness * np.cos(np.radians(profile.LE_Metal_Angle-90))
            SS[0,1] = profile.LE_Thickness * np.sin(np.radians(profile.LE_Metal_Angle-90))
            
            
            t = exp_ratio(1.2,npoint_ps)        # Pressure side thicknesses 
            for j in range(len(profile.ps_thickness)):
                dx,dth = camber.get_point_dt(t[j+1])
                angle = np.degrees(np.arctan2(dth,dx))
                PS[j+1,0] = profile.ps_thickness[j] * np.cos(np.radians(angle+90))
                PS[j+1,1] = profile.ps_thickness[j] * np.sin(np.radians(angle+90))
            dx,dth = camber.get_point_dt(t[-1]) # TE Radius
            ps_angle = np.degrees(np.arctan2(dth,dx))
            PS[-1,0] = te_radius * np.cos(np.radians(ps_angle+90-ps_wedge))
            PS[-1,1] = te_radius * np.sin(np.radians(ps_angle+90-ps_wedge))
            
            t = exp_ratio(1.2,npoint_ss)        # Suction side thicknesses
            for j in range(len(profile.ss_thickness)):
                dx,dth = camber.get_point_dt(t[j+1])
                angle = np.degrees(np.arctan2(dth,dx))
                SS[j+1,0] = profile.ss_thickness[j] * np.cos(np.radians(angle-90))
                SS[j+1,1] = profile.ss_thickness[j] * np.sin(np.radians(angle-90))
            dx,dth = camber.get_point_dt(t[-1]) # TE Radius
            ss_angle = np.degrees(np.arctan2(dth,dx))
            SS[-1,0] = te_radius * np.cos(np.radians(ss_angle-90+ss_wedge))
            SS[-1,1] = te_radius * np.sin(np.radians(ss_angle-90+ss_wedge))
            
            

            # Add TE Points 
            if te_radius_scale == 1: # Circle                
                ps_te = arc(PS[-1,0],PS[-1,1],te_radius,ps_angle,ps_angle+90-ps_wedge)
                ss_te = arc(SS[-1,0],SS[-1,1],te_radius,ss_angle,ss_angle-90+ss_wedge)
                
                ps_te_x, ps_te_y = ps_te.get_point(np.linspace(0,1,10))
                ss_te_x, ss_te_y = ss_te.get_point(np.linspace(0,1,10))
                ps_te_pts = np.column_stack([ps_te_x,ps_te_y])
                ss_te_pts = np.column_stack([ss_te_x,ss_te_y])
                ss_te_pts = np.flipud(ss_te_pts)
            else:
                te_metal_angle = np.radians(profile.TE_Metal_Angle)
                xc,yc = camber.get_point(1)
                ellispe_te = ellispe(xc,yc,te_radius_scale,te_radius,
                                 alpha_start=90-ps_wedge,
                                 alpha_stop=-90+ss_wedge)
                te_x, te_y = ellispe_te.get_point(np.linspace(0,1,20))

                n = te_x.shape[0]; n2 = int(n/2)
                ps_te_pts = np.flipud(np.stack([te_x[n2-1:],te_y[n2-1:]],axis=1))
                ss_te_pts = np.stack([te_x[:n2],te_y[:n2]],axis=1)
                
                rot = np.array([[np.cos(te_metal_angle), -np.sin(te_metal_angle)],
                    [np.sin(te_metal_angle), np.cos(te_metal_angle)]])[:,:,0]
                xc = (ps_te_pts[:,0].sum() + ss_te_pts[:,0].sum()) / (ps_te_pts.shape[0] + ss_te_pts.shape[0])
                yc = (ps_te_pts[:,1].sum() + ss_te_pts[:,1].sum()) / (ps_te_pts.shape[0] + ss_te_pts.shape[0])
                
                
                ps_te_pts[:,0] = ps_te_pts[:,0]-xc
                ps_te_pts[:,1] = ps_te_pts[:,1]-yc
                
                ss_te_pts[:,0] = ss_te_pts[:,0]-xc
                ss_te_pts[:,1] = ss_te_pts[:,1]-yc
                
                ps_te_pts = np.matmul(rot,ps_te_pts.transpose()).transpose()
                ss_te_pts = np.matmul(rot,ss_te_pts.transpose()).transpose()
                
                ps_te_pts[:,0] = ps_te_pts[:,0]+xc
                ps_te_pts[:,1] = ps_te_pts[:,1]+yc
                
                ss_te_pts[:,0] = ss_te_pts[:,0]+xc
                ss_te_pts[:,1] = ss_te_pts[:,1]+yc
            
            self.ss_t_theta.append(np.vstack([SS,ss_te_pts]))
            self.ps_t_theta.append(np.vstack([PS,ps_te_pts]))
            
            # ss = NURBS.Curve()
            # ss.degree = 3 # Cubic
            # ctrlpts = np.concatenate([ 
            #                         SS,
            #                         ss_te_pts
            #                     ])
            # ss.ctrlpts = ctrlpts
            # ss.delta = 1/self.npts_chord
            # ss.knotvector = knotvector.generate(ss.degree,ctrlpts.shape[0])
            
            # ps = NURBS.Curve()
            # ps.degree = 3 # Cubic
            # ctrlpts = np.concatenate([ 
            #                         PS,
            #                         ps_te_pts
            #                     ])
            # ps.ctrlpts = ctrlpts
            # ps.delta = 1/self.npts_chord
            # ps.knotvector = knotvector.generate(ps.degree,ctrlpts.shape[0])
            
            # ss_pts.append(ss.evalpts)
            # ps_pts.append(ps.evalpts)        
    def __interpolate__(self):
        ss_pts = np.zeros((len(self.ss_t_theta),self.npts_chord+10,4)) # x,r,theta,t 
        ps_pts = np.zeros((len(self.ss_t_theta),self.npts_chord+10,4))
        for i in range(len(self.ss_t_theta)):
            ss_t_theta = self.ss_t_theta[i]
            xrth = self.get_camber_points(i)            
            theta = PchipInterpolator(ss_t_theta[:,0],ss_t_theta[:,1])(self.t_chord)
            ss_pts[i,:,0] = xrth[:,0]
            ss_pts[i,:,1] = theta
            ss_pts[i,:,2] = xrth[:,1]
            
            ps_t_theta = self.ps_t_theta[i]
            xrth = self.get_camber_points(i)            
            theta = PchipInterpolator(ps_t_theta[:,0],ps_t_theta[:,1])(self.t_chord)
            ps_pts[i,:,0] = xrth[:,0]
            ps_pts[i,:,1] = theta
            ps_pts[i,:,2] = xrth[:,1]
            
    
    def __tip_clearance__(self):
        """Build the tspan matrix such that tip clearance is maintained
        """
        self.t_span = np.zeros((self.npts_span,self.npts_chord))
        self.t_chord = np.linspace(0,1,self.npts_chord)
        t = self.t_chord * (self.blade_position[1]-self.blade_position[0]) + self.blade_position[0]
        
        xh = self.func_xhub(t); xsh = self.func_xshroud(t)
        rh = self.func_rhub(t); rsh = self.func_rshroud(t)
                
        for j in range(len(self.t_chord)):
            cut = line2D([xh[j],rh[j]],[xsh[j],rsh[j]])
            t2 = cut.get_t(cut.length-self.tip_clearance)
            self.t_span[:,j] = np.linspace(0,t2,self.npts_span)
    
    def build(self,npts_span:int=100, npts_chord:int=100):
        """Build the centrif blade 

        Args:
            npts_span (int, optional): _description_. Defaults to 100.
            npts_chord (int, optional): _description_. Defaults to 100.
        """
        self.npts_chord = npts_chord; self.npts_span = npts_span
        self.t_chord = np.linspace(0,1,npts_chord)
        self.t_span = np.linspace(0,1,npts_span)
        
        splitter_start = self.profiles[0].splitter_camber_start
        if splitter_start == 0:
            t_splitter = self.t_chord * (self.blade_position[1]-self.blade_position[0]) + self.blade_position[0]
        else:
            self.t_chord = self.t_chord * (self.blade_position[1]-splitter_start) + splitter_start
        self.t_camber = (self.t_chord - self.t_chord.min())/(self.t_chord.max()-self.t_chord.min())
        self.__build_camber__()
        self.__build_hub_shroud__()
        self.__apply_thickness__()
        self.__interpolate__()
        
    def plot_camber(self,plot_hub_shroud:bool=True):
        """Plot the camber line
        """
        t = self.t_blade
        fig = plt.figure(num=1,dpi=150)
        for i,b in enumerate(self.camber_t_th):
            [x,y] = b.get_point(t)        
            plt.plot(x, y,'-b',label=f"curve {i}")
            plt.plot(b.x, b.y,'or',label=f"curve {i}")
        plt.legend()
        plt.show()
        
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
        
    def plot_profiles(self):
        
        plt.figure(num=1,clear=True)
        plt.plot(xcamber,ycamber, color='black', linestyle='solid', 
            linewidth=2)
        plt.plot(self.ps_pts[:,0],self.ps_pts[:,1],'b',label='pressure side')
        plt.plot(self.ss_pts[:,0],self.ss_pts[:,1],'r',label='suction side')
        plt.plot(self.ss_pts[max_indx,0],self.ss_pts[max_indx,1],'rx',label='suction side max')
        
        plt.plot(self.ps_x,self.ps_y,'ob',label='ps ctrl pts')
        plt.plot(self.ss_x,self.ss_y,'or',label='ss ctrl pts')
        plt.plot(self.ps_te_pts[:,0],self.ps_te_pts[:,1],'ok',label='ps te ctrl pts')
        plt.plot(self.ss_te_pts[:,0],self.ss_te_pts[:,1],'om',label='ss te ctrl pts')
        plt.legend()
        plt.axis('scaled')
        plt.show()
    def plot_front_view(self):
        pass