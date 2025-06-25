from code import interact
import numpy as np
import numpy.typing as npt
import math
from typing import List, Optional, Tuple
from ..helper import convert_to_ndarray, bezier, bezier3, centroid, check_replace_max, check_replace_min, csapi, resample_by_curvature
from ..helper import create_cubic_bounding_box, cosd, sind, uniqueXY, pspline, line2D, ray2D, pspline_intersect, dist, spline_type
from .airfoil2D import Airfoil2D
from ..helper import StackType, combine_and_sort
from scipy.optimize import minimize_scalar
import enum
import copy
import os
import glob
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from tqdm import trange
from stl import mesh

class Airfoil3D:
    '''
        Properties
    '''
    profileArray: List[Airfoil2D]
    profileSpan: npt.NDArray
    span:float 
    IsSplineFitted: bool
    IsSplineFittedShell: bool 
    
    stackType:StackType 
    '''Bezier X,Y,Z are for the stacking line of the blade'''
    stack_bezier_ctrl_pts: npt.NDArray

    te_center: npt.NDArray     # Center of trailing edge for each span profile
    
    stack_bezier: bezier3                 # 3D Bezier curve that defines the stacking
    bImportedBlade: bool

    sweepY: npt.NDArray          # Array defining the sweep and lean points.
    sweepZ: npt.NDArray          #

    leanX: npt.NDArray
    leanY: npt.NDArray

    nte: int                    # Number of points defining the trailing edge
    npts: int                   # Number of points defining the suction and pressure side
    nspan: int                  # Number of cross sections along the span


    '''3D Blade points: Final blade points after lean and sweep are accounted for'''
    spline_ps: npt.NDArray     # Profile Suction side (nprofiles,npts,3)
    spline_ss: npt.NDArray     # Profile Pressure Side (nprofiles,npts,3)
    
    spline_te_ss: npt.NDArray   # Trailing edge suction side points
    spline_te_ps: npt.NDArray

    spine:npt.NDArray 
    # Blade cross sections defined without Lean or Sweep (nprofiles,npoints,3)
    ss: npt.NDArray             
    ps: npt.NDArray             
    te_ss: npt.NDArray
    te_ps: npt.NDArray
    
    # Location of the stacking line control points
    control_ss: npt.NDArray    
    control_ps: npt.NDArray
    
    c_te_ss: npt.NDArray
    c_te_ps: npt.NDArray
    
    # Suction side Final points with lean and sweep, These are the points that are exported
    shft_ss: npt.NDArray        
    shft_ps: npt.NDArray        # Pressure side Final points with lean and sweep, These are the points that are exported
    
    '''Shell'''
    shell_xss: List[PchipInterpolator]  # curves defining the shelled geometry
    shell_yss: List[PchipInterpolator]
    shell_zss: List[PchipInterpolator]

    shell_xps: List[PchipInterpolator]
    shell_yps: List[PchipInterpolator]
    shell_zps: List[PchipInterpolator]

    """Defines a 3D Airfoil to be used in a channel
    """
    def __init__(self, profileArray: List[Airfoil2D], profile_loc: List[float], height: float):
        """Constructs a 3D Airfoil

        Args:
            profileArray (List[Airfoil2D]): array of Airfoil2D profiles
            profile_loc (List[float]): location of the Airfoil2D profiles
            height (float): height of the 3D blade
        """
        self.profileArray = profileArray
        self.profileSpan = convert_to_ndarray(profile_loc)
        self.span = height
        self.IsSplineFitted = False
        self.IsSplineFittedShell = False

    def stack(self, stackType=StackType.centroid):
        """Each airfoil profile is stacked on top of each other based on the leading edge, trailing edge, or centroid

        Args:
            stackType (StackType, optional): Options are centroid, Leading Edge, Trailing Edge. Defaults to StackType.centroid.
        """
        stack_bezier_ctrl_pts = np.zeros((len(self.profileArray),3))
        self.stackType = stackType
        te_center = np.zeros((len(self.profileArray),3))
        if (len(self.profileArray) >= 2):
            # Stack the airfoils about LE
            if (stackType == StackType.leading_edge):
                hub = self.profileArray[0]
                [hx, hy] = hub.camberBezier.get_point(0)
                stack_bezier_ctrl_pts[0,0] = hx[0]
                stack_bezier_ctrl_pts[0,1] = hy[0]
                stack_bezier_ctrl_pts[0,2] = 0

                [hx_te, hy_te] = hub.camberBezier.get_point(1)
                te_center[0] = hx_te[0]
                te_center[1] = hy_te[0]

                for i in range(1, len(self.profileArray)):
                    [x, y] = self.profileArray[i].camberBezier.get_point(0)
                    dx = hx[0]-x[0]
                    dy = hy[0]-y[0]
                    # Shift the points based on camber
                    self.profileArray[i].shift(dx, dy)
                    stack_bezier_ctrl_pts[i,0] = hx[0]
                    stack_bezier_ctrl_pts[i,1] = hy[0]
                    stack_bezier_ctrl_pts[i,2] = self.profileSpan[i]*self.span

                    [hx_te, hy_te] = self.profileArray[i].camberBezier.get_point(1)
                    te_center[i,0] = hx_te[0]
                    te_center[i,1] = hy_te[0]
            # Stack the airfoils about TE
            elif (stackType == StackType.trailing_edge):
                hub = self.profileArray[0]
                [hx, hy] = hub.camberBezier.get_point(1)
                stack_bezier_ctrl_pts[0,0] = hx[0]
                stack_bezier_ctrl_pts[0,1] = hy[0]
                stack_bezier_ctrl_pts[0,2] = 0

                [hx_te, hy_te] = hub.camberBezier.get_point(1)
                te_center[0,0] = hx_te[0]
                te_center[0,1] = hy_te[0]

                for i in range(1, len(self.profileArray)):
                    [x, y] = self.profileArray[i].camberBezier.get_point(1)
                    dx = 0
                    dy = 0
                    # Shift the points based on camber
                    self.profileArray[i].shift(dx, dy)
                    stack_bezier_ctrl_pts[0,0] = hx[0]
                    stack_bezier_ctrl_pts[0,1] = hy[0]
                    stack_bezier_ctrl_pts[0,2] = self.profileSpan[i]*self.span

                    [hx_te, hy_te] = self.profileArray[i].camberBezier.get_point(
                        1)
                    te_center[i,0] = hx_te[0]
                    te_center[i,1] = hy_te[0]
            elif (stackType == StackType.centroid):
                [hx, hy] = self.profileArray[0].get_centroid()
                stack_bezier_ctrl_pts[0,0] = hx
                stack_bezier_ctrl_pts[0,1] = hy
                stack_bezier_ctrl_pts[0,2] = 0

                [hx_te, hy_te] = self.profileArray[0].camberBezier.get_point(1)
                te_center[0,0] = hx_te[0]
                te_center[0,1] = hy_te[0]

                for i in range(1, len(self.profileArray)):
                    [x, y] = self.profileArray[i].get_centroid()
                    dx = hx-x
                    dy = hy-y
                    # Shift the points based on camber
                    self.profileArray[i].shift(dx, dy)
                    stack_bezier_ctrl_pts[i,0] = hx
                    stack_bezier_ctrl_pts[i,1] = hy
                    stack_bezier_ctrl_pts[i,2] = self.profileSpan[i]*self.span

                    [hx_te, hy_te] = self.profileArray[i].camberBezier.get_point(
                        1)
                    te_center[i,0] = hx_te[0]
                    te_center[i,1] = hy_te[0]
                
            self.stack_bezier_ctrl_pts = stack_bezier_ctrl_pts
            self.stack_bezier = bezier3(stack_bezier_ctrl_pts[:,0],stack_bezier_ctrl_pts[:,1],stack_bezier_ctrl_pts[:,2])
            self.bImportedBlade = False
            self.te_center = te_center

    
    
    def add_sweep(self,sweep_y:List[float]=[],sweep_z:List[float]=[]):
        """Sweep bends the blade towards the leading edge or trailing edge. Blades are first stacked and then sweep can be applied

        Args:
            sweep_y (List[float], optional): adds bezier points to the sweep of the blade. Defaults to [].
            sweep_z (List[float], optional): defines where along the span the sweep points should be located. Defaults to [].
        """
        self.sweepY = convert_to_ndarray(sweep_y)
        self.sweepZ = convert_to_ndarray(sweep_z)

        # add sweep where z exists
        self.sweepZ *= self.span
        self.sweepY *= self.span
        
        x_sweep = self.sweepZ*0
        
        b = np.vstack([x_sweep,self.sweepY,self.sweepZ]).transpose()
        results = combine_and_sort(self.stack_bezier_ctrl_pts,b)
        self.stack_bezier_ctrl_pts = results  # New set of points
        

    def add_lean(self,leanX:List[float],leanZ:List[float]):
        """Leans the blade towards the suction or pressure side. This applies points that are fitted by a bezier curve. Profiles are adjusted to follow this curve simulating lean.

        Args:
            leanX (List[float]): lean points
            leanZ (List[float]): spanwise location of the lean points
        """
        self.leanX = convert_to_ndarray(leanX)
        self.leanZ = convert_to_ndarray(leanZ)

        self.leanZ *= self.span
        self.leanX *= self.span
        
        y_lean = self.leanX*0
        
        b = np.vstack([self.leanX,y_lean,self.leanZ]).transpose()
        results = combine_and_sort(self.stack_bezier_ctrl_pts,b)
        self.stack_bezier_ctrl_pts = results
        
    def build(self,nProfiles:int,num_points:int,trailing_edge_points:int):
        """Takes the control profiles specified in the construct and creates intermediate profiles filling the blade geometry. These profiles can be shifted or modified later. 

        Args:
            nProfiles (int): number of intermeidate profiles to generate
            num_points (int): number of points per profile. Suction and Pressure side will have this number of points
            trailing_edge_points (int): Number of trailing edge points
        """
        self.bImportedBlade = False
        
        # n - number of points to use for pressure and suction sides
        self.npts = num_points # number of points to use for suction and pressure side
        self.nte = trailing_edge_points
        self.nspan = nProfiles

        self.zz = np.linspace(0,self.span,self.nspan) # Spanwise locations of the profiles
        t = np.linspace(0,1,self.npts)
        t_te = np.linspace(0,1,self.nte)
        n_profiles = len(self.profileArray)
        # x,y,z profiles - stores the x,y,z coordinate for each profile for a given time
        spline_ps_temp = np.zeros((n_profiles,self.npts,3)) # x,y,z
        spline_ss_temp = np.zeros((n_profiles,self.npts,3))
        
        # z coordinates of the blade
        # --- Initialize and clear the profile points ---
        self.ps = np.zeros((self.nspan,self.npts,3)) # nprofiles in span, npts, (x,y,z)
        self.ss = np.zeros((self.nspan,self.npts,3)) 
        self.te_ss = np.zeros((self.nspan,self.nte,3)) # nprofiles in span, npts, (x,y,z)
        self.te_ps = np.zeros((self.nspan,self.nte,3)) # nprofiles in span, npts, (x,y,z)

        # --- Make a spline for each profile for each point
        for j in range(n_profiles):
            spline_ps_temp[j,:,0],spline_ps_temp[j,:,1] = self.profileArray[j].psBezier.get_point(t,equally_space_pts=True)
            spline_ps_temp[j,:,2] = self.profileSpan[j]*self.span              # Span

            spline_ss_temp[j,:,0],spline_ss_temp[j,:,1] = self.profileArray[j].ssBezier.get_point(t,equally_space_pts=True)
            spline_ss_temp[j,:,2] = self.profileSpan[j]*self.span              # Span

        for i in range(self.npts):
            self.ps[:,i,0] = csapi(spline_ps_temp[:,i,2],spline_ps_temp[:,i,0],self.zz) # Natural spline
            self.ps[:,i,1] = csapi(spline_ps_temp[:,i,2],spline_ps_temp[:,i,1],self.zz)

            self.ss[:,i,0] = csapi(spline_ss_temp[:,i,2],spline_ss_temp[:,i,0],self.zz) # Natural spline
            self.ss[:,i,1] = csapi(spline_ss_temp[:,i,2],spline_ss_temp[:,i,1],self.zz)

        self.c_te_ps = np.zeros((self.nte,n_profiles,3))
        self.c_te_ss = np.zeros((self.nte,n_profiles,3))
        for j in range(n_profiles): # Trailing edge contains less points
            # trailing edge
            [self.c_te_ps[:,j,0],self.c_te_ps[:,j,1]] = self.profileArray[j].TE_ps_arc.get_point(t_te)
            [self.c_te_ss[:,j,0],self.c_te_ss[:,j,1]] = self.profileArray[j].TE_ss_arc.get_point(t_te)
            self.c_te_ps[:,j,2] = self.profileSpan[j]*self.span
            self.c_te_ss[:,j,2] = self.profileSpan[j]*self.span
        
        for i in range(self.nte):
            # Trailing edge pressure side
            self.te_ps[:,i,0] = csapi(self.c_te_ps[i,:,2],self.c_te_ps[i,:,0],self.zz)
            self.te_ps[:,i,1] = csapi(self.c_te_ps[i,:,2],self.c_te_ps[i,:,1],self.zz)
            self.te_ps[:,i,2] = self.zz
            # Trailing edge suction side
            self.te_ss[:,i,0] = csapi(self.c_te_ss[i,:,2],self.c_te_ss[i,:,0],self.zz)
            self.te_ss[:,i,1] = csapi(self.c_te_ss[i,:,2],self.c_te_ss[i,:,1],self.zz)
            self.te_ss[:,i,2] = self.zz
            
        te_center = np.zeros((nProfiles,3)) # Trailing edge center for each profile
        te_center[:,0] = csapi(self.profileSpan*self.span,self.te_center[:,0],self.zz)
        te_center[:,1] = csapi(self.profileSpan*self.span,self.te_center[:,1],self.zz)
        te_center[:,2] = self.zz
        
        self.te_center = te_center
        # Populate Control Points
        self.control_ps = np.zeros((n_profiles,num_points+trailing_edge_points,3))
        self.control_ss = np.zeros((n_profiles,num_points+trailing_edge_points,3))
        for i in range(n_profiles):
            ss, ps = self.profileArray[i].get_points(num_points+trailing_edge_points)
            self.control_ps[i,:,0] = ps[:,0] 
            self.control_ps[i,:,1] = ps[:,1]
            self.control_ss[i,:,0] = ss[:,0] 
            self.control_ss[i,:,1] = ss[:,1]

        # Lets stack the profiles 
        self.stack_bezier = bezier3(self.stack_bezier_ctrl_pts[:,0],self.stack_bezier_ctrl_pts[:,1],self.stack_bezier_ctrl_pts[:,2])
    
        # Combine suction and pressure side with trailing edge
        self.ps = np.hstack([self.ps,self.te_ps])
        self.ss = np.hstack([self.ss,self.te_ss])
        
        nprofiles,_,_ = self.ps.shape
        
        # Shift all points by bezier curve
        self.shft_ps = copy.deepcopy(self.ps)
        self.shft_ss = copy.deepcopy(self.ss)
        
        # Shift all the generated turbine profiles points based on the bezier curve
        self.__stack_profiles__(self.shft_ss,self.shft_ps,self.te_center)
        self.__stack_profiles__(self.control_ss,self.control_ps)

        # Equal Space points
        for i in trange(nProfiles,desc='Equal Spacing suction and pressure side'):
            self.shft_ss[i,:,:] = resample_by_curvature(self.shft_ss[i,:,:],self.shft_ss.shape[1])
            self.shft_ps[i,:,:] = resample_by_curvature(self.shft_ps[i,:,:],self.shft_ps.shape[1])
         
        for i in range(len(self.profileSpan)):
            self.control_ps[i,:,2] = self.profileSpan[i]*self.span
            self.control_ss[i,:,2] = self.profileSpan[i]*self.span

    def shift(self,x:float,y:float):
        """Moves the blade
            Step 1 - shifts the profiles
            Step 2 - add lean and sweep again
            Step 3 - recompute the geometry if npts != 0

        Args:
            x (float): shift in x direction
            y (float): shift in y directrion
        """
        [nprofiles,_,_] = self.shft_ss.shape
        for i in range(nprofiles):
            self.shft_ss[i,:,1] += y
            self.shft_ss[i,:,0] += x
            self.shft_ps[i,:,1] += y
            self.shft_ps[i,:,0] += x

    def flip_x(self):
        """Mirrors the blade by multiplying -1*x direction. This is assuming axial chord is in the y direction and span is in z
        """
        self.shft_ps[:,:,0] *= -1
        self.shft_ss[:,:,0] *= -1
        self.control_ps[:,:,0] *= -1 
        self.control_ss[:,:,0] *= -1 

    def flip_y(self):
        """Mirrors the blade by multiplying y direction by -1. This is assuming axial chord is in the y direction and span is in z
        """
        self.shft_ps[:,:,1] = -1*self.shft_ps[:,:,1]
        self.shft_ss[:,:,1] = -1*self.shft_ss[:,:,1]

    def section_z(self,zStartPercent:float,zEndPercent:float):
        """Chops the blade in between 2 spanwise lines. Think of it as cutting the blade between (zStartPercent) 10% and (zEndPercent) 50%
        Args:
            zStartPercent (float): bottom % of the blade to cut from
            zEndPercent (float): top % of blade to cut from. stuff in middle is saved
        """
        npts = self.shft_ps.shape[0]

        for i in range(npts):
            # create a spline for each profile's shift points
            mn_zss = min(self.shft_ss[:,i,2])
            mn_zps = min(self.shft_ps[:,i,2])
            mx_zss = max(self.shft_ss[:,i,2])
            mx_zps = max(self.shft_ps[:,i,2])
            h_ss = mx_zss-mn_zss # height suction side
            h_ps = mx_zps-mn_zps
            
            zss = np.linspace(mn_zss+h_ss*zStartPercent,mn_zss+h_ss*zEndPercent,self.npts)
            zps = np.linspace(mn_zps+h_ps*zStartPercent,mn_zps+h_ps*zEndPercent,self.npts)

            self.shft_ss[:,i,1]= csapi(self.shft_ss[:,i,2],self.shft_ss[:,i,1],zss)
            self.shft_ss[:,i,0]= csapi(self.shft_ss[:,i,2],self.shft_ss[:,i,0],zss)
            self.shft_ps[:,i,1]= csapi(self.shft_ps[:,i,2],self.shft_ps[:,i,1],zps)
            self.shft_ps[:,i,0]= csapi(self.shft_ps[:,i,2],self.shft_ps[:,i,0],zps)
            self.shft_ps[:,i,2] = zps
            self.shft_ss[:,i,2] = zss

        # self.cylindrical()

    def cartesian(self):
        """
            Converts the default cylindrical coordinates to cartesian system
        """
        nprofiles = self.shft_ss.shape[0]
        for i in range(nprofiles): # for each 2d blade profile in the 3d blade
            [self.shft_ss[i,:,0],self.shft_ss[i,:,2],_] = self.convert_cyl_cartesian(self.shft_ss[i,:,0],self.shft_ss[i,:,2])
            [self.shft_ps[i,:,0],self.shft_ps[i,:,2],_] = self.convert_cyl_cartesian(self.shft_ps[i,:,0],self.shft_ps[i,:,2])

    def export_solidworks(self,name:str):
        """Export the blades in RTheta,Z,R coordinate format

        Args:
            name (string): exported filename
        """
        if (not os.path.exists('solidworks')):
            os.mkdir('solidworks')
        folder = 'solidworks/{0}'.format(name)
        if (not os.path.exists(folder)):
            os.mkdir(folder)
        # Export all the sections into RTheta,Z,R format
        # Export the Blade
        n = self.shft_ss.shape[0]; # n - number of points, m - number of sections
        for j in range(n):
            x = np.append(self.shft_ss[j,:,0], np.flip(self.shft_ps[j,:,0]))
            y = np.append(self.shft_ss[j,:,1], np.flip(self.shft_ps[j,:,1]))
            [x,y] = uniqueXY(x,y)

            with open('{0}/blade_section{1:03d}.txt'.format(folder,j),'w') as f:
                for k in range(len(x)):
                    f.write("{0:08f} {1:08f} {2:08f}\n".format(x[k],y[k],self.zz[j])) # Number of sections

    def plot3D(self,only_blade=False):
        """Plots a 3D representation of the blade and control points trailing edge center line is also plotted along with the blade's stacking spine

        Args:
            only_blade (bool, optional): Only plot the blade, no stacking spine. Defaults to False.

        Returns:
            (matplotlib.figure): figure object (fig.show())
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if not only_blade:
            # Plot the control profiles
            if (not self.bImportedBlade):
                nprofiles = self.control_ps.shape[0]
                for p in range(nprofiles):
                    ax.plot3D(self.control_ps[p,:,0], self.control_ps[p,:,1], self.control_ps[p,:,2],color='green') # type: ignore
                    ax.plot3D(self.control_ss[p,:,0], self.control_ss[p,:,1], self.control_ss[p,:,2],color='green') # type: ignore
                # Plot trailing edge center
                ax.plot3D(self.te_center[:,0],self.te_center[:,1],self.te_center[:,2],color='black') # type: ignore

        # Plot the profiles
        nprofiles = self.shft_ss.shape[0]
        xmax=0.0; ymax=0.0; zmax=0.0
        xmin=0.0; ymin=0.0; zmin=0.0
        for i in range(nprofiles):
            ax.plot3D(self.shft_ss[i,:,0],self.shft_ss[i,:,1],self.shft_ss[i,:,2],color='red') # type: ignore
            ax.plot3D(self.shft_ps[i,:,0],self.shft_ps[i,:,1],self.shft_ps[i,:,2],color='blue') # type: ignore
            xmax = check_replace_max(xmax,np.max(np.append(self.shft_ps[i,:,0],self.shft_ss[i,:,0])))
            xmin = check_replace_min(xmin,np.min(np.append(self.shft_ps[i,:,0],self.shft_ss[i,:,0])))

            ymax = check_replace_max(ymax,np.max(np.append(self.shft_ps[i,:,1],self.shft_ss[i,:,1])))
            ymin = check_replace_min(ymin,np.min(np.append(self.shft_ps[i,:,1],self.shft_ss[i,:,1])))

            zmax = check_replace_max(zmax,np.max(np.append(self.shft_ps[i,:,2],self.shft_ss[i,:,2])))
            zmin = check_replace_min(zmin,np.min(np.append(self.shft_ps[i,:,2],self.shft_ss[i,:,2])))


        Xb,Yb,Zb = create_cubic_bounding_box(xmax,xmin,ymax,ymin,zmax,zmin)

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z") # type: ignore
        plt.show()

    def nblades(self,pitchChord:float,rhub:float):
        """Calculates the number of blades given a pitch-to-chord ratio

        Args:
            pitchChord (float): pitch to chord ratio
            rhub (float): hub radius
        """
        pitch = self.profileArray[0].chord*pitchChord
        return math.floor(2*math.pi*rhub/pitch)


    def get_chord(self):
        """Returns the chord, axial chord for all the profiles
        """
        chord = np.sqrt((self.shft_ps[:,-1,0] - self.shft_ps[:,0,0])**2 + (self.shft_ps[:,-1,1] - self.shft_ps[:,0,1])**2)
        axial_chord = abs(self.shft_ps[:,-1,0] - self.shft_ps[:,0,0])
        # max_chord = max(chord)
        # avg_chord = np.mean(chord)

        # max_axial_chord = max(axial_chord)
        # avg_axial_chord = np.mean(axial_chord)
        return chord,axial_chord

    def get_pitch(self,nBlades:int):
        """Get the pitch distribution for a 3D blade row

        Args:
            nBlades (int): Number of blades

        Returns:
            (tuple): tuple containing:

            - **s** (numpy.ndarray): pitch distribution
            - **s_c** (numpy.ndarray): pitch to chord distribution
        """
        chord,_ = self.get_chord()
        r = self.shft_ps[:,-1,2]
        s = 2*math.pi*r/nBlades
        s_c = s/chord
        return s, s_c

    def __stack_profiles__(self,ss:npt.NDArray,ps:npt.NDArray,te_center:Optional[npt.NDArray]=None):
        nprofiles = ss.shape[0]
        spine = np.zeros((nprofiles,3))
        
        t = np.linspace(0,1,nprofiles)
        [bx,by,_] = self.stack_bezier.get_point(t,equally_space_pts=False)

        # Get centroid before combining with TE
        cx = np.zeros(nprofiles); cy = np.zeros(nprofiles)
        for i in range(nprofiles):
            x = bx[i]; y = by[i]
            # Get the centroid of each profile
            [cx[i], cy[i]] = centroid(np.concatenate((ps[i,:,0],ss[i,:,0])),np.concatenate((ps[i,:,1],ss[i,:,1])))

            # Shift the stack_bezier to align with the stacking 
            x = bx[i]; y = by[i]
            if (self.stackType == StackType.centroid):
                sx = cx[i]; sy = cy[i]
            elif (self.stackType == StackType.leading_edge):
                sx = ps[i,0,0]
                sy = ps[i,0,1]
            else: # (self.stackType == StackType.trailing_edge)
                sx = 0
                sy = 0

            # Pressure profiles
            spine[i,0] += x - sx
            spine[i,1] += y - sy
            ps[i,:,0] = ps[i,:,0] + x - sx
            ps[i,:,1] = ps[i,:,1] + y - sy
            # Suction profiles
            ss[i,:,0] = ss[i,:,0] + x - sx
            ss[i,:,1] = ss[i,:,1] + y - sy
            if te_center is not None:
                # Shift the trailing edge center
                te_center[i,0] = te_center[i,0] + x - sx   
                te_center[i,1] = te_center[i,1] + y - sy

            ps[i,:,2] = self.zz[i]
            ss[i,:,2] = self.zz[i]
        

    def convert_cyl_cartesian(self,rth:npt.NDArray,radius:npt.NDArray):
        """Convert a single profile from cylindrical to cartesian coordinates.

        Args:
            rth (npt.NDArray): points in rtheta coordinate system
            radius (npt.NDArray): radius values of those points

        Returns:
            (tuple): tuple containing:

            - **yss** (numpy.ndarray): y direction
            - **zss** (numpy.ndarray): z direction
            - **thss** (numpy.ndarray): theta direction
        """
        # Jorge's Program
        thss=np.zeros(len(rth))
        yss=np.zeros(len(rth))
        zss=np.zeros(len(rth))

        for i in range(0,len(rth)):
            thss[i]=rth[i]/radius[i]
            yss[i]=radius[i]*math.sin(thss[i])
            zss[i]=radius[i]*math.cos(thss[i])

        return yss,zss,thss

    def __check_camber_intersection__(self,ray,camber_x:npt.NDArray,camber_y:npt.NDArray):
        bIntersect = False
        for p in range(len(camber_x)-1): # check if ray intersects with
            camber_line = line2D((camber_x[p],camber_y[p]),(camber_x[p+1],camber_y[p+1]))
            [t,bIntersect] = camber_line.intersect_ray(ray)
            if (bIntersect):
                if (t==0 and (ray.x == camber_line.p[0] and ray.y == camber_line.p[1])): # if ray starting point is the same as line, don't count it
                    bIntersect = False
                elif (t<0): # type: ignore # if ray time vector is negative then it doesn't intersect
                    bIntersect = False
                else:
                    break
        return bIntersect

    def __check_ss_ray_intersection__(self,ray,ss_x:npt.NDArray,ss_y:npt.NDArray):
        """
            checks to see if the ray intersects the suction side
        """
        bIntersect = False
        for p in range(0,len(ss_x)-1): # check if ray intersects with
            ss_line = line2D((ss_x[p], ss_y[p]),(ss_x[p+1], ss_y[p+1]))
            [t,bIntersect] = ss_line.intersect_ray(ray)
            if (bIntersect):
                if (t==0 and (ray.x == ss_line.p[0]) and ray.y == ss_line.p[1]): # if ray starting point is the same as line, don't count it
                    bIntersect = False
                elif (t<0): # type: ignore # if ray time vector is negative then it doesn't intersect
                    bIntersect = False
                else:
                    break
        return bIntersect

    def __check_ps_ray_intersection__(self,ray,ps_x,ps_y):
        """
            checks to see if the ray intersects the pressure side
        """
        bIntersect = False
        for p in range(0,len(ps_x)-1): # check if ray intersects with
            ps_line = line2D((ps_x[p],ps_y[p]),(ps_x[p+1], ps_y[p+1]))
            [t,bIntersect] = ps_line.intersect_ray(ray)
            if (bIntersect):
                if (t==0 and (ray.x == ps_line.p[0] and ray.y == ps_line.p[1])): # if ray starting point is the same as line, don't count it
                    bIntersect = False
                elif (t[0]<0): # type: ignore # if ray time vector is negative then it doesn't intersect
                    bIntersect = False
                else:
                    break
        return bIntersect

    def get_cross_section_normal(self,ss_x:npt.NDArray,ss_y:npt.NDArray,ps_x:npt.NDArray,ps_y:npt.NDArray):
        """Gets the normal of a cross section for a given profile

        Args:
            ss_x (npt.NDArray): x - suction side points of a given profile
            ss_y (npt.NDArray): y - suction side points of a given profile
            ps_x (npt.NDArray): x - pressure side points of a given profile
            ps_y (npt.NDArray): y - pressure side points

        Returns:
            (tuple): tuple containing:

            - **ss_nx** (npt.NDArray): normal vector on suction side
            - **ss_ny** (npt.NDArray): normal vector on suction side
            - **ps_nx** (npt.NDArray): normal vector on pressure side
            - **ps_ny** (npt.NDArray): normal vector on pressure side
        """
        npts = len(ss_x)
        # Get Normal Vectors
        ss_nx = np.zeros(npts)
        ss_ny = np.zeros(npts)
        ps_nx = np.zeros(npts)
        ps_ny = np.zeros(npts)
        camber_x = (ss_x + ps_x)/2.0
        camber_y = (ss_y + ps_y)/2.0
        for i in range(npts):
            if (i==0):
                dx_1 = ss_x[i+1]-ss_x[i]
                dy_1 = ss_y[i+1]-ss_y[i]
                dx_2 = ps_x[i+1]-ps_x[i]
                dy_2 = ps_y[i+1]-ps_y[i]
            elif (i==npts-1):
                dx_1 = ss_x[i-1]-ss_x[i]
                dy_1 = ss_y[i-1]-ss_y[i]
                dx_2 = ps_x[i-1]-ps_x[i]
                dy_2 = ps_y[i-1]-ps_y[i]
            else:
                dx_1 = 1/2*(ss_x[i+1]-ss_x[i-1])
                dy_1 = 1/2*(ss_y[i+1]-ss_y[i-1])
                dx_2 = 1/2*(ps_x[i+1]-ps_x[i-1])
                dy_2 = 1/2*(ps_y[i+1]-ps_y[i-1])

            abs1 = np.linalg.norm([-dy_1,dx_1])
            abs2 = np.linalg.norm([-dy_2,dx_2])
            ss_nx[i] = -dy_1/abs1
            ss_ny[i] = dx_1/abs1
            ps_nx[i] = -dy_2/abs2
            ps_ny[i] = dx_2/abs2

        #  Use ray-ray intersection to make sure normals will not intersect
        for j in range(len(ps_x)):
            ss_ray = ray2D(ss_x[j],ss_y[j],ss_nx[j],ss_ny[j])
            ps_ray = ray2D(ps_x[j],ps_y[j],ps_nx[j],ps_ny[j])
            if (self.__check_camber_intersection__(ss_ray,camber_x,camber_y)):
                ss_nx[j] = -ss_nx[j]
                ss_ny[j] = -ss_ny[j]
            if (self.__check_camber_intersection__(ps_ray,camber_x,camber_y)):
                ps_nx[j] = -ps_nx[j]
                ps_ny[j] = -ps_ny[j]

        return ss_nx,ss_ny,ps_nx,ps_ny

    def spanwise_spline_fit(self):
        """
            Fits the blade with splines that run along the profiles
            Helps with the wavy design
        """
        [nprofiles,npts] = self.shft_ps.shape
        self.spline_xps = []
        self.spline_yps = []
        self.spline_zps = []

        self.spline_xss = []
        self.spline_yss = []
        self.spline_zss = []
        t = np.linspace(0,1,nprofiles)
        for p in range(npts):
            self.spline_xps.append(PchipInterpolator(t,self.shft_ps[:,p,0])) # Percent, x
            self.spline_yps.append(PchipInterpolator(t,self.shft_ps[:,p,1]))
            self.spline_zps.append(PchipInterpolator(t,self.shft_ps[:,p,2]))

            self.spline_xss.append(PchipInterpolator(t,self.shft_ss[:,p,0]))
            self.spline_yss.append(PchipInterpolator(t,self.shft_ss[:,p,1]))
            self.spline_zss.append(PchipInterpolator(t,self.shft_ss[:,p,2]))
        self.IsSplineFitted = True

    def spanwise_spline_fit_shell(self,shell_thickness:float,smooth:float=0.1):
        """Applies a shell in the spanwise direction. Creates a new blade inside the current blade. Helps with heat pipe design

        Args:
            shell_thickness (float): how much material to remove from the blade
            smooth (float, optional): How much smoothing should be performed. Defaults to 0.1.
        """
        self.IsSplineFittedShell = False
        shell_xss = []
        shell_yss = []

        shell_xps = []
        shell_yps = []
        nprofiles = self.shft_ps.shape[0]
        t = np.linspace(0,1,nprofiles)
        for profile in range(nprofiles):
            percent_profile = profile/nprofiles
            [ss_x,ss_y,_,ps_x,ps_y,_] = self.get_shell_2D(percent_profile,shell_thickness,smooth)
            shell_xss.append(ss_x)
            shell_yss.append(ss_y)
            shell_xps.append(ps_x)
            shell_yps.append(ps_y)

        shell_xss = convert_to_ndarray(shell_xss)
        shell_yss = convert_to_ndarray(shell_yss)
        shell_xps = convert_to_ndarray(shell_xps)
        shell_yps = convert_to_ndarray(shell_yps)

        self.shell_xss = []; self.shell_yss = []; self.shell_zss = []
        self.shell_xps = []; self.shell_yps = []; self.shell_zps = []

        [nprofiles,npts] = shell_xss.shape
        for p in range(npts):
            self.shell_xss.append(PchipInterpolator(t,shell_xss[:,p])) # Percent, x
            self.shell_yss.append(PchipInterpolator(t,shell_yss[:,p]))
            self.shell_zss.append(PchipInterpolator(t,self.shft_ps[:,p,0]))

            self.shell_xps.append(PchipInterpolator(t,shell_xps[:,p]))
            self.shell_yps.append(PchipInterpolator(t,shell_yps[:,p]))
            self.shell_zps.append(PchipInterpolator(t,self.shft_ss[:,p,2]))
        self.IsSplineFittedShell = True

    def get_cross_section(self,percent_span:float):
        """Get an arbirtary cross section at percent span. Doesn't factor into account the shift caused by the channel. Assumes the blade hasn't been placed inside of a channel.

        Args:
            percent_span (float): percent span to get the cross section

        Returns:
            (tuple): tuple containing:

            - **xss** (numpy.ndarray): suction side x coordinate
            - **yss** (numpy.ndarray): suction side y coordinate
            - **zss** (numpy.ndarray): suction side z coordinate
            - **xps** (numpy.ndarray): pressure side x coordinate
            - **yps** (numpy.ndarray): pressure side y coordinate
            - **zps** (numpy.ndarray): pressure side z coordinate

        """
        npts = self.ss.shape[1]

        if (not self.IsSplineFitted):
            self.spanwise_spline_fit()
        xps = np.zeros(npts); yps = np.zeros(npts); zps = np.zeros(npts)
        xss = np.zeros(npts); yss = np.zeros(npts); zss = np.zeros(npts)
        for i in range(npts):
            xps[i] = self.spline_xps[i](percent_span)
            yps[i] = self.spline_yps[i](percent_span)
            zps[i] = self.spline_zps[i](percent_span)

            xss[i] = self.spline_xss[i](percent_span)
            yss[i] = self.spline_yss[i](percent_span)
            zss[i] = self.spline_zss[i](percent_span)

        camber_x = (xps+xss)/2.0
        camber_y = (yps+yss)/2.0

        return xss,yss,zss,xps,yps,zps,camber_x,camber_y

    def get_shell_2D(self,percent_span:float,shell_thickness:float,smooth:float=0.1,shell_points:int=80):
        """Gets the 2D shell for a given % span


        Args:
            percent_span (float): where along the span do you want the shell the blade
            shell_thickness (float): offset from the outer wall
            smooth (float, optional): what percentage of the points do you want to use to smooth the design. Defaults to 0.1.
            shell_points (int, optional): number of points to describe the shell. Defaults to 80.

        Returns:
            (tuple): tuple containing:

            - **ss_x** (numpy.ndarray): suction side tangential (x) coordinates
            - **ss_y** (numpy.ndarray): suction side axial (y) coordinates
            - **ss_z** (numpy.ndarray): suction size radial (z) coordinates
            - **ps_x** (numpy.ndarray): pressure side tangential (x) coordinates
            - **ps_y** (numpy.ndarray): pressure side axial (y) coordinates
            - **ps_z** (numpy.ndarray): pressure side axial (z) coordinates

        """
        if (not self.IsSplineFittedShell):
            [ss_x,ss_y,ss_z,ps_x,ps_y,ps_z,_,_] = self.get_cross_section(percent_span)
            [ss_nx,ss_ny,ps_nx,ps_ny] = self.get_cross_section_normal(ss_x,ss_y,ps_x,ps_y)

            ss_x_new = ss_x + ss_nx * shell_thickness
            ss_y_new = ss_y + ss_ny * shell_thickness
            ps_x_new = ps_x + ps_nx * shell_thickness
            ps_y_new = ps_y + ps_ny * shell_thickness

            # Create a spline for suction and pressure side, check whether they intersect
            ss_spline = pspline(np.flip(ss_y_new),np.flip(ss_x_new)) # Points need to be in increasing order
            ps_spline = pspline(np.flip(ps_y_new),np.flip(ps_x_new))


            intersect1 = pspline_intersect(ss_spline,ps_spline,0,0.5) # Check for intersection at the beginning
            intersect2 = pspline_intersect(ss_spline,ps_spline,0.7,1.0) # Check for intersection at the end

            if (intersect1[0]!=-1.0): # near leading edge
                pt1,_ = ss_spline.get_point(intersect1[0])
                intersect1_yss = pt1[0,1]
                pt2,_ = ps_spline.get_point(intersect1[1])
                intersect1_yps = pt2[0,1]
                ss_x_new = ss_x_new[ss_y_new<intersect1_yss]
                ss_y_new = ss_y_new[ss_y_new<intersect1_yss]
                ps_x_new = ps_x_new[ps_y_new<intersect1_yps]
                ps_y_new = ps_y_new[ps_y_new<intersect1_yps]

            if (intersect2[0]!=-1.0): # trailing edge
                pt1,_ = ss_spline.get_point(intersect2[0])
                intersect2_yss = pt1[0,1]
                pt2,_ = ps_spline.get_point(intersect2[1])
                intersect2_yps = pt2[0,1]
                ss_x_new = ss_x_new[ss_y_new>intersect2_yss]
                ss_y_new = ss_y_new[ss_y_new>intersect2_yss]
                ps_x_new = ps_x_new[ps_y_new>intersect2_yps]
                ps_y_new = ps_y_new[ps_y_new>intersect2_yps]

            # Make sure the start and end points are the same
            # Make sure LE and TE start and end at the same point.
            startPointX = (ss_x_new[0] + ps_x_new[0])/2.0 # First X coordinate of all the profiles
            startPointY = (ss_y_new[0] + ps_y_new[0])/2.0 # First Y coordinate of all the profiles

            endPointX = (ss_x_new[-1] + ps_x_new[-1])/2.0 # last x-coordinate of all the profiles
            endPointY = (ss_y_new[-1] + ps_y_new[-1])/2.0 # last y-coordinate of all the profiles
            ss_x_new[0] = startPointX; ps_x_new[0] = startPointX
            ss_y_new[0] = startPointY; ps_y_new[0] = startPointY

            ss_x_new[-1] = endPointX; ps_x_new[-1] = endPointX
            ss_y_new[-1] = endPointY; ps_y_new[-1] = endPointY

            # Interpolate the curves
            t = np.linspace(0,1,shell_points)
            ss_spline = pspline(ss_x_new,ss_y_new,method=spline_type.pchip)
            ps_spline = pspline(ps_x_new,ps_y_new,method=spline_type.pchip)

            pt,_ = ss_spline.get_point(t)
            ss_x_new = pt[:,0]; ss_y_new = pt[:,1]
            pt,_ = ps_spline.get_point(t)
            ps_x_new = pt[:,0]; ps_y_new = pt[:,1]

            # Smooth the shell at the leading edge
            x = np.append(np.flip(ss_x_new[0:math.floor(shell_points*smooth)]),ps_x_new[1:math.floor(shell_points*smooth)])
            y = np.append(np.flip(ss_y_new[0:math.floor(shell_points*smooth)]),ps_y_new[1:math.floor(shell_points*smooth)])
            # Fit with bezier curve
            t = np.linspace(0,1,math.floor(shell_points*smooth)*2)
            b = bezier(x,y)
            [x,y] = b.get_point(t,equally_space_pts=False)
            d = dist(x,y,ss_x[0],ss_y[0])
            min_indx = np.where(d== np.amin(d))[0][0]
            ss_x_temp = x[0:min_indx+1]
            ss_y_temp = y[0:min_indx+1]
            ps_x_temp = x[min_indx:len(x)]
            ps_y_temp = y[min_indx:len(y)]

            ss_spline = pspline(ss_x_temp,ss_y_temp)
            ps_spline = pspline(ps_x_temp,ps_y_temp)
            t = np.linspace(0,1,math.floor(shell_points*smooth))

            pt1,_ = ss_spline.get_point(t)

            ss_x_new[0:math.floor(shell_points*smooth)] = np.flip(pt1[:,0])
            ss_y_new[0:math.floor(shell_points*smooth)] = np.flip(pt1[:,1])

            pt2,_ = ps_spline.get_point(t)

            ps_x_new[0:math.floor(shell_points*smooth)] = pt2[:,0]
            ps_y_new[0:math.floor(shell_points*smooth)] = pt2[:,1]

            # Smooth the shell at the trailing edge
            x = np.append(ss_x_new[(shell_points-math.floor(shell_points*smooth)):shell_points],np.flip(ps_x_new[(shell_points-math.floor(shell_points*smooth)):]))
            y = np.append(ss_y_new[(shell_points-math.floor(shell_points*smooth)):shell_points],np.flip(ps_y_new[(shell_points-math.floor(shell_points*smooth)):]))
            # Fit with bezier curve
            t = np.linspace(0,1,math.floor(shell_points*smooth)*2)
            b = bezier(x,y)

            [x,y] = b.get_point(t,equally_space_pts=False)
            d = dist(x,y,ss_x[-1],ss_y[-1])
            min_indx = np.where(d== np.amin(d))[0][0]
            ss_x_temp = x[0:min_indx+1]
            ss_y_temp = y[0:min_indx+1]
            ps_x_temp = x[min_indx:len(x)]
            ps_y_temp = y[min_indx:len(y)]

            ss_spline = pspline(ss_x_temp,ss_y_temp,method=spline_type.pchip)
            ps_spline = pspline(ps_x_temp,ps_y_temp,method=spline_type.pchip)
            t = np.linspace(0,1,int(len(x)/2))

            pt1,_ = ss_spline.get_point(t)

            ss_x_new[shell_points-math.floor(shell_points*smooth):shell_points] = pt1[:,0]
            ss_y_new[shell_points-math.floor(shell_points*smooth):shell_points] = pt1[:,1]

            pt2,_ = ps_spline.get_point(t)

            ps_x_new[(shell_points-math.floor(shell_points*smooth)):shell_points] = np.flip(pt2[:,0])
            ps_y_new[(shell_points-math.floor(shell_points*smooth)):shell_points] = np.flip(pt2[:,1])
        else:
            ps_x_new = np.zeros(shell_points); ps_y_new = np.zeros(shell_points); ps_z = np.zeros(shell_points)
            ss_x_new = np.zeros(shell_points); ss_y_new = np.zeros(shell_points); ss_z = np.zeros(shell_points)
            for i in range(shell_points):
                ps_x_new[i] = self.shell_xss[i](percent_span)
                ps_y_new[i] = self.shell_yss[i](percent_span)
                ps_z[i] = self.shell_zss[i](percent_span)

                ss_x_new[i] = self.shell_xps[i](percent_span)
                ss_y_new[i] = self.shell_yps[i](percent_span)
                ss_z[i] = self.shell_zps[i](percent_span)

        return ss_x_new,ss_y_new,ss_z,ps_x_new,ps_y_new,ps_z

    def rotate(self,cx:float,cy:float,angle:float=0):
        """Rotate all shifted profiles about the leading edge including control profiles

        Args:
            cx (float): rotation point x coordinate
            cy (float): rotation point y coordinate
            angle (float, optional): clockwise angle to rotate. Defaults to 0.
        """
        # Rotate each profile
        R = np.array([[cosd(angle), -sind(angle)],[sind(angle),cosd(angle)]])

        nprofiles = self.shft_ss.shape[0]
        for i in range(nprofiles):
            dx = self.shft_ss[i,:,0] - cx
            dy = self.shft_ss[i,:,1] - cy
            self.shft_ss[i,:,0] = (dx*cosd(angle) - dy*sind(angle)) + cx
            self.shft_ss[i,:,1] = (dx*sind(angle) + dy*cosd(angle)) + cy

            dx = self.shft_ps[i,:,0] - cx
            dy = self.shft_ps[i,:,1] - cy
            self.shft_ps[i,:,0] = (dx*cosd(angle) - dy*sind(angle)) + cx
            self.shft_ps[i,:,1] = (dx*sind(angle) + dy*cosd(angle)) + cy

        for i in range(self.control_ps.shape[0]):
            dx = self.control_ss[i,:,0] - cx
            dy = self.control_ss[i,:,1] - cy
            self.control_ss[i,:,0] = (dx*cosd(angle) - dy*sind(angle)) + cx
            self.control_ss[i,:,1] = (dx*sind(angle) + dy*cosd(angle)) + cy
            
            dx = self.control_ps[i,:,0] - cx
            dy = self.control_ps[i,:,1] - cy
            self.control_ps[i,:,0] = (dx*cosd(angle) - dy*sind(angle)) + cx
            self.control_ps[i,:,1] = (dx*sind(angle) + dy*cosd(angle)) + cy



    def center_le(self):
        """centers the blade by placing leading edge at 0,0
        """
        xc = self.shft_ss[0,0,0]
        yc = self.shft_ss[0,0,1]
        zc = self.shft_ss[0,0,2]

        self.shft_ss -= np.array([xc, yc, zc])
        self.shft_ps -= np.array([xc, yc, zc])

        self.control_ps -= np.array([xc, yc, zc])
        self.control_ss -= np.array([xc, yc, zc])
        
        self.stack_bezier.x = self.stack_bezier.x - xc
        self.stack_bezier.y = self.stack_bezier.y - yc
        self.stack_bezier.z = self.stack_bezier.z - zc

        self.te_center -= np.array([xc, yc, zc])
        self.zz = self.zz - zc

    def plot_shell_2D(self,percent_span:float,shell_thickness:float):
        """Plots the 2D shell used for placement of heat pipes

        Args:
            percent_span (float): percent span where shell occurs
            shell_thickness (float): thickness to remove from the blade

        Returns:
            (matplotlib.figure):  matplotlib figure object
        """

        [ss_x,ss_y,_,ps_x,ps_y,_,_,_] = self.get_cross_section(percent_span)
        ss_x_new,ss_y_new,ss_z_new,ps_x_new,ps_y_new,ps_z_new = self.get_shell_2D(percent_span,shell_thickness)

        fig,ax = plt.subplots()
        ax.plot(ss_x,ss_y,color='red',linestyle='solid',linewidth=2)
        ax.plot(ps_x,ps_y,color='blue',linestyle='solid',linewidth=2)

        ax.scatter(ss_x_new,ss_y_new,color='green',marker='.',s=0.5)
        ax.scatter(ps_x_new,ps_y_new,color='green',marker='.',s=0.5)
        ax.set_aspect('equal')
        return fig

    def get_circumference(self,ss_x:npt.NDArray,ss_y:npt.NDArray,ps_x:npt.NDArray,ps_y:npt.NDArray):
        """returns the circumferene of a 2D airfoil profile

        Args:
            ss_x (npt.NDArray): suction size x
            ss_y (npt.NDArray): suction side y
            ps_x (npt.NDArray): pressure side x
            ps_y (npt.NDArray): pressure side y

        Returns:
            (tuple): tuple containing:

            - **ss_len** (float): suction side length
            - **ps_len** (float): pressure side length
        """
        ss_len = np.sum(np.sqrt(np.diff(ss_x)*np.diff(ss_x) + np.diff(ss_y)*np.diff(ss_y)))
        ps_len = np.sum(np.sqrt(np.diff(ps_x)*np.diff(ps_x) + np.diff(ps_y)*np.diff(ps_y)))
        return ss_len,ps_len

    def export_stl(self,filename:str="blade.stl"):
        """Exports the finished blade to STL

        Args:
            filename (str, optional): Name of the STL file . Defaults to "blade.stl".
        """
        x = np.concatenate([self.shft_ss[:,:,0], np.flip(self.shft_ps[:,1:-1,0],axis=1)],axis=1)
        y = np.concatenate([self.shft_ss[:,:,1], np.flip(self.shft_ps[:,1:-1,1],axis=1)],axis=1)
        z = np.concatenate([self.shft_ss[:,:,2], np.flip(self.shft_ps[:,1:-1,2],axis=1)],axis=1)

        # Create triangles
        nspan = x.shape[0] # number of spans
        nv = x.shape[1]*nspan # number of verticies
        faces = list()

        # Bottom triangles
        i=0 # span 0
        for j in range(int(x.shape[1]/2)):
            v1 = j
            v2 = j+1
            v3 = x.shape[1] - (j+1)
            faces.append([v3,v2,v1])

            v1 = j + 1
            v2 = x.shape[1] - (j+2)
            v3 = x.shape[1] - (j+1)
            faces.append([v3,v2,v1])

        # Construct side wall triangles
        for i in range(1,x.shape[0]): # Spans
            for j in range(1,x.shape[1]): # points for each span
                # Blue triangle
                v1 = (i-1)*x.shape[1] + j-1
                v2 = (i-1)*x.shape[1] + j
                v3 = i    *x.shape[1] + j-1
                faces.append([v1,v2,v3])

                # Red triangle
                v1 = (i-1)*x.shape[1] + j
                v2 = i    *x.shape[1] + j
                v3 = i    *x.shape[1] + j-1
                faces.append([v1,v2,v3])

                if j == x.shape[1]-1:
                    # Blue triangle
                    #   v2
                    #   v3    v1
                    v1 = (i-1)*x.shape[1]
                    v2 = i    *x.shape[1] + j
                    v3 = i    *x.shape[1] - 1
                    faces.append([v1,v2,v3])

                    # Red triangle
                    #   v3   v2
                    #        v1
                    v1 = (i-1)*x.shape[1]
                    v2 =  i   *x.shape[1]
                    v3 =  i   *x.shape[1] + j
                    faces.append([v1,v2,v3])

        # Top triangles
        for j in range(int(x.shape[1]/2)):
            v1 = j
            v2 = j+1
            v3 = x.shape[1] - (j+1)
            faces.append([v1+(nspan-1)*x.shape[1],v2+(nspan-1)*x.shape[1],v3+(nspan-1)*x.shape[1]])

            v1 = j + 1
            v2 = x.shape[1] - (j+2)
            v3 = x.shape[1] - (j+1)
            faces.append([v1+(nspan-1)*x.shape[1],v2+(nspan-1)*x.shape[1],v3+(nspan-1)*x.shape[1]])


        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        vertices = np.array([x,y,z]).transpose()
        faces = np.array(faces)
        blade = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                blade.vectors[i][j] = vertices[f[j],:] # type: ignore


        blade.save(filename) # type: ignore
        
    def scale_z(self,hub:npt.NDArray,shroud:npt.NDArray):
        """Scale the airfoil in the z direction

        Args:
            hub (npt.NDArray): _description_
            shroud (npt.NDArray): _description_
        """
        def rescale_z(z: npt.NDArray, zmin: float, zmax: float) -> npt.NDArray:
            """
            Rescale the z-values of a (N,) array to a new [zmin, zmax] range.

            Parameters:
                points: (N,) array of (z) coordinates.
                zmin: New minimum for z-axis.
                zmax: New maximum for z-axis.

            Returns:
                (N, 3) array with rescaled z-values.
            """
            znew = z*0
            z_old_min, z_old_max = z.min(), z.max()
            
            # Normalize and scale to new range
            z_scaled = (z - z_old_min) / (z_old_max - z_old_min)
            znew = z_scaled * (zmax - zmin) + zmin
            
            return znew

        fhub_z_x = PchipInterpolator(hub[:,0],hub[:,1])
        fshroud_z_x = PchipInterpolator(shroud[:,0],shroud[:,1])
        
        for j in range(self.shft_ps.shape[1]):
            zhub = float(fhub_z_x(self.shft_ss[0,j,0]))
            zshroud = float(fshroud_z_x(self.shft_ss[-1,j,0]))
            self.shft_ss[:,j,2] = rescale_z(self.shft_ss[:,j,2],zhub,zshroud)
            
            zhub = float(fhub_z_x(self.shft_ps[-1,j,0]))
            zshroud = float(fshroud_z_x(self.shft_ps[-1,j,0]))
            self.shft_ps[:,j,2] = rescale_z(self.shft_ps[:,j,2],zhub,zshroud)
        
def import_geometry(folder:str,npoints:int=100,nspan:int=2,axial_chord:float=1,span:List[float]=[0,1],ss_ps_split:int=0) -> Airfoil3D:
    """imports geometry from a folder. Make sure there are 2 files inside the folder example: airfoil_0.txt and airfoil_1.txt. In this example, these two files represent the hub and tip. You can have as many as you want but 2 is the minimum. Filenames are sorted before import

    airfoil_0 can contain 2 columns x,y or 3 columns x,y,z. Z column isn't used because you set the span. The span determines the spanwise location of the two airfoils



    Args:
        folder ([str]): folder containing the airfoils
        npoints (int, optional): Number of points to scale the points to. Defaults to 100.
        nspan (int, optional): Number of spanwise profiles to create. Defaults to 2.
        axial_chord (int, optional): Defines the length of the axial chord. Axial chord within the file will be scaled. Defaults to 1.
        span (List[float], optional): Spanwise location of the profiles. Defaults to [0,1] for 2 profiles.
        ss_ps_split (int, optional): determines what index to split the suction and pressure side, default is number of points/2. Defaults to 0.

    Returns:
        (Airfoil3D): airfoil3D object
    """
    a3D = Airfoil3D([],[],0)
    def readFile(filename):
        with open(filename,'r') as fp:
            x = np.zeros(10000); y = np.zeros(10000); z = np.zeros(10000)
            indx = 0
            while (True):
                line = fp.readline()
                if not line.strip().startswith("#"):
                    line_no_comment = line.split("#")[0]
                    line_no_comment = line_no_comment.strip().split(' ')
                    arr = [float(l) for l in line_no_comment if l]
                    # arr = [s.strip() for s in line.splitlines()]
                    try:
                        if (len(arr) == 2):
                            arr = convert_to_ndarray(arr)
                            x[indx] = arr[0]
                            y[indx] = arr[1]
                            indx +=1
                        elif (len(arr) == 3):
                            arr = convert_to_ndarray(arr)
                            x[indx] = arr[0]
                            y[indx] = arr[1]
                            z[indx] = arr[2]
                            indx+=1
                    except Exception as test:
                        print(test)
                if not line:
                    break
            x = x[0:indx]
            y = y[0:indx]
            z = z[0:indx]
        return x,y,z
    
    pwd = os.getcwd()
    os.chdir(folder)
    listing = glob.glob('*.txt')

    nprofiles = len(listing)
    ss = np.zeros((nprofiles,npoints,3))
    ps = np.zeros((nprofiles,npoints,3))
    centroid = np.zeros((len(listing),3))

    for i in range(len(listing)):
        airfoil_file = listing[i]
        x,y,z = readFile(airfoil_file)
        xmin = x.min()
        xmax = x.max()
        scale = 1
        if (i==0):
            scale = axial_chord/(xmax-xmin)

        x = x*scale
        y = y*scale
        z = x*0+span[i]

        if (ss_ps_split<0):
            if ((len(x) % 2) == 0):
                te_indx = int(len(x)/2)
            else:
                te_indx = int((len(x)-1)/2)+1
        else:
            te_indx = ss_ps_split

        xps_temp = x[0:te_indx+1]
        yps_temp = y[0:te_indx+1]
        xss_temp = x[te_indx::]
        yss_temp = y[te_indx::]

        sp = pspline(xss_temp,yss_temp)
        pt,_ = sp.get_point(np.linspace(0,1,npoints))
        ss[i,:,0] = pt[:,0]; ss[i,:,1] = pt[:,1]

        sp = pspline(xps_temp,yps_temp)
        pt2,_ = sp.get_point(np.linspace(0,1,npoints))
        ps[i,:,0] = pt2[:,0]; ps[i,:,1] = pt2[:,1]

        ps[i,:,0] = np.flip(ps[i,:,0])
        ps[i,:,1] = np.flip(ps[i,:,1])

        # if (abs(yps[i,-1]-yss[i,0]) > abs(yps[i,-1]-yss[i,0])):
        #     yps[i,:] = np.flip(yps[i,:])
        #     xps[i,:] = np.flip(xps[i,:])
        centroid[i,2] = z[0]

        centroid[i,0] = np.sum((ss[i,:,0]+ps[i,:,0])/2)/npoints # Calculate and store the centroid
        centroid[i,1] = np.sum((ss[i,:,0]+ps[i,:,1])/2)/npoints

    a3D.stack_bezier_ctrl_pts = np.zeros((centroid.shape[0],3))
    a3D.stack_bezier_ctrl_pts = centroid
    
    a3D.stack_bezier = bezier3(centroid[:,0],centroid[:,1],centroid[:,2])
    t = np.linspace(0,1,nspan)
    [x,y,z] = a3D.stack_bezier.get_point(t,equally_space_pts=False)
    
    a3D.shft_ss = np.zeros((nspan,npoints,3))
    a3D.shft_ps = np.zeros((nspan,npoints,3))
    # populate the other varibles
    a3D.zz = z
    for i in range(npoints):
        a3D.shft_ss[:,i,0]= csapi(centroid[:,2],ss[:,i,0],z)
        a3D.shft_ss[:,i,1]= csapi(centroid[:,2],ss[:,i,1],z)
        a3D.shft_ps[:,i,0]= csapi(centroid[:,2],ps[:,i,0],z)
        a3D.shft_ps[:,i,1]= csapi(centroid[:,2],ps[:,i,1],z)
        a3D.shft_ss[:,i,2]= z
        a3D.shft_ps[:,i,2]= z

    a3D.control_ss = ss
    a3D.control_ps = ps 
    
    a3D.ss = copy.deepcopy(a3D.shft_ss)
    a3D.ps = copy.deepcopy(a3D.shft_ps)
    a3D.bImportedBlade = True
    a3D.stackType=StackType.centroid # Centroid
    a3D.span = max(z)-min(z)
    a3D.spanwise_spline_fit()
    a3D.nspan = nspan
    os.chdir(pwd)
    return a3D