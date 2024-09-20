from code import interact
import numpy as np
import math
from typing import List
from ..helper import convert_to_ndarray, bezier, bezier3, centroid, check_replace_max, check_replace_min, csapi
from ..helper import create_cubic_bounding_box, cosd, sind, uniqueXY, pspline, line2D, ray2D, pspline_intersect, dist, spline_type
from .airfoil2D import airfoil2D
from scipy.optimize import minimize_scalar
import enum
import copy
import os
import glob
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
from stl import mesh


class stack_type(enum.Enum):
    """class defining the type of stacking for airfoil2D profiles

    Args:
        enum (enum.Emum): inherits enum
    """
    leading_edge = 1
    centroid = 2
    trailing_edge = 3



class Airfoil3D():
    '''
        Properties
    '''
    profileArray: List[airfoil2D]
    profileSpan: List[float] 
    span:float 
    IsSplineFitted: bool
    IsSplineFittedShell: bool 
    
    stackType:stack_type 
    '''Bezier X,Y,Z are for the stacking line of the blade'''
    bezierX: List[float]    
    bezierY: List[float]
    bezierZ: List[float]

    te_center_x: np.ndarray     # Center of trailing edge for each span profile
    te_center_y: np.ndarray 
    b3: bezier3                 # 3D Bezier curve that defines the stacking
    bImportedBlade: bool

    sweepY: np.ndarray          # Array defining the sweep and lean points. 
    sweepZ: np.ndarray          # 

    leanX: np.ndarray
    leanY: np.ndarray

    nte: int                    # Number of points defining the trailing edge
    npts: int                   # Number of points defining the suction and pressure side
    nspan: int                  # Number of cross sections along the span
    
    '''Profile points: these are the 2D Profiles that you passed into the constructor'''
    spline_xpss: np.ndarray     # Profile Suction side 
    spline_ypss: np.ndarray
    spline_zpp: np.ndarray

    spline_xpps: np.ndarray     # Profile Pressure side
    spline_ypps: np.ndarray

    '''3D Blade points: Final blade points after lean and sweep are accounted for'''
    spline_xss: List[PchipInterpolator]      # Suction side list of curves
    spline_yss: List[PchipInterpolator]
    spline_zss: List[PchipInterpolator]

    spline_xps: List[PchipInterpolator]      # Pressure side: list of curves 
    spline_yps: List[PchipInterpolator]
    spline_zps: List[PchipInterpolator]

    spline_te_xss: np.ndarray   # Trailing edge suction side points
    spline_te_yss: np.ndarray

    spline_te_xps: np.ndarray   # Trailing edge pressure side points 
    spline_te_yps: np.ndarray

    xss: np.ndarray             # Blade cross sections defined without Lean or Sweep
    yss: np.ndarray             
    te_ss_x: np.ndarray
    te_ss_y: np.ndarray
    
    xps: np.ndarray
    yps: np.ndarray
    te_ps_x: np.ndarray
    te_ps_y: np.ndarray

    control_x_ss: np.ndarray    # Location of the stacking line control points
    control_y_ss: np.ndarray
    control_x_ps: np.ndarray    
    control_y_ps: np.ndarray
    c_te_x_ss: np.ndarray
    c_te_y_ss: np.ndarray
    c_te_x_ps: np.ndarray
    c_te_y_ps: np.ndarray

    shft_xss: np.ndarray        # Suction side Final points with lean and sweep
    shft_yss: np.ndarray        # These are the points that are exported
    shft_zss: np.ndarray

    shft_xps: np.ndarray        # Pressure side Final points with lean and sweep
    shft_yps: np.ndarray        # These are the points that are exported
    shft_zps: np.ndarray

    '''Shell'''    
    shell_xss: List[PchipInterpolator]  # curves defining the shelled geometry
    shell_yss: List[PchipInterpolator]  
    shell_zss: List[PchipInterpolator]

    shell_xps: List[PchipInterpolator]
    shell_yps: List[PchipInterpolator]
    shell_zps: List[PchipInterpolator] 

    """Defines a 3D Airfoil to be used in a channel
    """
    def __init__(self, profileArray: List[airfoil2D], profile_loc: List[float], height: float):
        """Constructs a 3D Airfoil

        Args:
            profileArray (List[airfoil2D]): array of airfoil2D profiles
            profile_loc (List[float]): location of the airfoil2D profiles
            height (float): height of the 3D blade
        """
        self.profileArray = profileArray
        self.profileSpan = convert_to_ndarray(profile_loc)
        self.span = height
        self.IsSplineFitted = False
        self.IsSplineFittedShell = False

    def stack(self, stackType=stack_type.centroid):
        """Defines how the airfoil profiles are stacked

        Args:
            stackType (stack_type, optional): Options are centroid, Leading Edge, Trailing Edge. Defaults to stack_type.centroid.
        """
        self.bezierX = []; self.bezierY = []; self.bezierZ = []
        self.stackType = stackType
        self.te_center_x = np.zeros(len(self.profileArray))
        self.te_center_y = np.zeros(len(self.profileArray))
        if (len(self.profileArray) >= 2):
            # Stack the airfoils about LE
            if (stackType == stack_type.leading_edge):
                hub = self.profileArray[0]
                [hx, hy] = hub.camberBezier.get_point(0)
                self.bezierX.append(hx[0])
                self.bezierY.append(hy[0])
                self.bezierZ.append(0)

                [hx_te, hy_te] = hub.camberBezier.get_point(1)
                self.te_center_x[0] = hx_te[0]
                self.te_center_y[0] = hy_te[0]

                for i in range(1, len(self.profileArray)):
                    [x, y] = self.profileArray[i].camberBezier.get_point(0)
                    dx = hx[0]-x[0]
                    dy = hy[0]-y[0]
                    # Shift the points based on camber
                    self.profileArray[i].shift(dx, dy)
                    self.bezierX.append(hx[0])
                    self.bezierY.append(hy[0])
                    self.bezierZ.append(self.profileSpan[i]*self.span)

                    [hx_te, hy_te] = self.profileArray[i].camberBezier.get_point(
                        1)
                    self.te_center_x[i] = hx_te[0]
                    self.te_center_y[i] = hy_te[0]
            # Stack the airfoils about TE
            elif (stackType == stack_type.trailing_edge):
                hub = self.profileArray[0]
                [hx, hy] = hub.camberBezier.get_point(1)
                self.bezierX.append(hx[0])
                self.bezierY.append(hy[0])
                self.bezierZ.append(0)

                [hx_te, hy_te] = hub.camberBezier.get_point(1)
                self.te_center_x[0] = hx_te[0]
                self.te_center_y[0] = hy_te[0]

                for i in range(1, len(self.profileArray)):
                    [x, y] = self.profileArray[i].camberBezier.get_point(1)
                    dx = 0
                    dy = 0
                    # Shift the points based on camber
                    self.profileArray[i].shift(dx, dy)
                    self.bezierX.append(hx[0])
                    self.bezierY.append(hy[0])
                    self.bezierZ.append(self.profileSpan[i]*self.span)

                    [hx_te, hy_te] = self.profileArray[i].camberBezier.get_point(
                        1)
                    self.te_center_x[i] = hx_te[0]
                    self.te_center_y[i] = hy_te[0]

            elif (stackType == stack_type.centroid):
                [hx, hy] = self.profileArray[0].get_centroid()
                self.bezierX.append(hx)
                self.bezierY.append(hy)
                self.bezierZ.append(0)

                [hx_te, hy_te] = self.profileArray[0].camberBezier.get_point(1)
                self.te_center_x[0] = hx_te[0]
                self.te_center_y[0] = hy_te[0]

                for i in range(1, len(self.profileArray)):
                    [x, y] = self.profileArray[i].get_centroid()
                    dx = hx-x
                    dy = hy-y
                    # Shift the points based on camber
                    self.profileArray[i].shift(dx, dy)
                    self.bezierX.append(hx)
                    self.bezierY.append(hy)
                    self.bezierZ.append(self.profileSpan[i]*self.span)

                    [hx_te, hy_te] = self.profileArray[i].camberBezier.get_point(
                        1)
                    self.te_center_x[i] = hx_te[0]
                    self.te_center_y[i] = hy_te[0]

            self.bezierX = convert_to_ndarray(self.bezierX)
            self.bezierY = convert_to_ndarray(self.bezierY)
            self.bezierZ = convert_to_ndarray(self.bezierZ)
            self.b3 = bezier3(self.bezierX,self.bezierY,self.bezierZ)
            self.bImportedBlade = False

    def sweep(self,sweep_y=[],sweep_z=[]):
        """Sweep bends the blade towards the leading edge or trailing edge. Blades are first stacked and then sweep can be applied

        Args:
            sweep_y (List[float], optional): adds bezier points to the sweep of the blade. Defaults to [].
            sweep_z (List[float], optional): defines where along the span the sweep points should be located. Defaults to [].
        """
        self.sweepY = convert_to_ndarray(sweep_y)
        self.sweepZ = convert_to_ndarray(sweep_z)

        # add sweep where z exists
        self.sweepZ = self.sweepZ*self.span
        [_,ind_com1,ind_com2] = np.intersect1d(self.bezierZ, self.sweepZ,return_indices=True)        
        self.bezierY[ind_com1] = self.bezierY[ind_com1] + self.sweepY[ind_com2]*self.span


        # add sweep where z does not exist 
        i1 = np.setxor1d(self.sweepZ, self.bezierZ) # tells what items in leanZ do not exist in bezierZ
        if i1.size >0:
            i1 = np.where(self.sweepZ == i1) # get the index
            self.bezierZ = np.append(self.bezierZ, self.sweepZ[i1])
            self.bezierY = np.append(self.bezierY, self.sweepY[i1]*self.span+self.bezierY[0])
            xx = self.sweepZ; xx[i1] = self.bezierX[0]
            self.bezierX = np.append(self.bezierX, xx[i1])

        # SORT
        # A = sortrows([self.bezierZ' self.bezierX' self.bezierY'])
        indx = np.argsort(self.bezierZ)
        self.bezierZ = self.bezierZ[indx]
        self.bezierX = self.bezierX[indx]
        self.bezierY = self.bezierY[indx]

        if (self.bImportedBlade): # imported blade doesn't have 2D airfoil profiles defined, just points
            self.profiles_shift()

    def lean(self,leanX:List[float],leanZ:List[float]):
        """Leans the blade towards the suction or pressure side. This applies points that are fitted by a bezier curve. Profiles are adjusted to follow this curve simulating lean.

        Args:
            leanX (List[float]): lean points 
            leanZ (List[float]): spanwise location of the lean points
        """
        leanX = convert_to_ndarray(leanX)
        leanZ = convert_to_ndarray(leanZ)

        self.leanX = leanX
        self.leanZ = leanZ
        leanZ = leanZ*self.span

        # Add lean where z exists, basically adds more lean 
        [_,ind_com1,ind_com2] = np.intersect1d(self.bezierZ,leanZ,return_indices=True)
        self.bezierX[ind_com1] = self.bezierX[ind_com1]+leanX[ind_com2]*self.span

        #  Add Lean where Z does not exist
        # [~,i1] = setxor(leanZ,self.bezierZ); %  MATLAB CODE tells what items in leanZ do not exist in bezierZ
        i1 = np.setxor1d(leanZ,self.bezierZ)
        if i1.size >0:
            i1 = np.where(leanZ == i1) # get the index
            self.bezierZ = np.append(self.bezierZ, leanZ[i1])
            self.bezierX = np.append(self.bezierX, self.leanX[i1]*self.span+self.bezierX[0])
            yy = leanZ; yy[i1] = self.bezierY[0] #TODO Need to check this
            self.bezierY = np.append(self.bezierY, yy[i1])

        # Sort
        indx = np.argsort(self.bezierZ)
        self.bezierZ = self.bezierZ[indx]
        self.bezierX = self.bezierX[indx]
        self.bezierY = self.bezierY[indx]
        
        if (self.bImportedBlade):
            self.profiles_shift()     
        
    def create_blade(self,nProfiles:int,num_points:int,trailing_edge_points:int):
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
        t = np.linspace(0,1,self.npts)
        t_te = np.linspace(0,1,self.nte) # use 20 points for the trailing edge
        n_profiles = len(self.profileArray)
        # x,y,z profiles - stores the x,y,z coordinate for each profile for a given time
        self.spline_xpps = np.zeros((self.npts,n_profiles))
        self.spline_ypps = np.zeros((self.npts,n_profiles))
        self.spline_zpp = np.zeros((self.npts,n_profiles))
        self.spline_xpss = np.zeros((self.npts,n_profiles))
        self.spline_ypss = np.zeros((self.npts,n_profiles)) 
        self.spline_te_xss = np.zeros((self.nte,n_profiles)) # trailing edge points fixed at 100
        self.spline_te_yss = np.zeros((self.nte,n_profiles))
        self.spline_te_xps = np.zeros((self.nte,n_profiles))
        self.spline_te_yps = np.zeros((self.nte,n_profiles))
        # z coordinates of the blade 
        # --- Initialize and clear the profile points ---
        self.zz = np.linspace(0,self.span,self.nspan)  # 0 to whatever the span.          
        self.xps = np.zeros((self.nspan,self.npts)) # [x,x,x,x] -> each vector of x represents a different z coordinate
        self.yps = np.zeros((self.nspan,self.npts)) # [y,y,y,y] -> each vector of y represents a different z coordinate
        self.xss = np.zeros((self.nspan,self.npts)) # suction side
        self.yss = np.zeros((self.nspan,self.npts))
        self.te_ss_x = np.zeros((self.nspan,self.nte))
        self.te_ss_y = np.zeros((self.nspan,self.nte))
        self.te_ps_x = np.zeros((self.nspan,self.nte))
        self.te_ps_y = np.zeros((self.nspan,self.nte))

        # --- Make a spline for each profile for each point
        # new code below
        # tmpXps = np.zeros((self.npts,n_profiles)); tmpYps = np.zeros((self.npts,n_profiles))
        # tmpXss = np.zeros((self.npts,n_profiles)); tmpYss = np.zeros((self.npts,n_profiles))
        for j in range(n_profiles):
            # [tmpXps[:,j], tmpYps[:,j]] = self.profileArray[j]._psBezier.get_point(t)
            # [tmpXss[:,j], tmpYss[:,j]] = self.profileArray[j]._ssBezier.get_point(t)
            [self.spline_xpps[:,j], self.spline_ypps[:,j]] = self.profileArray[j].psBezier.get_point(t,equal_space=True)
            [self.spline_xpss[:,j], self.spline_ypss[:,j]] = self.profileArray[j].ssBezier.get_point(t,equal_space=True)
            self.spline_zpp[:,j] = self.profileSpan[j]*self.span              # Span
        
            
        for i in range(self.npts): 
            self.xps[:,i]= csapi(self.spline_zpp[i,:],self.spline_xpps[i,:],self.zz) # Natural spline
            self.yps[:,i]= csapi(self.spline_zpp[i,:],self.spline_ypps[i,:],self.zz)
            
            self.xss[:,i]= csapi(self.spline_zpp[i,:],self.spline_xpss[i,:],self.zz)
            self.yss[:,i]= csapi(self.spline_zpp[i,:],self.spline_ypss[i,:],self.zz)
        

        spline_zpp_te = np.zeros((self.nte,n_profiles))
        for j in range(n_profiles): # Trailing edge contains less points 
            # trailing edge                    
            [self.spline_te_xps[:,j],self.spline_te_yps[:,j]] = self.profileArray[j].TE_ps_arc.get_point(t_te)
            [self.spline_te_xps[:,j],self.spline_te_yps[:,j]] = self.profileArray[j].TE_ps_arc.get_point(t_te)
            [self.spline_te_xss[:,j],self.spline_te_yss[:,j]] = self.profileArray[j].TE_ss_arc.get_point(t_te)
            [self.spline_te_xss[:,j],self.spline_te_yss[:,j]] = self.profileArray[j].TE_ss_arc.get_point(t_te)
            spline_zpp_te[:,j] = self.profileSpan[j]*self.span 
            
        for i in range(self.nte):   
            # Trailing edge suction side
            self.te_ss_x[:,i]= csapi(spline_zpp_te[i,:],self.spline_te_xss[i,:],self.zz)
            self.te_ss_y[:,i]= csapi(spline_zpp_te[i,:],self.spline_te_yss[i,:],self.zz)
            # Trailing edge pressure side
            self.te_ps_x[:,i]= csapi(spline_zpp_te[i,:],self.spline_te_xps[i,:],self.zz)
            self.te_ps_y[:,i]= csapi(spline_zpp_te[i,:],self.spline_te_yps[i,:],self.zz)             
        
        self.te_center_x = csapi(self.profileSpan*self.span,self.te_center_x,self.zz)
        self.te_center_y = csapi(self.profileSpan*self.span,self.te_center_y,self.zz)

        # Populate Control Points 
        self.control_x_ps = np.zeros((self.npts,n_profiles))
        self.control_y_ps = np.zeros((self.npts,n_profiles))
        self.control_x_ss = np.zeros((self.npts,n_profiles))
        self.control_y_ss = np.zeros((self.npts,n_profiles))
        self.c_te_x_ps = np.zeros((self.nte,n_profiles))
        self.c_te_x_ss = np.zeros((self.nte,n_profiles))
        self.c_te_y_ps = np.zeros((self.nte,n_profiles))
        self.c_te_y_ss = np.zeros((self.nte,n_profiles))
        
        for i in range(n_profiles):                
            [self.control_x_ps[:,i], self.control_y_ps[:,i]] = self.profileArray[i].psBezier.get_point(t)
            [self.control_x_ss[:,i], self.control_y_ss[:,i]] = self.profileArray[i].ssBezier.get_point(t)
            # add trailing edge
            [self.c_te_x_ps[:,i], self.c_te_y_ps[:,i]] = self.profileArray[i].TE_ps_arc.get_point(t_te)
            [self.c_te_x_ss[:,i], self.c_te_y_ss[:,i]] = self.profileArray[i].TE_ss_arc.get_point(t_te)

        # Shift all the generated turbine profiles points based on the bezier curve
        self.profiles_shift()
 
    def shift(self,x:float,y:float):
        """Moves the blade and recomputes the geometry
            Step 1 - shifts the profiles
            Step 2 - add lean and sweep again
            Step 3 - recompute the geometry if npts != 0 

        Args:
            x (float): shift in x direction
            y (float): shift in y directrion
        """
        [nprofile,_] = self.shft_xss.shape            
        for i in range(nprofile):          
            self.shft_yss[i,:] = self.shft_yss[i,:] + y
            self.shft_xss[i,:] = self.shft_xss[i,:] + x
            self.shft_yps[i,:] = self.shft_yps[i,:] + y
            self.shft_xps[i,:] = self.shft_xps[i,:] + x
 
    def scale_zss(self,zmin:List[float],zmax:List[float]):
        """scales the z axis to match the channel height. Channel height may be defined as the radius

        Args:
            zmin (List[float]): Array of size n corresponding to minimum radius
            zmax (List[float]): Array of size n corresponding to maximum radius
        """
        
        [_,nzz] = self.shft_xss.shape

        z = np.zeros((self.nspan,len(zmin)))
        for i in range(len(zmin)):
            z[:,i] = np.linspace(zmin[i],zmax[i],self.nspan)      # Rows (Z) Columns (Each point)          
            
        self.shft_zss = self.xss  
        for i in range(nzz): # number of profile sections created from bezier curve + spline
            self.shft_zss[:,i] = z[:,i]                
        
        if (self.IsSplineFittedShell):
            for i in range(nzz): # number of profile sections created from bezier curve + spline
                self.shell_zss[:,i] = z[:,i]
    
    def scale_zps(self,zmin:List[float],zmax:List[float]):
        """scales the z axis to match the channel height. Channel height may be defined as the radius

        Args:
            zmin (List[float]): Array of size n corresponding to minimum radius
            zmax (List[float]): Array of size n corresponding to maximum radius
        """
        [_,nzz] = self.shft_xps.shape

        z = np.zeros((self.nspan,len(zmin)))
        for i in range(len(zmin)):
            z[:,i] = np.linspace(zmin[i],zmax[i],self.nspan)      # Rows (Z) Columns (Each point)          
            
        self.shft_zps = self.xps  
        for i in range(nzz): # number of profile sections created from bezier curve + spline
            self.shft_zps[:,i] = z[:,i]            
        
        if (self.IsSplineFittedShell):
            for i in range(nzz): # number of profile sections created from bezier curve + spline
                self.shell_zps[:,i] = z[:,i]
    
    def flip_cw(self):
        """Mirrors the blade by multiplying -1*x direction. This is assuming axial chord is in the y direction and span is in z
        """
        self.shft_xps = -1*self.shft_xps
        self.shft_xss = -1*self.shft_xss

    def flip(self):
        """Mirrors the blade by multiplying y direction by -1. This is assuming axial chord is in the y direction and span is in z
        """
        self.shft_yps = -1*self.shft_yps
        self.shft_yss = -1*self.shft_yss

    def section_z(self,zStartPercent:float,zEndPercent:float):  
        """Chops the blade in between 2 spanwise lines. Think of it as cutting the blade between (zStartPercent) 10% and (zEndPercent) 50%


        Args:
            zStartPercent (float): bottom % of the blade to cut from
            zEndPercent (float): top % of blade to cut from. stuff in middle is saved
        """
        [_,npts] = self.shft_zss.shape
        for i in range(npts):
            # create a spline for each profile's shift points 
            mn_zss = min(self.shft_zss[:,i])
            mn_zps = min(self.shft_zps[:,i])
            mx_zss = max(self.shft_zss[:,i])
            mx_zps = max(self.shft_zps[:,i])
            h_ss = mx_zss-mn_zss # height suction side
            h_ps = mx_zps-mn_zps 
            zss = np.linspace(mn_zss+h_ss*zStartPercent,mn_zss+h_ss*zEndPercent,self.npts)
            
            zps = np.linspace(mn_zps+h_ps*zStartPercent,mn_zps+h_ps*zEndPercent,self.npts)
            
            self.shft_yss[:,i]= csapi(self.shft_zss[:,i],self.shft_yss[:,i],zss)
            self.shft_xss[:,i]= csapi(self.shft_zss[:,i],self.shft_xss[:,i],zss)
            self.shft_yps[:,i]= csapi(self.shft_zps[:,i],self.shft_yps[:,i],zps)
            self.shft_xps[:,i]= csapi(self.shft_zps[:,i],self.shft_xps[:,i],zps)
            self.shft_zps[:,i]= zps
            self.shft_zss[:,i]= zss
        
        # self.cylindrical()
      
    def cartesian(self):
        """
            Converts the default cylindrical coordinates to cartesian system   
        """
        [nprofiles,_] = self.shft_xss.shape
        for i in range(nprofiles): # for each 2d blade profile in the 3d blade
            [self.shft_xss[i,:],self.shft_zss[i,:],_] = self.convert_cyl_cartesian(self.shft_xss[i,:],self.shft_zss[i,:])
            [self.shft_xps[i,:],self.shft_zps[i,:],_] = self.convert_cyl_cartesian(self.shft_xps[i,:],self.shft_zps[i,:])      
         
    def plot_profile(self,figureNum):
        """Incomplete

        Args:
            figureNum ([type]): [description]
        """
        n2D = 100          
        nprofiles = len(self.profileArray)
        PSx = np.zeros(n2D*2,nprofiles)
        PSy = np.zeros(n2D*2,nprofiles)
        SSx = PSx; SSy = PSy; z = np.zeros(n2D*2,nprofiles)
        # figure(figureNum)
        # hold on
        # for i = 1:nprofiles
        #     PSx[:,i]= self.shft_control_x_ps[:,i]; PSy[:,i]= self.shft_control_y_ps[:,i];
        #     SSx[:,i]= self.shft_control_x_ss[:,i]; SSy[:,i]= self.shft_control_y_ss[:,i];
        #     z[:,i]= ones(n2D*2,1)*self.profileSpan(i)*self.span;
        #     plot3(PSx[:,i],PSy[:,i],z[:,i],'r','linewidth',1.5);       
        #     plot3(SSx[:,i],SSy[:,i],z[:,i],'b','linewidth',1.5);   
        # end
        # plot3(self.spineX,self.spineY,self.zz,'k','Linewidth',1.5);
        # hold off
        # axis equal            
    
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
        [n,_] = self.shft_xss.shape; # n - number of points, m - number of sections
        for j in range(n):
            x = np.append(self.shft_xss[j,:], np.flip(self.shft_xps[j,:]))
            y = np.append(self.shft_yss[j,:], np.flip(self.shft_yps[j,:]))
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
            if (self.b3): # Plot the spline 
                [bx,by,bz] = self.b3.get_point(np.linspace(0,1,num=50),equal_space=False)
                ax.plot3D(bx, by, bz, 'gray')
        
            # Plot the control profiles 
            if (not self.bImportedBlade):
                [_,nprofiles] = self.control_x_ps.shape
                for p in range(nprofiles):
                    ax.plot3D(self.control_x_ps[:,p], self.control_y_ps[:,p], self.control_z_ps[:,p],color='green')
                    ax.plot3D(self.control_x_ss[:,p], self.control_y_ss[:,p], self.control_z_ss[:,p],color='green')
                # Plot trailing edge center
                ax.plot3D(self.te_center_x,self.te_center_y,self.zz,color='black')

                # Plot the spine
                [bx,by,bz] = self.b3.get_point(np.linspace(0,1,nprofiles),equal_space=False)
                ax.plot3D(bx,by,bz,color='black')

        # Plot the profiles
        [nprofiles,_] = self.shft_xss.shape
        xmax=0.0; ymax=0.0; zmax=0.0
        xmin=0.0; ymin=0.0; zmin=0.0
        for i in range(nprofiles):
            ax.plot3D(self.shft_xss[i,:],self.shft_yss[i,:],self.shft_zss[i,:],color='red')
            ax.plot3D(self.shft_xps[i,:],self.shft_yps[i,:],self.shft_zps[i,:],color='blue')
            xmax = check_replace_max(xmax,np.max(np.append(self.shft_xps[i,:],self.shft_xss[i,:])))
            xmin = check_replace_min(xmin,np.min(np.append(self.shft_xps[i,:],self.shft_xss[i,:])))

            ymax = check_replace_max(ymax,np.max(np.append(self.shft_yps[i,:],self.shft_yss[i,:])))
            ymin = check_replace_min(ymin,np.min(np.append(self.shft_yps[i,:],self.shft_yss[i,:])))
            
            zmax = check_replace_max(zmax,np.max(np.append(self.shft_zps[i,:],self.shft_zss[i,:])))
            zmin = check_replace_min(zmin,np.min(np.append(self.shft_zps[i,:],self.shft_zss[i,:])))

        
        Xb,Yb,Zb = create_cubic_bounding_box(xmax,xmin,ymax,ymin,zmax,zmin)

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        plt.show()


        '''
            Creates a 3D plot of the blade using plotly
            Trailing edge center line is also plotted along with the blade's stacking spine
        '''
    
    def plot3D_ly(self,only_blade=False):
        """Plots a 3D representation of the blade and control points trailing edge center line is also plotted along with the blade's stacking spine

        Args:
            only_blade (bool, optional): Only plot the blade, no stacking spine. Defaults to False.

        """

        import plotly.graph_objects as go
        # df = px.data.gapminder().query("continent=='Europe'")
        # fig = px.line_3d(df, x="gdpPercap", y="pop", z="year", color='country')
        # fig.show()

        marker=dict(size=0.001, color="red", colorscale='Viridis')
        line=dict(color='green',width=2)
        # Plot the profiles
        [nprofiles,_] = self.shft_xss.shape
        for i in range(nprofiles):
            if i == 0:
                fig = go.Figure(data=go.Scatter3d(x=self.shft_xss[i,:], y=self.shft_yss[i,:], z=self.shft_zss[i,:],  marker=marker,line=dict(color='red',width=2)))
            else:
                fig.add_trace(go.Scatter3d(x=self.shft_xss[i,:], y=self.shft_yss[i,:], z=self.shft_zss[i,:],  marker=marker,line=dict(color='red',width=2)))
                fig.add_trace(go.Scatter3d(x=self.shft_xps[i,:], y=self.shft_yps[i,:], z=self.shft_zps[i,:],  marker=marker,line=dict(color='blue',width=2)))

        if (not self.bImportedBlade):
            if (not only_blade):
                [_,nprofiles] = self.control_x_ps.shape
                for p in range(nprofiles):
                    fig.add_trace(go.Scatter3d(x=self.control_x_ps[:,p], y=self.control_y_ps[:,p], z=self.control_z_ps[:,p], marker=marker,line=line))
                    fig.add_trace(go.Scatter3d(x=self.control_x_ss[:,p], y=self.control_y_ss[:,p], z=self.control_z_ss[:,p], marker=marker,line=line))
                
                # Plot trailing edge center
                fig.add_trace(go.Scatter3d(x=self.te_center_x, y=self.te_center_y, z=self.zz,  marker=marker,line=line))
                # Plot the spine
                [bx,by,bz] = self.b3.get_point(np.linspace(0,1,nprofiles))
                fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz,  marker=marker,line=dict(color='black',width=2)))

        fig.update_layout(showlegend=False,scene= dict(aspectmode='manual',aspectratio=dict(x=1, y=1, z=1)))
        fig.show()

    def calc_nblades(self,pitchChord:float,rhub:float):   
        """Calculates the number of blades 

        Args:
            pitchChord (float): pitch to chord ratio
            rhub (float): hub radius
        """
        pitch = self.profileArray[0].c*pitchChord
        return math.floor(2*math.pi*rhub/pitch)

    
    @staticmethod
    def import_geometry(folder:str,npoints:int=100,nspan:int=2,axial_chord:float=1,span:List[float]=[0,1],ss_ps_split:int=0):
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
            (airfoil3D): airfoil3D object
        """
        a3D = airfoil3D([],[],0)
        def readFile(filename):
            with open(filename,'r') as fp:
                x = np.zeros(10000); y = np.zeros(10000); z = np.zeros(10000)
                indx = 0
                while (True): 
                    line = fp.readline()
                    if not line.strip().startswith("#"):
                        line_no_comment = line.split("#")[0]
                        line_no_comment = line_no_comment.strip().split(' ')
                        arr = [l for l in line_no_comment if l]                    
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
        xss = np.zeros((nprofiles,npoints))
        yss = np.zeros((nprofiles,npoints))
        xps = np.zeros((nprofiles,npoints))
        yps = np.zeros((nprofiles,npoints))
        zz = np.zeros(nprofiles)
        cx = np.zeros(len(listing))
        cy = np.zeros(len(listing))
        cz = np.zeros(len(listing))

        for i in range(len(listing)):
            airfoil_file = listing[i]
            [x,y,z] = readFile(airfoil_file)
            xmin = min(x)
            xmax = max(x)
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
            xss[i,:] = pt[:,0]; yss[i,:] = pt[:,1]
            
            sp = pspline(xps_temp,yps_temp)
            pt2,_ = sp.get_point(np.linspace(0,1,npoints))
            xps[i,:] = pt2[:,0]; yps[i,:] = pt2[:,1]

            xps[i,:] = np.flip(xps[i,:])
            yps[i,:] = np.flip(yps[i,:])

            # if (abs(yps[i,-1]-yss[i,0]) > abs(yps[i,-1]-yss[i,0])):
            #     yps[i,:] = np.flip(yps[i,:])
            #     xps[i,:] = np.flip(xps[i,:])
            zz[i] = z[0]
            
            cx[i] = np.sum((xss[i,:]+xps[i,:])/2)/npoints # Calculate and store the centroid
            cy[i] = np.sum((yss[i,:]+yps[i,:])/2)/npoints
        
        a3D.bx =cx[0]; a3D.by = cy[0]
        a3D.bezierX = cx
        a3D.bezierY = cy
        a3D.bezierZ = zz
        
        cz = zz
        a3D.b3 = bezier3(cx,cy,cz)
        t = np.linspace(0,1,nspan)
        [x,y,z] = a3D.b3.get_point(t,equal_space=False)
        # populate the other varibles 
        a3D.shft_xss = np.zeros((nspan,npoints))
        a3D.shft_yss = np.zeros((nspan,npoints))
        a3D.shft_xps = np.zeros((nspan,npoints))
        a3D.shft_yps = np.zeros((nspan,npoints))
        a3D.shft_zss = np.zeros((nspan,npoints))
        a3D.shft_zps = np.zeros((nspan,npoints))
        a3D.zz = z
        for i in range(npoints):
            a3D.shft_xss[:,i]= csapi(zz,xss[:,i],z)
            a3D.shft_yss[:,i]= csapi(zz,yss[:,i],z)
            a3D.shft_xps[:,i]= csapi(zz,xps[:,i],z)
            a3D.shft_yps[:,i]= csapi(zz,yps[:,i],z)
            a3D.shft_zss[:,i]= z
            a3D.shft_zps[:,i]= z
        
        a3D.control_x_ss = xss
        a3D.control_y_ss = yss
        a3D.control_x_ps = xps
        a3D.control_y_ps = yps
        a3D.xss = copy.deepcopy(a3D.shft_xss)
        a3D.yss = copy.deepcopy(a3D.shft_yss)
        a3D.xps = copy.deepcopy(a3D.shft_xps)
        a3D.yps = copy.deepcopy(a3D.shft_yps)
        a3D.bImportedBlade = True
        a3D.stackType=2 # Centroid
        a3D.span = max(z)-min(z)
        a3D.spanwise_spline_fit()
        a3D.nspan = nspan
        os.chdir(pwd)
        return a3D
    
    def get_chord(self):
        """Returns the chord, axial chord for all the profiles 
        """
        chord = np.sqrt((self.shft_xps[:,-1] - self.shft_xps[:,0])**2 + (self.shft_yps[:,-1] - self.shft_yps[:,0])**2)
        axial_chord = abs(self.shft_xps[:,-1] - self.shft_xps[:,0])
        max_chord = max(chord)
        avg_chord = np.mean(chord)

        max_axial_chord = max(axial_chord)
        avg_axial_chord = np.mean(axial_chord)
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
        r = self.shft_zps[:,-1]
        s = 2*math.pi*r/nBlades
        s_c = s/chord
        return s, s_c

    def profiles_shift(self):
        """
            Shift all profiles based on bezier curve that describes the spline of the blade
        """
        self.b3 = bezier3(self.bezierX,self.bezierY,self.bezierZ)
        # Get centroid before combining with TE
        [n,_] = self.xps.shape
        cx = np.zeros(n); cy = np.zeros(n)
        for i in range(0,n):
            [cx[i], cy[i]] = centroid(np.concatenate((self.xps[i,:],self.xss[i,:])),np.concatenate((self.yps[i,:],self.yss[i,:])))
            
        # Combine with TE
        self.yps = np.concatenate((self.yps, self.te_ps_y[:,1:]),axis=1)
        self.xps = np.concatenate((self.xps, self.te_ps_x[:,1:]),axis=1)
        self.yss = np.concatenate((self.yss, self.te_ss_y[:,1:]),axis=1)
        self.xss = np.concatenate((self.xss, self.te_ss_x[:,1:]),axis=1)   

        # Shift all points by bezier curve        
        self.shft_xps = copy.deepcopy(self.xps) # Add the trailing edge 
        self.shft_yps = copy.deepcopy(self.yps)
        self.shft_zps = copy.deepcopy(self.yps)

        self.shft_xss = copy.deepcopy(self.xss)
        self.shft_yss = copy.deepcopy(self.yss)
        self.shft_zss = copy.deepcopy(self.yss)

        [nprofiles,npoints] = self.xps.shape

        self.spineX = np.zeros(nprofiles)
        self.spineY = np.zeros(nprofiles)
        self.spineZ = copy.deepcopy(self.zz)
        t = np.linspace(0,1,nprofiles)   
        [bx,by,_] = self.b3.get_point(t,equal_space=False)

        for i in range(0,nprofiles):
            x = bx[i]; y = by[i]
            if (self.stackType == stack_type.centroid):
                sx = cx[i]; sy = cy[i]
            elif (self.stackType == stack_type.leading_edge):
                sx = self.xps[i,0]
                sy = self.yps[i,0]
            else: # (self.stackType == stack_type.trailing_edge)
                sx = 0
                sy = 0
            
            # Pressure profiles
            self.shft_xps[i,:] = self.xps[i,:] + x - sx
            self.shft_yps[i,:] = self.yps[i,:] + y - sy
            # Suction profiles                
            self.shft_xss[i,:] = self.xss[i,:] + x - sx
            self.shft_yss[i,:] = self.yss[i,:] + y - sy

            self.spineX[i] = self.xps[i,0] + x - sx
            self.spineY[i] = self.yps[i,0] + y - sy
            
            if (len(self.te_center_x)>0): # if self is an imported blade then te_center wont be defined. 
                self.te_center_x[i] = self.te_center_x[i] + x - sx   # Shift the trailing edge center
                self.te_center_y[i] = self.te_center_y[i] + y - sy
            
            self.shft_zps[i,:] = self.zz[i]
            self.shft_zss[i,:] = self.zz[i]

        # Equal Space points 
        t2 = np.linspace(0,1,npoints)
        for i in trange(nprofiles,desc='Equal Spacing'):
            p = pspline(self.shft_xps[i,:],self.shft_yps[i,:])
            xy,_ = p.get_point(t2)
            self.shft_xps[i,:] = xy[:,0]
            self.shft_yps[i,:] = xy[:,1]

            p = pspline(self.shft_xss[i,:],self.shft_yss[i,:])
            xy,_ = p.get_point(t2)
            self.shft_xss[i,:] = xy[:,0]
            self.shft_yss[i,:] = xy[:,1]
            # fig,ax = plt.subplots()
            # ax.plot(self.shft_xps[i,:],self.shft_yps[i,:],'.')
            # ax.plot(self.shft_xss[i,:],self.shft_yss[i,:],'.')
            # ax.set_aspect('equal')
            # plt.show()
        
        
        # Shift Control Profiles
        self.control_x_ps = np.concatenate((self.control_x_ps, self.c_te_x_ps),axis=0)
        self.control_y_ps = np.concatenate((self.control_y_ps, self.c_te_y_ps),axis=0)
        self.control_x_ss = np.concatenate((self.control_x_ss, self.c_te_x_ss),axis=0)
        self.control_y_ss = np.concatenate((self.control_y_ss, self.c_te_y_ss),axis=0)

        self.control_z_ps = copy.deepcopy(self.control_x_ps)*0
        self.control_z_ss = copy.deepcopy(self.control_x_ss)*0
        for i in range(len(self.profileSpan)):
            self.control_z_ps[:,i] = self.control_z_ps[:,i] + self.profileSpan[i]*self.span
            self.control_z_ss[:,i] = self.control_z_ss[:,i] + self.profileSpan[i]*self.span

    def convert_cyl_cartesian(self,rth:np.ndarray,radius:np.ndarray):
        """Convert a single profile from cylindrical to cartesian coordinates. 

        Args:
            rth (np.ndarray): points in rtheta coordinate system
            radius (np.ndarray): radius values of those points 

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

    def __check_camber_intersection__(self,ray,camber_x,camber_y):
        bIntersect = False
        for p in range(len(camber_x)-1): # check if ray intersects with
            camber_line = line2D([camber_x[p],camber_y[p]],[camber_x[p+1],camber_y[p+1]])
            [t,bIntersect] = camber_line.intersect_ray(ray)
            if (bIntersect):
                if (t==0 and (ray.x == camber_line.p[0] and ray.y == camber_line.p[1])): # if ray starting point is the same as line, don't count it 
                    bIntersect = False
                elif (t<0): # if ray time vector is negative then it doesn't intersect
                    bIntersect = False
                else:
                    break
        return bIntersect
    
    def __check_ss_ray_intersection__(self,ray,ss_x,ss_y):
        """
            checks to see if the ray intersects the suction side
        """
        bIntersect = False
        for p in range(0,len(ss_x)-1): # check if ray intersects with
            ss_line = line2D([ss_x[p], ss_y[p]],[ss_x[p+1], ss_y[p+1]])
            [t,u,bIntersect] = ss_line.intersect_ray(ray)
            if (bIntersect):
                if (t==0 and (ray.x == ss_line.p[0]) and ray.y == ss_line.p[1]): # if ray starting point is the same as line, don't count it 
                    bIntersect = False
                elif (u<0): # if ray time vector is negative then it doesn't intersect
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
            ps_line = line2D([ps_x[p],ps_y[p]],[ps_x[p+1], ps_y[p+1]])
            [t,u,bIntersect] = ps_line.intersect_ray(ray)
            if (bIntersect):
                if (t==0 and (ray.x == ps_line.p[0] and ray.y == ps_line.p[1])): # if ray starting point is the same as line, don't count it 
                    bIntersect = False
                elif (u<0): # if ray time vector is negative then it doesn't intersect
                    bIntersect = False
                else:
                    break
        return bIntersect

    def get_cross_section_normal(self,ss_x:np.ndarray,ss_y:np.ndarray,ps_x:np.ndarray,ps_y:np.ndarray):
        """Gets the normal of a cross section for a given profile

        Args:
            ss_x (np.ndarray): x - suction side points of a given profile
            ss_y (np.ndarray): y - suction side points of a given profile
            ps_x (np.ndarray): x - pressure side points of a given profile
            ps_y (np.ndarray): y - pressure side points 

        Returns:
            (tuple): tuple containing:

            - **ss_nx** (np.ndarray): normal vector on suction side
            - **ss_ny** (np.ndarray): normal vector on suction side
            - **ps_nx** (np.ndarray): normal vector on pressure side
            - **ps_ny** (np.ndarray): normal vector on pressure side
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
        [nprofiles,npts] = self.shft_xps.shape
        self.spline_xps = []
        self.spline_yps = []
        self.spline_zps = []

        self.spline_xss = []
        self.spline_yss = []
        self.spline_zss = []
        t = np.linspace(0,1,nprofiles)
        for p in range(npts):
            self.spline_xps.append(PchipInterpolator(t,self.shft_xps[:,p])) # Percent, x
            self.spline_yps.append(PchipInterpolator(t,self.shft_yps[:,p]))
            self.spline_zps.append(PchipInterpolator(t,self.shft_zps[:,p]))

            self.spline_xss.append(PchipInterpolator(t,self.shft_xss[:,p]))
            self.spline_yss.append(PchipInterpolator(t,self.shft_yss[:,p]))
            self.spline_zss.append(PchipInterpolator(t,self.shft_zss[:,p]))
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
        [nprofiles,_] = self.shft_xps.shape
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
            self.shell_zss.append(PchipInterpolator(t,self.shft_zps[:,p]))

            self.shell_xps.append(PchipInterpolator(t,shell_xps[:,p]))
            self.shell_yps.append(PchipInterpolator(t,shell_yps[:,p]))
            self.shell_zps.append(PchipInterpolator(t,self.shft_zss[:,p]))
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
        [_,npts] = self.xss.shape

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
            [x,y] = b.get_point(t,equal_space=False)
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
            
            [x,y] = b.get_point(t,equal_space=False)
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
        
        nprofiles,_=self.shft_xss.shape
        for i in range(nprofiles):
            dx = self.shft_xss[i,:] - cx
            dy = self.shft_yss[i,:] - cy
            self.shft_xss[i,:] = (dx*cosd(angle) - dy*sind(angle)) + cx
            self.shft_yss[i,:] = (dx*sind(angle) + dy*cosd(angle)) + cy

            dx = self.shft_xps[i,:] - cx
            dy = self.shft_yps[i,:] - cy
            self.shft_xps[i,:] = (dx*cosd(angle) - dy*sind(angle)) + cx
            self.shft_yps[i,:] = (dx*sind(angle) + dy*cosd(angle)) + cy
        
        for i in range(self.control_x_ps.shape[1]):
            dx = self.control_x_ps[i,:] - cx
            dy = self.control_y_ps[i,:] - cy
            self.control_x_ps[i,:] = (dx*cosd(angle) - dy*sind(angle)) + cx
            self.control_y_ps[i,:] = (dx*sind(angle) + dy*cosd(angle)) + cy

            dx = self.control_x_ss[i,:] - cx
            dy = self.control_y_ss[i,:] - cy
            self.control_x_ss[i,:] = (dx*cosd(angle) - dy*sind(angle)) + cx
            self.control_y_ss[i,:] = (dx*sind(angle) + dy*cosd(angle)) + cy
            
    def center_le(self):
        """centers the blade by placing leading edge at 0,0
        """
        xc = self.shft_xss[0,0]
        yc = self.shft_yss[0,0]
        zc = self.shft_zss[0,0]

        self.shft_xss = self.shft_xss-xc
        self.shft_yss = self.shft_yss-yc
        self.shft_zss = self.shft_zss-zc

        self.shft_xps = self.shft_xps-xc
        self.shft_yps = self.shft_yps-yc
        self.shft_zps = self.shft_zps-zc

        self.control_x_ps = self.control_x_ps - xc
        self.control_y_ps = self.control_y_ps - yc
        self.control_z_ps = self.control_z_ps - zc

        self.control_x_ss = self.control_x_ss - xc
        self.control_y_ss = self.control_y_ss - yc
        self.control_z_ss = self.control_z_ss - zc

        self.b3.x = self.b3.x - xc
        self.b3.y = self.b3.y - yc
        self.b3.z = self.b3.z - zc

        self.te_center_x = self.te_center_x - xc
        self.te_center_y = self.te_center_y - yc
        self.zz = self.zz - xc
        
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

    def get_circumference(self,ss_x:np.ndarray,ss_y:np.ndarray,ps_x:np.ndarray,ps_y:np.ndarray):
        """returns the circumferene of a 2D airfoil profile

        Args:
            ss_x (np.ndarray): suction size x
            ss_y (np.ndarray): suction side y
            ps_x (np.ndarray): pressure side x
            ps_y (np.ndarray): pressure side y

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
        x = np.concatenate([self.shft_xss, np.flip(self.shft_xps[:,1:-1],axis=1)],axis=1)
        y = np.concatenate([self.shft_yss, np.flip(self.shft_yps[:,1:-1],axis=1)],axis=1)
        z = np.concatenate([self.shft_zss, np.flip(self.shft_zps[:,1:-1],axis=1)],axis=1)

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
            for j in range(x.shape[1]): # points for each span                 
                # Blue triangle
                v1 = (i-1)*x.shape[1]+(j-1)
                v2 = (i-1)*x.shape[1]+j
                v3 = i*x.shape[1] + j-1
                faces.append([v1,v2,v3])

                # red triangle
                v1 = (i-1)*x.shape[1]+j 
                v2 = i*x.shape[1] + j
                v3 = i*x.shape[1] + j-1 
                faces.append([v1,v2,v3])
                if j == x.shape[1]-1 and i<x.shape[0]-1:
                    # Blue triangle
                    v1 = (i-1)*x.shape[1]+(x.shape[1]-1)
                    v2 = (i-1)*x.shape[1]+j-1
                    v3 = i*x.shape[1] + x.shape[1]-1
                    faces.append([v1,v2,v3])

                    # red triangle
                    v1 = (i-1)*x.shape[1]+x.shape[1] 
                    v2 = i*x.shape[1] + x.shape[1]
                    v3 = i*x.shape[1] + x.shape[1]-1 
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
                blade.vectors[i][j] = vertices[f[j],:]

        
        blade.save(filename)
        
