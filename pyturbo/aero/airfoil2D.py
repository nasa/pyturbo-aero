import numpy as np
from typing import List, Optional, Tuple
from math import cos,sin,radians,degrees,atan2,sqrt,atan
from scipy.optimize import minimize_scalar
from ..helper import bezier,line2D,ray2D,arc,ray2D_intersection,exp_ratio,convert_to_ndarray,derivative,dist,pw_bezier2D,resample_curve
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import copy
import numpy.typing as npt

class Airfoil2D:
    """Design a 2D Airfoil using bezier curves 
    """
    '''Initial values'''
    alpha1:float                # Leading edge metal angle
    alpha2:float                # Trailing edge metal angle
    stagger:float               # Angle from Leading Edge to Trailing edge 
    chord:float                 # Length from leading edge to trailing edge

    camberBezier: bezier        # Bezier curve descrbing the camberline
    cambBezierX: List[float]    # Coordinates of the camber bezier curve control points
    cambBezierY: List[float]

    ssBezier:bezier             # Bezier curve describing the suction side
    ssBezier_pw:Optional[pw_bezier2D]
    ssBezierX:List[float]       # Coordinates of the suction side bezier curve control points
    ssBezierY:List[float]

    psBezier:bezier
    psBezier_pw:Optional[pw_bezier2D]
    psBezierX:List[float]
    psBezierY:List[float]

        
    def __init__(self,alpha1:float,alpha2:float,axial_chord:float,stagger:float):
        """Constructor for Airfoil2D 

        Args:
            alpha1 (float): inlet metal angle of the blade 0 to 90 deg
            alpha2 (float): outlet metal angle of the blade 0 to 90 deg
            axial_chord (float): Axial chord of the blade
            stagger (float): stagger angle measured from trailing edge to leading edge
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.stagger = stagger
        self.chord = axial_chord/cos(radians(self.stagger))
        self.__create_camber()
    
    def __create_camber(self):
        """Create the camberline of the blade
            Called by the constructor to automatically build the camberline
        """
        # Creates the airfoil camberline
        x2 = 0
        y2 = 0
        # Assumes CCW, call flip to convert to CW
        x1 = -self.chord*sin(radians(self.stagger))
        y1 = self.chord*cos(radians(self.stagger))
        r1 = ray2D(x1,y1,-sin(radians(self.alpha1)),-cos(radians(self.alpha1)))
        r2 = ray2D(x2,y2,-sin(radians(self.alpha2)),cos(radians(self.alpha2)))

        # Find intersection Point
        [x,y,_,_] = ray2D_intersection(r1,r2)
        self.cambBezierX = [x1,x,x2]
        self.cambBezierY = [y1,y,y2]
        b = bezier(self.cambBezierX,self.cambBezierY)
        self.camberBezier = b # Save the bezier curve

    def custom_camber(self,x:float,y:float):
        """Creates a custom bezier curve camberline from 3 points [LEPoint, (x,y), TEPoint]
            Usage: 
                Example profile.custom_camber(0.5,0.5)

        Args:
            x (float): arbitrary x coordinate
            y (float): arbitrary y coordinate
        """
        x1 = -self.chord * sin(self.stagger)
        y1 = self.chord * cos(self.stagger)
        x2 = 0
        y2 = 0
        self.cambBezierX = [x1,x,x2]
        self.cambBezierY = [y1,y,y2]
        self.camberBezier = bezier(self.cambBezierX,self.cambBezierY)

    def add_le_thickness(self,thickness:float,counter_rotation:bool=False):
        """Adds thickness to leading edge either on pressure side or suction side. 
           If counter rotation is false, thickness is added to the suction side 

        Args:
            thickness (float): float value for defining a bezier curve thickness
            counter_rotation (bool, optional): switches the side for the bezier curve initial thickness. Defaults to False.
        """
        self.le_thickness = thickness*self.chord
        self._counter_rotation=counter_rotation # self basically swaps the pressure side and suction size

        ssBezierX = [self.cambBezierX[0]] # First point is always the start of the camberline
        ssBezierY = [self.cambBezierY[0]]

        psBezierX = [self.cambBezierX[0]]
        psBezierY = [self.cambBezierY[0]]

        # Add Thickness to the suction side 
        if (not counter_rotation):
            theta_ss = 180-self.alpha1
            ssBezierX.append(ssBezierX[0]+cos(radians(theta_ss))*self.le_thickness)
            ssBezierY.append(ssBezierY[0]+sin(radians(theta_ss))*self.le_thickness)
        else:
            theta_ps = -self.alpha1
            psBezierX.append(psBezierX[0]+cos(radians(theta_ps))*self.le_thickness)
            psBezierY.append(psBezierY[0]+sin(radians(theta_ps))*self.le_thickness)

        self.ssBezierX = ssBezierX; self.ssBezierY = ssBezierY
        self.psBezierX = psBezierX; self.psBezierY = psBezierY
        
        b = bezier(ssBezierX,ssBezierY)
        self.ssBezier = b
        b = bezier(psBezierX,psBezierY)
        self.psBezier = b
    
    def match_le_thickness(self):
        """Matches the second derivative by changing the thickness of the opposite side

        Args:

        Returns:
            float: error in matching the second derivative

        """
        # Point that controls the thickness is index [1] 
        x = np.zeros(3)
        y = np.zeros(3)
        xx = np.zeros(3)
        yy = np.zeros(3)
        t1 = 0.001
        t2 = 0.002

        x[0] = self.cambBezierX[0]
        y[0]= self.cambBezierY[0]
        [x[1], y[1]] = self.ssBezier.get_point(t1)   # Derivative on the suction side (Needs to match)
        [x[2], y[2]] = self.ssBezier.get_point(t2)
        dydx1 = derivative.derivative_2(x,y)
        theta_ps = -self.alpha1
        
        def nest_ps_derivative2(h):
            self.psBezierX[1] = self.psBezierX[0]+cos(radians(theta_ps))*h
            self.psBezierY[1] = self.psBezierY[0]+sin(radians(theta_ps))*h
            self.psBezier = bezier(self.psBezierX,self.psBezierY)
            xx[0] = x[0]
            yy[0] = y[0]
            [xx[1],yy[1]] = self.psBezier.get_point(t1) # Gets the second derivative near the leading edge
            [xx[2],yy[2]] = self.psBezier.get_point(t2)
            dydx2 = derivative.derivative_2(xx,yy)
            err = abs(dydx2[1] - dydx1[1])
            return err

        def nest_ss_derivative2(h:float) -> float:
            self.ssBezierX[1] = self.ssBezierX[0]+cos(radians(theta_ps))*h
            self.ssBezierY[1] = self.ssBezierY[0]+sin(radians(theta_ps))*h
            self.ssBezier = bezier(self.ssBezierX,self.ssBezierY)
            xx[0] = x[0]
            yy[0] = y[0]
            [xx[1],yy[1]] = self.ssBezier.get_point(0.001)
            [xx[2],yy[2]] = self.ssBezier.get_point(0.002)
            dydx2 = derivative.derivative_2(xx,yy)
            err = abs(dydx2[1] - dydx1[1])
            return err

        if (not self._counter_rotation):
            temp = minimize_scalar(nest_ps_derivative2,bounds=(0,self.le_thickness*30),method="bounded") 
            h = temp.x # type: ignore
            if (nest_ps_derivative2(h) > 1):
                h = self.le_thickness

            self.psBezierX[1] = self.psBezierX[0]+cos(radians(theta_ps))*h  # Changes the second point to match the second derivative at the leading edge 
            self.psBezierY[1] = self.psBezierY[0]+sin(radians(theta_ps))*h
            self.psBezier = bezier(self.psBezierX,self.psBezierY)
        else:
            h = minimize_scalar(nest_ss_derivative2,bounds=(0,self.le_thickness*30),method="bounded")
            if (nest_ss_derivative2(h.x)>1): # type: ignore 
                h = self.le_thickness

            self.ssBezierX[1] = self.ssBezierX[0]+np.cos(radians(theta_ps))*h
            self.ssBezierY[1] = self.ssBezierY[0]+np.sin(radians(theta_ps))*h
            self.ssBezier = bezier(self.ssBezierX,self.ssBezierY)
    
    def te_create(self,radius:float,wedge_ss:float,wedge_ps:float):
        """Creates the trailing edge using a semi-circle

        Args:
            radius (float): circle radius
            wedge_ss (float): wedge angle on the suction side
            wedge_ps (float): wedge angle on the pressure side
        """

        self.te_radius = radius
        self.te_wedge_ss = wedge_ss
        self.te_wedge_ps = wedge_ps
        b = self.camberBezier
        [x, y] = b.get_point(1) # Get the last point
        [dx, dy] = b.get_point_dt(1)
        theta = degrees(atan2(dy,dx))
        
        if (theta>0):
            theta = theta-90
        else:
            theta = theta+90
        
        # Pressure side
        t = self.t_ps[-2]
        [xc,yc] = self.camberBezier.get_point([t-0.01, t, t+0.01])
        m2 = derivative.derivative_1(xc,yc)
        m2 = -1/m2[2]
        xc = xc[2]
        yc = yc[2]
        x_wedge_ps = x+cos(radians(theta-wedge_ps))*radius
        y_wedge_ps = y+sin(radians(theta-wedge_ps))*radius
        
        alpha_start = theta-wedge_ps
        angle =  alpha_start+360-(theta+180+wedge_ps)
        alpha_stop = alpha_start -angle
        alpha_mid = (alpha_start + alpha_stop)/2
        self.TE_ps_arc = arc(x,y,radius,alpha_start,alpha_mid) # type: ignore
        # Pressure Side - Match first derivative
        # Compute first derivative on the arc
        [xx,yy] = self.TE_ps_arc.get_point([0,0.01,0.02])
        dydx1 = derivative.derivative_1(xx,yy)
        m = dydx1[0]
        xo = (y_wedge_ps-yc-m*x_wedge_ps+m2*xc)/(m2-m)
        yo = y_wedge_ps-m*(x_wedge_ps-xo)
        
        
        #yo = (y_wedge_ps-m*x_wedge_ps)/(1+m*yc/xc)
        #xo = -yo*yc/xc
        
        # mc = (yo-y_wedge_ps)/(xo-x_wedge_ps) # Check
        self.psBezierX[-2] = xo[0] # set the last bezier point on the pressure side such that it matches the slope of the TE Arc
        self.psBezierY[-2] = yo[0]
        self.psBezierX[-1] = x_wedge_ps[0]
        self.psBezierY[-1] = y_wedge_ps[0]
        b = bezier(self.psBezierX,self.psBezierY) # Extends the bezier curve
        self.psBezier = b
        
        # Suction side
        if (int(max(self.t_ss))<1): # in case a camber percent (i.e. 0.7) is set
            t = self.t_ss[-1]
        else:
            t = self.t_ss[-2]
        
        [xc,yc] = self.camberBezier.get_point([t-0.01, t, t+0.01])
        m2 = derivative.derivative_1(xc,yc)
        m2 = -1/m2[1] 
        xc = xc[1]
        yc = yc[1] # find the perpendicular line slope
        x_wedge_ss = x+cos(radians(theta+180+wedge_ss))*radius
        y_wedge_ss = y+sin(radians(theta+180+wedge_ss))*radius
        
        alpha_start = theta-wedge_ss
        angle =  alpha_start+360-(theta+180+wedge_ss)
        alpha_stop = alpha_start-angle
        alpha_mid = (alpha_start + alpha_stop)/2
        self.TE_ss_arc = arc(x,y,radius,alpha_stop,alpha_mid) # type: ignore
        # Suction side - Match first derivative
        # Compute first derivative on the arc
        [xx,yy] = self.TE_ss_arc.get_point([0,0.01,0.02])
        dydx1 = derivative.derivative_1(xx,yy)
        m = dydx1[0]
        xo = (y_wedge_ss-yc-m*x_wedge_ss+m2*xc)/(m2-m) # interception of the 2nd to last point
        yo = y_wedge_ss-m*(x_wedge_ss-xo)
        # mc = (yo-y_wedge_ss)/(xo-x_wedge_ss) # Check
        self.ssBezierX[-2] = xo[0]
        self.ssBezierY[-2] = yo[0]
        self.ssBezierX[-1] = x_wedge_ss[0]
        self.ssBezierY[-1] = y_wedge_ss[0]
        b = bezier(self.ssBezierX,self.ssBezierY)
        self.ssBezier = b      
    
    def add_ss_thickness(self,thicknessArray:List[float],camberPercent:float=1,thickness_loc:Optional[List[float]]=None,expansion_ratio:float=1.2):
        """Adds thickness to the suction side by specifying bezier control points 

        Args:
            thicknessArray (List[float]): thickness along the suction side. Example: [0.2400, 0.2000, 0.1600, 0.1400]
            camberPercent (float, optional): Percent camber where straightening of the suction side happens. Defaults to 1.
            thickness_loc (List[float], optional): Location where thickness is applied. Defaults to None.
            expansion_ratio (float, optional): If thickness location is specified then this is not necessary otherwise thickness_loc is calculated by the expansion ratio. Defaults to 1.2.
        """
        if (thickness_loc is None):
            if expansion_ratio: # if expansion ratio is specified              
                t = exp_ratio(expansion_ratio,len(thicknessArray)+3,camberPercent)[1:] # Allocate 2 extra points for continunity at the trailing edge and for flow guidance
                # Define location of bezier control points from 0 to
                # camberPercent
            else:  
                t = convert_to_ndarray(thickness_loc) # type: ignore
        else:
            t = convert_to_ndarray(thickness_loc) # Manual custom definition 

        self.SS_thickness = np.array(thicknessArray)
        self.t_ss = t
        self.ss_exp_ratio = expansion_ratio
        thicknessArray = self.SS_thickness*self.chord # type: ignore
        # t - array
        # thicknessArray - distance from camber as an array
        # Check if camber is defined
        # Need to add leading and trailing edge points
        if (self.ssBezierX is None):
            self.ssBezierX = np.zeros(len(t)+2).tolist()  # type: ignore
            self.ssBezierY = self.ssBezierX
            self.ssBezierX[0] = self.cambBezierX[0]
            self.ssBezierY[0] = self.cambBezierY[0]
            self.ssBezierX[-1] = self.cambBezierX[-1]
            self.ssBezierY[-1] = self.cambBezierY[-1]
        
        b = self.camberBezier
        for i in range(0,len(t)):
            ## Compute the angle perpendicular
            [x, y] = b.get_point(t[i])
            if (t[i] == 0):
                theta = 180-self.alpha1
            elif (t[i]==1):
                theta = self.alpha2-180
            else:
                [dx, dy] = b.get_point_dt(t[i])
                theta = degrees(atan(-dx/dy))
            
            ## Compute the bezier thickness
            if (i>=len(thicknessArray)):
                self.ssBezierX.append(0)
                self.ssBezierY.append(0)
            else:
                t_ray = thicknessArray[i]
                xn = x-cos(radians(theta))*t_ray
                yn = y-sin(radians(theta))*t_ray
                self.ssBezierX.append(xn[0])
                self.ssBezierY.append(yn[0])

        self.ssBezier = bezier(self.ssBezierX,self.ssBezierY)

    # Adds thickness to pressure side
    def add_ps_thickness(self,thicknessArray:List[float],expansion_ratio:float=1.2):
        """Add thickness to the pressure side 

        Args:
            thicknessArray (List[float]): thickness along the suction side. Example: [0.2400, 0.2000, 0.1600, 0.1400]
            expansion_ratio (float, optional): determines the spacing of the thickness array from leading edge. Defaults to 1.2.

        """
        # 3 Extra points is added to account for the Leading Edge being
        # computed and last 2 are for the trailing edge
        
        ps_height_loc = exp_ratio(expansion_ratio,len(thicknessArray)+3,0.95)[1:]
        ps_height_loc = np.append(ps_height_loc,[1])
        t = convert_to_ndarray(ps_height_loc)

        self.PS_thickness = -1*np.array(thicknessArray)
        self.t_ps = t
        thicknessArray = (self.PS_thickness*self.chord).tolist()
        # t - array
        # thicknessArray - distance from camber as an array
        # Check if camber is defined
        
        # Need to add leading and trailing edge points
        # indx = len(self.psBezierX)
        b = self.camberBezier

        for i in range(len(t)):
            [x, y] = b.get_point(t[i])
            ## Compute the angle perpendicular
            if (t[i] ==0):
                theta = -self.alpha1
            elif (t[i]==1):
                theta = self.alpha2
            else:
                [dx, dy] = b.get_point_dt(t[i])
                theta = degrees(atan2(-dx,dy))
                #if (theta>0)
                #    theta = theta-90
                #else
                #    theta = theta+90
                #end
            
            ## Compute the bezier thickness
            if (i==0):
                self.psBezierX.append(0) # These two points are computed
                self.psBezierY.append(0)
            elif (i>len(thicknessArray)):
                self.psBezierX.append(0) # These two points are computed
                self.psBezierY.append(0)
            else:
                t_ray = thicknessArray[i-1]
                xn = x+cos(radians(theta))*t_ray
                yn = y+sin(radians(theta))*t_ray
                self.psBezierX.append(xn[0])
                self.psBezierY.append(yn[0])
            
        self.psBezier = bezier(self.psBezierX,self.psBezierY)

    def add_pitch(self,x_pitch:float):
        """Adds extra pitch by shifting the turbine blade over by a x direction

        Args:
            x_pitch (float): [description]
        """
        # Pitch_Add Adds pitch or shifts turbine by x direction
        self.shift(x_pitch,0)

    def shift(self,x:float,y:float):
        """Shifts the blade over by x or y direction. LE points +y where TE is at (0,0) Be sure to take into account the rotation of the blade
            
        Args:
            x (float): amount to shift the blade by in x direction
            y (float): amount to shift the bade by in y direction 
        """
        cambBezierX = convert_to_ndarray(self.cambBezierX)
        cambBezierY = convert_to_ndarray(self.cambBezierY)

        psBezierX = convert_to_ndarray(self.psBezierX)
        psBezierY = convert_to_ndarray(self.psBezierY)

        ssBezierX = convert_to_ndarray(self.ssBezierX)
        ssBezierY = convert_to_ndarray(self.ssBezierY)

        cambBezierX += x
        cambBezierY += y
        psBezierX += x
        psBezierY += y
        ssBezierX += x
        ssBezierY += y 
        
        self.camberBezier = bezier(cambBezierX,cambBezierY)
        self.cambBezierX = cambBezierX.tolist(); self.cambBezierY = cambBezierY.tolist()
        
        self.psBezierX = psBezierX.tolist(); self.psBezierY = psBezierY.tolist()
        self.ssBezierX = ssBezierX.tolist(); self.ssBezierY = ssBezierY.tolist()

        if isinstance(self.psBezier, pw_bezier2D):
            self.psBezier = self.psBezier.shift(x,y) # type: ignore
        else:
            self.psBezier = bezier(self.psBezierX,self.psBezierY)


        if isinstance(self.ssBezier, pw_bezier2D):
            self.ssBezier = self.ssBezier.shift(x,y) # type: ignore
        else:
            self.ssBezier = bezier(self.ssBezierX,self.ssBezierY)

        self.TE_ps_arc.x = self.TE_ps_arc.x + x
        self.TE_ps_arc.y = self.TE_ps_arc.y + y
        self.TE_ss_arc.x = self.TE_ss_arc.x + x
        self.TE_ss_arc.y = self.TE_ss_arc.y + y

    def get_points(self,n:int=100) -> Tuple[npt.NDArray,npt.NDArray]:
        """Get the airfoil points. This will you 100 poitns defining the suction side and pressure side

        Args:
            n (int, optional): Number of points. Defaults to 100.

        Returns:
            Tuple[npt.NDArray,npt.NDArray]: _description_
        """
        t = np.linspace(0,1,n)
        t_te = np.linspace(0,1,int(n*0.2))
        
        x_te,y_te = self.TE_ps_arc.get_point(t_te)
        [xPS, yPS] = self.psBezier.get_point(t)
        ps = np.vstack((
                np.hstack((xPS,x_te)),
                np.hstack((yPS,y_te))
            )).transpose()
        
        
        x_te,y_te = self.TE_ss_arc.get_point(t_te)
        [xSS, ySS] = self.ssBezier.get_point(t)
        ss = np.vstack((
                np.hstack((xSS,x_te)),
                np.hstack((ySS,y_te))
            )).transpose()
        
        ss = resample_curve(ss,n)
        ps = resample_curve(ps,n)
        return ss,ps
        
    def get_centroid(self):
        """Returns the centroid of the airfoil

        Returns:
            float,float: centroid x and y coordinates (x,y)
        """
        n = 100
        t = np.linspace(0,1,n)
        [x1,y1] = self.psBezier.get_point(t)
        [x2,y2] = self.ssBezier.get_point(t)
        # [x3,y3] = self.TE_ss_arc.get_point(t)
        # [x4,y4] = self.TE_ps_arc.get_point(t)
        xc = sum(x1+x2)/(2*n)
        yc = sum(y1+y2)/(2*n)
        return xc,yc

    def flip_cw(self):
        # FLIP Flips the turbine from CCW to CW design
        xc = 0
        self.CCW = 0
        # Flip the Camber
        for i in range(0,len(self.cambBezierX)):
            dx =  xc -self.cambBezierX[i]
            self.cambBezierX[i] = xc+dx
        
        self.camberBezier = bezier(self.cambBezierX,self.cambBezierY)
        
        for i in range(0,len(self.psBezierX)):
            dx =  xc -self.psBezierX[i]
            self.psBezierX[i] = xc+dx
        
        self.psBezier = bezier(self.psBezierX,self.psBezierY)
        
        if (type(self.ssBezier) == 'pw_bezier2D'):
            for i in range(len(self.ssBezier.bezierArray)): # type: ignore
                for j in range(len(self.ssBezier.bezierArray[i].x)): # type: ignore
                    dx =  xc - self.ssBezier.bezierArray[i].x[j] # type: ignore
                    self.ssBezier.bezierArray[i].x[j] = xc+dx  # type: ignore
        
            for i in range(0,len(self.ssBezierX)):
                dx =  xc - self.ssBezierX[i]
                self.ssBezierX[i] = xc + dx
        else:
            for i in range(0,len(self.ssBezierX)):
                dx =  xc -self.ssBezierX[i]
                self.ssBezierX[i] = xc + dx
            
            self.ssBezier = bezier(self.ssBezierX,self.ssBezierY)
        
        dx =  xc - self.TE_ss_arc.x
        self.TE_ss_arc.x = xc+dx
        dx =  xc - self.TE_ps_arc.x
        self.TE_ps_arc.x = xc+dx
        
        # ps_arc_start = self.TE_ps_arc.alpha_start
        # ps_arc_stop = self.TE_ps_arc.alpha_stop
        self.TE_ps_arc.alpha_start = 180-self.TE_ps_arc.alpha_start
        self.TE_ps_arc.alpha_stop = -self.TE_ps_arc.alpha_stop+180
        self.TE_ss_arc.alpha_start = -self.TE_ss_arc.alpha_start-180
        self.TE_ss_arc.alpha_stop = -self.TE_ss_arc.alpha_stop - 180

    def flip(self):
        """Swaps the leading edge with trailing edge 

        """
        y1c = self.cambBezierY[0]
        x1c = self.cambBezierX[0]
        for i in range(len(self.cambBezierX)):
            self.cambBezierX[i] = x1c - self.cambBezierX[i]
            self.cambBezierY[i] = y1c - self.cambBezierY[i]

        self.camberBezier = bezier(self.cambBezierX,self.cambBezierY)

        self.TE_ps_arc.x = x1c - self.TE_ps_arc.x
        self.TE_ps_arc.y = y1c - self.TE_ps_arc.y
        self.TE_ps_arc.alpha_start = self.TE_ps_arc.alpha_start+180
        self.TE_ps_arc.alpha_stop = self.TE_ps_arc.alpha_stop+180
        
        self.TE_ss_arc.x = x1c - self.TE_ss_arc.x
        self.TE_ss_arc.y = y1c - self.TE_ss_arc.y
        self.TE_ss_arc.alpha_start = self.TE_ss_arc.alpha_start+180
        self.TE_ss_arc.alpha_stop = self.TE_ss_arc.alpha_stop+180
        
        # Flip Pressure side
        y1ps = self.psBezierY[0]
        x1ps = self.psBezierX[0]
        for i in range(0,len(self.psBezierX)):
            self.psBezierX[i] = x1ps - self.psBezierX[i]
            self.psBezierY[i] = y1ps - self.psBezierY[i]
        
        self.psBezier = bezier(self.psBezierX,self.psBezierY)

        # Flip Suction side
        
        if isinstance(self.ssBezier,pw_bezier2D):
            y1ss = self.ssBezier.bezierArray[0].y[0]
            x1ss = self.ssBezier.bezierArray[0].x[0]
            for i in range(0,len(self.ssBezier.bezierArray)):
                for j in range(0,len(self.ssBezier.bezierArray[i].x)):
                    self.ssBezier.bezierArray[i].x[j] = x1ss - self.ssBezier.bezierArray[i].x[j]
                    self.ssBezier.bezierArray[i].y[j] = y1ss - self.ssBezier.bezierArray[i].y[j]
                
            for i in range(0,len(self.ssBezierX)):
                self.ssBezierX[i] = x1ss - self.ssBezierX[i]
                self.ssBezierY[i] = y1ss - self.ssBezierY[i]
            
        else:
            y1ss = self.ssBezierY[0]
            x1ss = self.ssBezierX[0]
            for i in range(0,len(self.ssBezierX)):
                self.ssBezierX[i] = x1ss - self.ssBezierX[i]
                self.ssBezierY[i] = y1ss - self.ssBezierY[i]
            
            self.ssBezier = bezier(self.ssBezierX,self.ssBezierY)

    # Gets the axial chord of the foil
    def get_axial_chord(self):
        """returns the axial chord

        Returns:
            float: axial_chord
        """
        return self.chord*cos(radians(self.stagger))

    def add_ss_flow_guidance(self,n:int=8):
        """This function straightens out the suction side by specifying n bezier control points instead of a straight line. 

        Args:.
            n (int): number of control points, increase this to make straightening more aggressive. Defaults to 8.
            
        """
        ssBezierX = convert_to_ndarray(self.ssBezierX)
        ssBezierY = convert_to_ndarray(self.ssBezierY)

        x1 = self.ssBezierX[-2]
        y1 = self.ssBezierY[-2]
        # Find ending point on trailing edge ss side 
        [x2, y2] = self.TE_ss_arc.get_point(0)       
        # Create a line from x1 to x2
        bl = bezier([x1, x2[0]],[y1, y2[0]])
        # Append points on the line at equal distance spacing to
        # ssBezierX,ssBezierY -> define new ssBezier
        [x, y] = bl.get_point(np.linspace(0,1,n))
        self.ssBezierX = np.append(ssBezierX[0:-2],x).tolist()
        self.ssBezierY = np.append(ssBezierY[0:-2],y).tolist()
        self.ssBezier = bezier(self.ssBezierX,self.ssBezierY)

    def add_ss_flow_guidance_2(self,s_c:float,n:int):
        """Straightens out the suction side. Computes the intersection point of the throat and draws a line, adds bezier points along the line

        Args:
            s_c (float): pitch-to-chord ratio
            n (int): number of bezier control points
        """
        self.s_c = s_c
        # Compute where throat starts in terms of ts (suction side)
        s,ss,ps,ts,airfoil = channel_get(self,self.s_c)
       
        # Find the point at ts (suction side)            
        x1,y1 = self.ssBezier.get_point(ts)
        [x2, y2] = self.TE_ss_arc.get_point(0)
        
        ssBezier_new = list() 
        for p,q in zip(self.ssBezierX,self.ssBezierY):
            if q>y1:
                ssBezier_new.append((float(p),float(q)))
                
        # ssBezierY_new = [p for p in self.ssBezierY if p>y1]
        ssBezier_new.append((float(x1),float(y1))) # type: ignore
        ssBezier_new = np.array(ssBezier_new)
        
        # Create a line from self point to the end of the ss bezier
        # curve
        bl = bezier([float(x1), float(x2)],[float(y1), float(y2)])            
        # Append points on the line at equal distance spacing to
        # ssBezierX,ssBezierY -> define new ssBezier
        [x, y] = bl.get_point(np.linspace(0,1,n))
        bezier_pts = np.vstack([np.hstack([ssBezier_new[:,0],x[1:]]),
                                np.hstack([ssBezier_new[:,1],y[1:]]).tolist()]).transpose()
        
        self.ssBezierX = bezier_pts[:,0].tolist()
        self.ssBezierY = bezier_pts[:,1].tolist()
        self.ssBezier = bezier(self.ssBezierX,self.ssBezierY)
        
        
    def plot_camber(self):
        """Plots the camber of the airfoil
        
        Returns:
            None
        """
        tplot = np.linspace(0,1,50)
        # plt.ion()
        marker_style = dict(markersize=8, markerfacecoloralt='tab:red')

        [xcamber, ycamber] = self.camberBezier.get_point(tplot)
        fig = plt.figure(num=1, clear=True)
        plt.plot(xcamber,ycamber, color='black', linestyle='solid', 
            linewidth=2)
        plt.plot(self.camberBezier.x,self.camberBezier.y, color='red', marker='o',linestyle='--',**marker_style)         # type: ignore
        plt.gca().set_aspect('equal')
        plt.show()

    def plot2D(self):
        """Plots the airfoil

        Returns:
            None
        """
        t = np.linspace(0,1,200)
        # plt.ion()
        [xcamber, ycamber] = self.camberBezier.get_point(t)
        [xPS, yPS] = self.psBezier.get_point(t)
        [xSS, ySS] = self.ssBezier.get_point(t)

        fig = plt.figure(num=1,clear=True)
        plt.plot(xcamber,ycamber, color='black', linestyle='solid', 
            linewidth=2)
        plt.plot(xPS,yPS, color='blue', linestyle='solid', 
            linewidth=2)
        plt.plot(xSS,ySS, color='red', linestyle='solid', 
            linewidth=2)
        plt.plot(self.psBezier.x,self.psBezier.y, color='blue', marker='o',markerfacecolor="None",markersize=8)
        plt.plot(self.ssBezier.x,self.ssBezier.y, color='red', marker='o',markerfacecolor="None",markersize=8)
        # Plot the line from camber to the control points
        # suction side
        for indx in range(0,len(self.ssBezierX)):
            x = self.ssBezierX[indx]
            y = self.ssBezierY[indx]
            d = dist(x,y,xcamber,ycamber)
            min_indx = np.where(d == np.amin(d))[0][0]
            plt.plot([x,xcamber[min_indx]],[y,ycamber[min_indx]], color='black', linestyle='dashed') # type: ignore
        # pressure side
        for indx in range(0,len(self.psBezierX)):
            x = self.psBezierX[indx]
            y = self.psBezierY[indx]
            d = dist(x,y,xcamber,ycamber)
            min_indx = np.where(d == np.amin(d))[0][0]
            plt.plot([x,xcamber[min_indx]],[y,ycamber[min_indx]], color='black', linestyle='dashed') # type: ignore
        # Plot the Trailing Edge
        t = np.linspace(0,1,20)
        [x, y] = self.TE_ps_arc.get_point(t)
        plt.plot(x,y, color='blue', linestyle='solid')

        [x, y] = self.TE_ss_arc.get_point(t)
        plt.plot(x,y, color='red', linestyle='solid')
        plt.gca().set_aspect('equal')
        plt.show()

    def plot2D_channel(self,pitchChordRatio:float):
        """plots the 2D airfoil in a channel with another airfoil given a pitch to chord ratio

        Args:
            pitchChordRatio (float): pitch to chord ratio (spacing between airfoils relative to the chord)

        Returns:
            plt.figure: [description]
        """
        
        # outputs the Area/A*
        t = np.linspace(0,1,100)
        t_te = np.linspace(0,1,20)
        [xcamber, ycamber] = self.camberBezier.get_point(t)
        [xPS, yPS] = self.psBezier.get_point(t)
        [xSS, ySS] = self.ssBezier.get_point(t)
        [x,y] = self.TE_ss_arc.get_point(t_te)
        xSS = np.append(xSS,x)
        ySS = np.append(ySS,y)

        [x,y] = self.TE_ps_arc.get_point(t_te)
        xPS = np.append(xPS,x)
        yPS = np.append(yPS,y)
        
        s,ss,ps,t_ss,turb2 = channel_get(self,pitchChordRatio)
        
        bcamber2=turb2.camberBezier
        bPS2 = turb2.psBezier
        bSS2 = turb2.ssBezier

        [xcamber2, ycamber2] = bcamber2.get_point(t)
        [xPS2, yPS2] = bPS2.get_point(t)
        [xSS2, ySS2] = bSS2.get_point(t)
        
        [x,y] = turb2.TE_ss_arc.get_point(t)
        xSS2 = np.append(xSS2,x)
        ySS2 = np.append(ySS2,y)

        [x,y] = turb2.TE_ps_arc.get_point(t)
        xPS2 = np.append(xPS2,x)
        yPS2 = np.append(yPS2,y)
        
        
        # Plot turbine and pitch
        fig= plt.figure(num=1, clear=True)
        plt.plot(xcamber,ycamber, color='black', linestyle='dashed', 
            linewidth=1.5)
        plt.plot(xPS,yPS, color='blue', linestyle='solid', 
            linewidth=1.5)
        plt.plot(xSS,ySS, color='red', linestyle='solid', 
            linewidth=1.5)

        plt.plot(xcamber2,ycamber2, color='black', linestyle='dashed', 
            linewidth=1.5)
        plt.plot(xPS2,yPS2, color='blue', linestyle='solid', 
            linewidth=1.5)
        plt.plot(xSS2,ySS2, color='red', linestyle='solid', 
            linewidth=1.5)
        plt.gca().set_aspect('equal')       

        # Throat
        plt.plot([ss[0,0],ps[0,1]],[ss[0,0],ps[0,1]], color='black', linestyle='dashed', 
            linewidth=1.1)
        plt.plot([ss[-1,0],ps[-1,1]],[ss[-1,0],ps[-1,1]], color='black', linestyle='dashed', 
            linewidth=1.1)
        
        plt.gca().set_aspect('equal')       

        plt.gca().set_xlim(1.05*min([min(xSS),min(xSS2),min(xPS),min(xPS2)]),1.05*max([max(xSS),max(xSS2),max(xPS),max(xPS2)]))
        plt.gca().set_ylim(1.05*min([min(ySS),min(ySS2),min(yPS),min(yPS2)]),1.05*max([max(ySS),max(ySS2),max(yPS),max(yPS2)]))
        plt.show()
    
    def plot_derivative2(self,xlim=[0,1],ylim=[-400,400]):
        """Plots the second derivative of the airfoil.

        References:

            https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/

        Args:
        
            xlim (list, optional): Plot x-axis. Defaults to [0,1].
            ylim (list, optional): Plot y-axis. Defaults to [-400,400].

        """
    
        t = np.linspace(0,1,100)
        xs,ys = self.ssBezier.get_point(t,equally_space_pts=True)
        xp,yp = self.psBezier.get_point(t,equally_space_pts=True)

        ds2 = np.zeros(len(xs)-1) 
        dp2 = np.zeros(len(xs)-1) 
        alens = np.zeros(len(xs)-1)   # len along the ss and ps side
        alenp = np.zeros(len(xs)-1) 
        for i in range(1,len(xs)-1): 
            alens[i] = sqrt((ys[i]-ys[i-1])**2+(xs[i]-xs[i-1])**2)+alens[i-1]
            alenp[i] = sqrt((yp[i]-yp[i-1])**2+(xp[i]-xp[i-1])**2)+alenp[i-1]

            #dx^2 / dy^2
            ds2[i]=2*xs[i-1]/((ys[i]-ys[i-1])*(ys[i+1]-ys[i-1]))   \
                -2*xs[i]/((ys[i+1]-ys[i])*(ys[i]-ys[i-1]))         \
                +2*xs[i+1]/((ys[i+1]-ys[i])*(ys[i+1]-ys[i-1]))

            dp2[i]=2*xp[i-1]/((yp[i]-yp[i-1])*(yp[i+1]-yp[i-1]))   \
                -2*xp[i]/((yp[i+1]-yp[i])*(yp[i]-yp[i-1]))         \
                +2*xp[i+1]/((yp[i+1]-yp[i])*(yp[i+1]-yp[i-1]))
        
        
        # Suction side
        # plt.ion()
        fig = plt.figure(num=1)
        plt.plot(alens/np.max(alens),ds2, color='red', linestyle='solid', 
            linewidth=2)
        plt.plot(alenp/np.max(alenp),dp2, color='blue', linestyle='solid', 
            linewidth=2)
        plt.xlim(xlim)
        plt.ylim(ylim)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, ['d^2y/dx^2 suction side','d^2y/dx^2 pressure side'])
        plt.xlabel('s(x)/s')
        plt.ylabel('d^2y/dx^2')
        plt.show()

    def thickness(self):
        """Calculates the location and value of maximum thickness along with average thickness

        Returns:
            int: along camberline of max thickness
            float: max thickness
            float: average thickness
        """
        t = np.linspace(0,1,100)
        [xss, yss] = self.ssBezier.get_point(t)
        [xps, yps] = self.psBezier.get_point(t)

        # Points are equally spaced
        # Compute minimum distance for each point on SS
        d=100*np.ones((len(xss),1))
        min_ps_point=np.ones((len(xss),1))
        xx_ps=xss 
        yy_ps=xss
        for i in range(1,len(xss)):
            for j in range(i,len(xps)):
                temp=sqrt((xss[i]-xps[j])**2+(yss[i]-yps[j])**2)
                if (temp<d[i]):
                    d[i]=temp
                    min_ps_point[i] = j
                    xx_ps[i] = xps[j]
                    yy_ps[i] = yps[j]
                        
        # Find max thickness
        max_thickness = d.max() # gives value and index
        indx = np.argmax(d)
        # Find avg thickness
        avg_thickness = np.mean(d)
        return indx, max_thickness, avg_thickness

def channel_get(airfoil:Airfoil2D,s_c:float):        
    """Gets adjacent airfoil 

    Args:
        airfoil (Airfoil2D): airfoil 2D object
        s_c (float): pitch to chord ratio

    Returns:
        List[float]: pitch in between airfoils
        numpy.ndarray: x coordinate of the suction side
        numpy.ndarray: x coordinate of the pressure side
        numpy.ndarray: y coordinate of the suction side
        numpy.ndarray: y coordinate of the pressure side
        airfoil2D: the adjacent airfoil
    """
    airfoil2 = copy.deepcopy(airfoil)
    airfoil2.add_pitch(s_c*airfoil2.chord)
    # Generate a bunch of points on the Pressure side (Turbine2) 
    t_ss = np.linspace(0,1,100)
    t_ps = np.linspace(0,1,100)


    if (airfoil._counter_rotation):
        [x2,y2] = airfoil2.psBezier.get_point(t_ps)
        [x1,y1] = airfoil.ssBezier.get_point(t_ss) # Pressure side to suction side
    else:
        [x1,y1] = airfoil.psBezier.get_point(t_ps)
        [x2,y2] = airfoil2.ssBezier.get_point(t_ss) # Pressure side to suction side
        
    s = np.zeros(len(t_ss)) 
    ss = np.zeros((len(t_ss),2))
    ps = np.zeros((len(t_ss),2))
    t_ss_min = np.zeros(len(t_ss))
    # Compute Minimum distance
    for i in range(len(t_ps)):
        # compute distance from pressure to suction
        dArray = dist(x1[i],y1[i],x2,y2) # Takes a point on pressure side and
                                    #   compute the distance to 
                                    #   all points on the suction side
        min_indx = np.where(dArray == np.amin(dArray))
        s[i] = dArray[min_indx]
        ps[i,0] = x1[i]
        ps[i,1] = y1[i]
        ss[i,0] = x2[min_indx]
        ss[i,1] = y2[min_indx]
        t_ss_min[i] = t_ss[min_indx]
    
    t_ss = t_ss_min[-1] # TODO Need to check if this is the most efficient way of doing it
    
    return s,ss,ps,t_ss,airfoil