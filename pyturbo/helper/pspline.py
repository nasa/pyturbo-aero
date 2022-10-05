import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import CubicSpline
from scipy.interpolate import PPoly
from scipy import integrate
import math
import enum
import copy
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize


def convert_to_ndarray(t):
    """
        converts a scalar or list to numpy array 
    """

    if type(t) is not np.ndarray and type(t) is not list: # Scalar
        t = np.array([t],dtype=float)
    elif (type(t) is list):
        t = np.array(t,dtype=float)
    return t



class spline_type(enum.Enum):
    pchip = 1
    spline = 2 # Cubic spline 

def spline_intersect(spline1,spline2,xmin,xmax):
    def intersect_test(x0):
        y1 = spline1(x0)
        y2 = spline2(x0)
        return abs(y1-y2)
    bnds = ((xmin,xmax),)
    x0 = (xmin+xmax)/2.0
    res = minimize(intersect_test,x0=(x0),bounds=bnds,tol=1E-5)
    if (res.success):
        return res.x[0]
    else:
        return -1

def pspline_intersect(pspline1,pspline2,tmin,tmax):
    def intersect_test(t0):
        pt1,_ = pspline1(t0[0])
        pt2,_ = pspline2(t0[1])
        dx = pt2[0,0]-pt1[0,0]
        dy = pt2[0,1]-pt1[0,1]
        return math.sqrt(dx*dx + dy*dy)

    bnds = ((tmin,tmax),(tmin,tmax),)
    t0 = (tmin+tmax)/2.0
    res = minimize(intersect_test,x0=(t0,t0),bounds=bnds,tol=1E-4)
    if (res.success and res.fun<0.005):
        return res.x
    else:
        return res.x*0-1 # return -1 for no intersect
        
class pspline:
    '''
        Percentage Spline - fits a spline on a curve and allows you to extract points with a percentage as input
    '''
    def __segkernel__(self,y,t):
        t = convert_to_ndarray(t)
        val = np.zeros(len(t))
        for k in range(0,self.ndim):
            val += np.polyval(self.polyarray[k,:],t)**2
        val = np.sqrt(val)
        return val
    
    def __init__(self,px,py,pz=[],method=spline_type.pchip):
        self.px = convert_to_ndarray(px)
        self.py = convert_to_ndarray(py)
        
        self.n = len(px)
        self.ndim = 3
        if (len(pz)==0):
            self.ndim = 2 # there's only x and y
            self.pxy = np.stack((self.px,self.py),axis=1)
        else:
            self.pz = convert_to_ndarray(pz)
            self.pxy = np.stack((self.px,self.py,self.pz),axis=1)

        self.chordlen = np.sqrt( 
            np.sum( 
                np.array((np.power(np.diff(self.px),2), np.power(np.diff(self.py),2)),dtype=np.float64)
                ,axis=0))
        self.chordlen = self.chordlen/np.sum(self.chordlen)

        self.cumarclen = np.append(np.zeros(1),np.cumsum(self.chordlen))

        # compute parametric splines
        self.spl = []
        self.spld = []
        diffarray = np.array([[3,0,0],[0,2,0],[0,0,1],[0,0,0]])
        for i in range(0,self.ndim):
            if (method==spline_type.pchip):
                self.spl.append(PchipInterpolator(self.cumarclen,self.pxy[:,i]))
            if (method==spline_type.spline):
                self.spl.append(CubicSpline(self.cumarclen,self.pxy[:,i]))
                # nc = numel(self.spl[i].coefs)
                # if nc < 4:
                #     # just pretend it has cubic segments
                #     self.spl[i].coefs = np.stack(np.zeros(1,4-nc),self.spl[i].coefs)
                #     self.spl[i].order = 4
           
            # and now differentiate them
            xp = copy.deepcopy(self.spl[i])
            xp.c = np.transpose(np.matmul(np.transpose(xp.c),diffarray))
            # self.xp.order = 3
            self.spld.append(xp)

        '''
            Catch the case where there were exactly three points
            in the curve, and spline was used to generate the
            interpolant. In self case, spline creates a curve with
            only one piece, not two.
        '''
        if (self.cumarclen.size == 3) and (method == spline_type.spline):
            self.cumarclen = self.spl[0].x
            n = np.size(self.cumarclen)
            self.chordlen = [sum(self.chordlen)]
                
        '''
            Generate the total arclength along the curve
            by integrating each segment and summing the
            results. The integration scheme does its job
            using an ode solver.
        '''
        
        # polyarray here contains the derivative polynomials
        # for each spline in a given segment
        self.polyarray = np.zeros((self.ndim,3))
        self.seglen = np.zeros((self.n-1,1))
        _,pieces = self.spl[0].c.shape
        for i in range(pieces):
            # extract polynomials for the derivatives
            for j in range(self.ndim):
                self.polyarray[j,:] = self.spld[j].c[:,i] # Grab the coefficients
            
            '''
                Integrate the arclength for the i'th segment
                using ode45 for the integral. I could have
                done self part with quad too, but then it
                would not have been perfectly (numerically)
                consistent with the next operation in self tool.
            '''
            temp = integrate.odeint(self.__segkernel__,y0=0,t=[0,self.chordlen[i]],rtol=1E-5)
            self.seglen[i] = temp[-1]
            # r = integrate.ode(self.__segkernel__).set_integrator('vode',method='bdf') 
            # r.set_initial_value(0,0)    
            
            # self.seglen[i] = r.integrate(1)
        
        # and normalize the segments to have unit total length
        self.totalsplinelength = np.sum(self.seglen)
        self.cumseglen = np.append(np.zeros(1),np.cumsum(self.seglen))

    def __call__(self, t):
        return self.get_point(t)

    def get_curve_len(self,t):
        return self.totalsplinelength*t
    
    def get_point(self,t):
        '''
            Inputs:
                t is from 0 to 1
            Returns:
                x,y,(z) of the spline curve
        '''
        t = convert_to_ndarray(t)      
        nt = len(t)  
        
        pt = np.empty((nt,self.ndim))
        pt[:] = np.nan

        '''
        Find out which bins we need

        '''
        tbins = np.digitize(t*self.totalsplinelength, self.cumseglen)     

        # Catch problems at the end             
          
        tbins[np.logical_or((tbins <= 0),(t <= 0))] = 1 # for self to work, needs to be numpy array
        tbins[np.logical_or((tbins >= (self.n-1)), (t >= 1))] = self.n - 2

        '''
        Do the fractional integration within each segment
        for the interpolated points. t is the parameter
        used to define the splines. It is defined in terms
        of a linear chordal arclength. This works nicely when
        a linear piecewise interpolant was used. However,
        what is asked for is an arclength interpolation
        in terms of arclength of the spline itself. Call s
        the arclength traveled along the spline.
        s = self.totalsplinelength*t;
        '''
        s = self.totalsplinelength*t

        ti = t
        for i in range(nt):
            # si is the piece of arc length that we will look
            # for in self spline segment.
            si = s[i] - self.cumseglen[tbins[i]]
            
            # extract polynomials for the derivatives
            # in the interval the point lies in
            for j in range(self.ndim):
                self.polyarray[j,:] = self.spld[j].c[:,tbins[i]]
            
            
            # the ode45 options will now include an events property
            # so we can catch zero crossings.            
            
            ''' 
            we need to integrate in t, until the integral
            crosses the specified value of si. Because we
            have defined totalsplinelength, the lengths will
            be normalized at self point to a unit length.
            '''
            
            # Start the ode solver at -si, so we will just
            # look for an event where y crosses zero.
            [temp,infodict] = integrate.odeint(self.__segkernel__,y0=-si,t=[0,self.chordlen[tbins[i]]],rtol=1E-9)

            # [tout,yout,te,ye] = ode45(@(t,y) self.segkernel(t,y),[0,self.chordlen(tbins(i))],-si,opts); %#ok
            
            # # we only need that point where a zero crossing occurred
            # # if no crossing was found, then we can look at each end.
            # if (not te): # Check if te is empty
            #     ti[i] = te[1] + self.cumarclen(tbins[i])
            # else:
            #     # a crossing must have happened at the very
            #     # beginning or the end, and the ode solver
            #     # missed it, not trapping that event.
            #     if abs(yout[1]) < abs(yout[-1]):
            #         # the event must have been at the start.
            #         ti[i] = tout[0] + self.cumarclen(tbins[i])
            #     else:
            #         # the event must have been at the end.
            #         ti[i] = tout[-1] + self.cumarclen(tbins[i])
                
        
        # Interpolate the parametric splines at ti to get
        # our interpolated value.
        for L in range(self.ndim):
            pp = PPoly(self.spl[L].c, self.spl[L].x) # ppval(self.spl{L},ti)
            pt[:,L] = pp(ti)
        pt = convert_to_ndarray(pt)
        
        # do we need to compute first derivatives here at each point?
        dudt = np.zeros((nt,self.ndim))
        for L in range(self.ndim):
            pp = PPoly(self.spld[L].c, self.spld[L].x) # ppval(ti,self.spld{L})
            dudt[:,L] = pp(ti)
        dudt = convert_to_ndarray(dudt)

        return pt,dudt