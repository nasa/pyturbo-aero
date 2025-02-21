'''
    Test code to design a passage 
'''
import numpy as np
from pyturbo.helper import bezier
import matplotlib.pyplot as plt 
from scipy.optimize import minimize_scalar

def create_passage_turbine_bezier(bPlot:bool=False):
    """_summary_

    Returns:
        Tuple: _description_
    """
    # Shroud is defined using a thickness offset from the hub to construct a spline
    rhub_ctrl_pts = [0.12,0.10,0.085,
                    0.06,0.04,
                    0.0235, 0.0235,0.0235]

    xhub_ctrl_pts = [0.0, 0.0, 0.0,
                    0.02,0.05,
                    0.08,0.12,0.13]

    dr = [0.008, 0.008, 0.008, 
        0.015, 0.02,
        0.025,0.025,0.025]
    t = [0, 0.1, 0.2,
        0.4, 0.6,
        0.92, 0.98, 1.0]

    hub = bezier(xhub_ctrl_pts,rhub_ctrl_pts)
    shroud_dh = bezier(t,dr)

    def r2(x:float,x1:float,r1:float,slope:float):
        return slope*(x-x1)+r1

    def dh_error(x2:float,x1:float,r1:float,dx:float,dr:float,h:float):
        slope = -dx/dr
        r2_guess = r2(x2,x1,r1,slope)
        return np.abs(h-np.sqrt((x1-x2)**2+(r1-r2_guess)**2))

    # Build Shroud
    npts = 30
    xhub,rhub = hub.get_point(np.linspace(0,1,npts))
    dx_pts = np.gradient(xhub, np.linspace(0,1,npts))
    dr_pts = np.gradient(rhub, np.linspace(0,1,npts))
    _, h_pts = shroud_dh.get_point(np.linspace(0,1,npts))
    xshroud = xhub*0
    rshroud = xhub*0; i = 0
    for dx,dr,x1,r1,h in zip(dx_pts,dr_pts,xhub,rhub,h_pts): 
        if abs(dx/dr) >20:
            xshroud[i] = x1
            rshroud[i] = r1+h
        else:
            res = minimize_scalar(dh_error,bounds=[x1,x1+1.5*h],args=(x1,r1,dx,dr,h))
            if r2(res.x,x1,r1,-dx/dr)<r1:
                res = minimize_scalar(dh_error,bounds=[x1-1.5*h,x1],args=(x1,r1,dx,dr,h))
            
            xshroud[i] = res.x
            rshroud[i] = r2(xshroud[i],x1,r1,-dx/dr)
            h_check = np.sqrt((x1-xshroud[i])**2+(r1-rshroud[i])**2)
            # print(f"h = {h} h_check = {h_check}")
        i+=1
    if bPlot:
        plt.figure(num=1,clear=True)
        plt.plot(xhub,rhub)
        plt.plot(xshroud,rshroud,'.')
        plt.axis('scaled')
        plt.xlabel('x-axial')
        plt.ylabel('r-radial')
        plt.show()
    
    return hub, rhub, xshroud, rshroud

def test_passage_centrif_compressor():
    # Inputs:
    ## Fixed coordinates
    inlet_gap = 0.15
    exit_gap = 0.1
    height = 0.3
    
    ## Control Points (percent_radius,theta)
    hub_ctrl_pts = [(0.9,20),(0.95,60)]
    shroud_ctrl_pts = [(0.9,20),(0.95,60)]
    
    # Calculations
    hub_radius = height
    shroud_radius = height - exit_gap
    
    observer = [0,height]
    
    
    