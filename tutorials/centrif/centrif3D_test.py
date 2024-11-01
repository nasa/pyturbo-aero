from pyturbo.aero import Centrif2D, Centrif3D
from pyturbo.helper import ellispe, StackType, bezier
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt

def create_passage_compressor(bPlot:bool=False):
    # Shroud is defined using a thickness offset from the hub to construct a spline
    xhub_ctrl_pts = [0.0, 0.02, 0.08, 0.11, 0.12, 0.12]
    rhub_ctrl_pts = [0.0, 0.0, 0.004, 0.06, 0.09, 0.10]
    hub = bezier(xhub_ctrl_pts,rhub_ctrl_pts)
    
    xshroud_ctrl_pts = [0.0, 0.02, 0.05, 0.09, 0.11, 0.11]
    rshroud_ctrl_pts = [0.03, 0.03, 0.04, 0.055, 0.09, 0.10]
    shroud = bezier(xshroud_ctrl_pts,rshroud_ctrl_pts)

    xhub,rhub = hub.get_point(np.linspace(0,1,100))
    xshroud,rshroud = shroud.get_point(np.linspace(0,1,100))
    
    if bPlot:
        plt.figure(num=1,clear=True)
        plt.plot(xhub,rhub,'b')
        plt.plot(xhub_ctrl_pts,rhub_ctrl_pts,'om')
        plt.plot(xshroud,rshroud,'r')
        plt.plot(xshroud_ctrl_pts,rshroud_ctrl_pts,'og')
        plt.axis('scaled')
        plt.xlabel('x-axial')
        plt.ylabel('r-radial')
        plt.show()
    return xhub, rhub, xshroud, rshroud

def create_passage_turbine(bPlot:bool=False):
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

def test_centrif3D_cut_te():
    hub = Centrif2D()

    hub.add_camber(alpha1=0,alpha2=70,stagger=35,x1=0.1,x2=0.98,aggressivity=(0.9,0.1))
    # hub.plot_camber()
    
    hub.add_le_thickness(0.02)
    hub.add_ps_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_ss_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_te_cut()
    hub.build(200)
    hub.plot()

def test_centrif3D_rounded_te():
    hub = Centrif2D()
    hub.add_camber(alpha1=0,alpha2=70,stagger=35,x1=0.1,x2=0.98,aggressivity=(0.9,0.1))
    # hub.plot_camber()
    
    hub.add_le_thickness(0.02)
    hub.add_ps_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_ss_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_te_radius(0.5,5,5,1)
    hub.build(200)
    # hub.plot()
    
    mid = Centrif2D()
    mid.add_camber(alpha1=0,alpha2=70,stagger=35,x1=0.1,x2=0.98,aggressivity=(0.9,0.1))
    # hub.plot_camber()
    
    mid.add_le_thickness(0.02)
    mid.add_ps_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    mid.add_ss_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    mid.add_te_radius(0.5,5,5,1)
    mid.build(200)
    
    tip = Centrif2D()
    tip.add_camber(alpha1=0,alpha2=70,stagger=35,x1=0.1,x2=0.98,aggressivity=(0.9,0.1))
    # hub.plot_camber()
    
    tip.add_le_thickness(0.02)
    tip.add_ps_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    tip.add_ss_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    tip.add_te_radius(0.5,5,5,1)
    tip.build(200)

    # Define hub and shroud
    xhub,rhub,xshroud,rshroud = create_passage_compressor()    
    comp = Centrif3D([hub,mid,tip],StackType.leading_edge)
    comp.add_hub(xhub,rhub)
    comp.add_shroud(xshroud,rshroud)
    comp.set_blade_position(0.01,0.95)
    comp.build(100,100)
    comp.plot()
    
def test_centrif_ellispe_te():
    hub = Centrif2D()

    hub.add_camber(alpha1=0,alpha2=70,stagger=35,x1=0.1,x2=0.98,aggressivity=(0.9,0.1))
    # hub.plot_camber()
    
    hub.add_le_thickness(0.02)
    hub.add_ps_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_ss_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_te_radius(0.5,5,5,1.2)
    hub.build(200)
    hub.plot()
    
if __name__=="__main__":
    test_centrif3D_rounded_te()