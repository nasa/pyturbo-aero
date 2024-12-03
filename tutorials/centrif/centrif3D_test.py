from pyturbo.aero import Centrif2D, Centrif3D
from pyturbo.helper import ellispe, StackType, bezier
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt

def create_passage_compressor(bPlot:bool=False):
    """Creates the hub and shroud curves for the compressor 

    Args:
        bPlot (bool, optional): Plots the hub and shroud curves . Defaults to False.

    Returns:
        _type_: _description_
    """
    # Shroud is defined using a thickness offset from the hub to construct a spline
    xhub_ctrl_pts = np.array([0.0, 0.02, 0.08, 0.11, 0.12, 0.12])
    rhub_ctrl_pts = np.array([0.0, 0.0, 0.004, 0.06, 0.09, 0.10]) + 0.02
    hub = bezier(xhub_ctrl_pts,rhub_ctrl_pts)
    
    xshroud_ctrl_pts = np.array([0.0, 0.02, 0.05, 0.09, 0.11, 0.11])
    rshroud_ctrl_pts = np.array([0.03, 0.03, 0.04, 0.055, 0.09, 0.10]) + 0.02
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

def test_Centrif3D_LE_cut_te():
    hub = Centrif2D()

    hub.add_camber(alpha1=0,alpha2=70,stagger=35,x1=0.1,x2=0.98,aggressivity=(0.9,0.1))
    # hub.plot_camber()
    
    hub.add_le_thickness(0.02)
    hub.add_ps_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_ss_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_te_cut()
    hub.build(200)
    
    xhub,rhub,xshroud,rshroud = create_passage_compressor()    
    comp = Centrif3D([hub,hub,hub],StackType.leading_edge)
    comp.add_hub(xhub,rhub)
    comp.add_shroud(xshroud,rshroud)
    comp.set_blade_position(0.01,0.95)
    comp.build(100,100)
    return comp 

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
    return comp

def test_centrif_splitter():
    xhub,rhub,xshroud,rshroud = create_passage_compressor()      
    hub = Centrif2D()
    hub.add_camber(alpha1=0,alpha2=70,stagger=35,x1=0.1,x2=0.98,aggressivity=(0.9,0.1))
    hub.add_le_thickness(0.02)
    hub.add_ps_thickness(thickness_array=[0.02,0.02,0.02,0.02])
    hub.add_ss_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_te_radius(0.5,5,5,1)
    hub.build(100)
    # hub.plot()
    
    comp = Centrif3D([hub,hub,hub],StackType.trailing_edge)
    comp.add_hub(xhub,rhub)
    comp.add_shroud(xshroud,rshroud)
    comp.set_blade_position(0.01,0.95)
    comp.build(100,100)
    # comp.plot()
    
    splitter_hub = Centrif2D(splitter_camber_start=0.4)
    splitter_hub.add_camber(alpha1=0,alpha2=70,stagger=35,x1=0.1,x2=0.98,aggressivity=(0.9,0.1))
    splitter_hub.add_le_thickness(0.02)
    splitter_hub.add_ps_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    splitter_hub.add_ss_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    splitter_hub.add_te_radius(0.5,5,5,1)
    splitter_hub.build(100)

    splitter = Centrif3D([splitter_hub,splitter_hub,splitter_hub],StackType.trailing_edge)
    splitter.add_hub(xhub,rhub)
    splitter.add_shroud(xshroud,rshroud)
    splitter.set_blade_position(0.01,0.95)
    splitter.build(100,100,comp)
    # splitter.plot()

    return comp,splitter

def wavy_centrif3D():
    blade = test_centrif3D_rounded_te()
    t = np.linspace(0,2*np.pi,100)
    sin_wave1 = 0.2*np.sin(4*t/(2*np.pi))
    cos_wave1 = 0.2*np.cos(4*t/(2*np.pi))
    sin_wave2 = 0.4*np.sin(3*t/(2*np.pi))
    
    blade.LE_Waves = sin_wave1
    blade.SS_Waves = cos_wave1
    blade.TE_Waves = sin_wave2
    blade.build(100,100)
    blade.plot()

def test_centrif_fillet():
    # Design the Fillet
    ps_fillet1 = bezier([0, 0, 0.3, 0.8, 1],
                        [1, 0.6, 0.2, 0, 0])  # LE
    
    ps_fillet2 = bezier([0, 0, 0.2, 0.8, 1],
                        [1, 0.6, 0.2, 0, 0])  # Mid: PS
    
    ps_fillet3 = bezier([0, 0, 0.5, 0.8, 1],
                        [1, 0.6, 0.1, 0.0, 0]) # TE: PS
    
    ss_fillet2 = bezier([0, 0, 0.2, 0.7, 1],
                         [1, 0.6, 0.2, 0, 0])  # Mid: SS
    # ps_fillet1.plot2D()
    # plt.show()
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
    
    # comp.add_hub_bezier_fillet(ps=ps_fillet1,ps_loc=0,r=0.002) # Radius is 5% of the height from hub to shroud
    # comp.add_hub_bezier_fillet(ps=ps_fillet2,ps_loc=0.5)
    # comp.add_hub_bezier_fillet(ps=ps_fillet3,ps_loc=1)
    # comp.add_hub_bezier_fillet(ss=ss_fillet2,ss_loc=0.5)
    comp.build(100,100)
    # comp.plot_x_slice(2)

    return comp 
    
if __name__=="__main__":
    # test_Centrif3D_LE_rounded_te()
    # splitter = test_centrif_splitter()
    blade = wavy_centrif3D()
    