'''
    Builds a centrifugal compressor and turbine profile in 2D
'''
import sys
from pyturbo.aero import Centrif2D
from pyturbo.helper import exp_ratio, ellispe
import numpy as np 

def test_centrif2D_cut_te():
    hub = Centrif2D()

    hub.add_camber(alpha1=0,alpha2=70,stagger=35,x1=0.1,x2=0.98,aggressivity=(0.9,0.1))
    # hub.plot_camber()
    
    hub.add_le_thickness(0.02)
    hub.add_ps_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_ss_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_te_cut()
    hub.build(200)
    hub.plot()

def test_centrif2D_rounded_te():
    hub = Centrif2D()

    hub.add_camber(alpha1=0,alpha2=70,stagger=35,x1=0.1,x2=0.98,aggressivity=(0.9,0.1))
    # hub.plot_camber()
    
    hub.add_le_thickness(0.02)
    hub.add_ps_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_ss_thickness(thickness_array=[0.02,0.03,0.02,0.02])
    hub.add_te_radius(0.5,5,5,1.2)
    hub.build(200)
    hub.plot()
    
def test_ellispe():
    a = ellispe(0.5,0.5,1.2,1,180,-180)
    a.get_point(np.linspace(0,1,20))
    a.plot()
    
if __name__=="__main__":
    # test_centrif2D_cut_te()
    # test_centrif2D_rounded_te()
    test_ellispe()