'''
    Builds a centrifugal compressor and turbine profile in 2D
'''
import sys
sys.path.insert(0,'../../')
from pyturbo.aero import Centrif2D
from pyturbo.helper import exp_ratio

def test_centrif2D_cut_te():
    hub = Centrif2D()

    hub.add_camber(alpha1=20,alpha2=50,stagger=35,x1=0.2,x2=0.8)

    hub.add_le_thickness(0.01)
    hub.add_ss_thickness(thickness_array=[0.05,0.08,0.08,0.06])
    hub.add_ps_thickness(thickness_array=[0.05,0.08,0.08,0.06])
    hub.match_le_thickness()

    hub.add_te_cut(0.05)
    hub.build()
    hub.plot_camber()

    hub.plot()
    
if __name__=="__main__":
    test_centrif2D_cut_te()