from centrif3D_test import test_centrif3D_rounded_te
from pyturbo.aero.passage3D import Passage3D, PatternPairCentrif

def fullwheel_no_fillet():
    blade = test_centrif3D_rounded_te()
    wheel = Passage3D(blade)
    
    pair1 = PatternPairCentrif(0.96,0.5)
    pair2 = PatternPairCentrif(0.96,-0.5)
    
    wheel.add_pattern_pair(pair1)
    wheel.add_pattern_pair(pair2)
    
    wheel.build(nblades=12,bSplitter=True,hub_resolution=48)
    wheel.plot()
    
if __name__=='__main__':
    fullwheel_no_fillet()