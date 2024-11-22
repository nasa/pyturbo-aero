from centrif3D_test import test_centrif3D_rounded_te, test_centrif_splitter
from pyturbo.aero.passage3D import Passage3D, PatternPairCentrif

def fullwheel_no_fillet():
    blade = test_centrif3D_rounded_te()
    wheel = Passage3D(blade)
        
    pair1 = PatternPairCentrif(0.96,0.5)
    pair2 = PatternPairCentrif(0.96,-0.5)
    pair3 = PatternPairCentrif(1,-0.5)
    pair4 = PatternPairCentrif(1,0.5)
    
    wheel.add_pattern_pair(pair1)
    wheel.add_pattern_pair(pair2)
    wheel.add_pattern_pair(pair3)
    wheel.add_pattern_pair(pair4)
    
    wheel.build(nblades=12,hub_resolution=48)
    wheel.plot()

def fullwheel_splitter():
    blade = test_centrif3D_rounded_te()
    splitter = test_centrif_splitter()
    wheel = Passage3D(blade)
        
    pair1 = PatternPairCentrif(0.96,0.5)
    pair2 = PatternPairCentrif(0.96,-0.5)
    pair3 = PatternPairCentrif(1,-0.5)
    pair4 = PatternPairCentrif(1,0.5)
    
    wheel.add_pattern_pair(pair1)
    wheel.add_pattern_pair(pair2)
    wheel.add_pattern_pair(pair3)
    wheel.add_pattern_pair(pair4)
    
    wheel.add_splitter(splitter)
    
    wheel.build(nblades=12,hub_resolution=48)
    wheel.plot()
    
if __name__=='__main__':
    # fullwheel_no_fillet()
    fullwheel_splitter()