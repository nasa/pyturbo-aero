import sys
import numpy as np
from pyturbo.aero import Airfoil3D
from pyturbo.helper import create_passage

def test_passage():

    hub,shroud = create_passage(PR=2.4,phi1=0.7, 
                       M1_rel=0.6, HTR1=0.5,
                       deHaller=1, outlet_yaw=-64, 
                       blade_circulation=0.6, tip_clearance=0.01,
                       P01=1, T01=300, mdot=5)


if __name__ == "__main__":
    test_passage()