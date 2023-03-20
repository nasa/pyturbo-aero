'''
    This tutorial shows how you can export a geometry to STL
'''

import sys
sys.path.insert(0,'../../')
import numpy as np
from pyturbo.aero import airfoil2D
from pyturbo.helper import exp_ratio
from pyturbo.aero import airfoil3D, stack_type
import matplotlib.pyplot as plt 

stator2D = airfoil2D(alpha1=0,alpha2=72,axial_chord=0.038,stagger=58) # This creates the camberline

# Building Leading Edge
stator2D.le_thickness_add(0.04)
# Building the Pressure side 
ps_height = [0.0500,0.0200,-0.0100] # These are thicknesses 
stator2D.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
stator2D.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)
stator2D.le_thickness_match()
stator2D.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)

stator2D.flow_guidance2(10)


stator3D = airfoil3D(profileArray=[stator2D,stator2D,stator2D],profile_loc=[0.0,0.5,1.0], height = 0.02)
stator3D.stack(stack_type.leading_edge) # Stators are typically stacked with leading edge; rotors with centroid or trailing edge
# You can also use stack_type.centroid or stack_type.trailing_edge
stator3D.create_blade(20,160,20) 
# stator3D.center_le() # Centers the leading edge at (0,0); use this only if you are simulating a single blade. Makes creating planes for data extraction easier. 
stator3D.plot3D()

