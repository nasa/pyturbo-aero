import sys
import numpy as np
from pyturbo.aero import Airfoil2D
from pyturbo.helper import exp_ratio

stator_hub = Airfoil2D(alpha1=0,alpha2=72,axial_chord=0.038,stagger=58) # This creates the camberline
# Building Leading Edge
stator_hub.add_le_thickness(0.04)
# Building the Pressure side 
ps_height = [0.0500,0.0200,-0.0100] # These are thicknesses 
stator_hub.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
stator_hub.add_ss_thickness(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)
stator_hub.match_le_thickness()
stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)

stator_hub.flow_guidance2(10)
# stator_hub.plot2D()
stator_mid = Airfoil2D(alpha1=10,alpha2=72,axial_chord=0.038,stagger=45) # This creates the camberline
# Building Leading Edge
stator_mid.add_le_thickness(0.06)
# Building the Pressure side 
ps_height = [0.0900,0.0500,0.0200] # These are thicknesses normalized by the chord
stator_mid.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.200, 0.2500, 0.1200, 0.1400]
stator_mid.add_ss_thickness(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)
stator_mid.match_le_thickness()
stator_mid.te_create(radius=0.0012,wedge_ss=3.5,wedge_ps=2.4)

stator_mid.flow_guidance2(10)

stator_tip = Airfoil2D(alpha1=5,alpha2=60,axial_chord=0.038,stagger=40) # This creates the camberline
# Building Leading Edge
stator_tip.add_le_thickness(0.04)
# Building the Pressure side 
ps_height = [0.0900,0.0500,0.0200] # These are thicknesses normalized by the chord
stator_tip.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.200, 0.2500, 0.2000, 0.1400]
stator_tip.add_ss_thickness(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)
stator_tip.match_le_thickness()
stator_tip.te_create(radius=0.001,wedge_ss=1.5,wedge_ps=2.4)

stator_tip.flow_guidance2(5)


from pyturbo.aero import Airfoil3D
from pyturbo.helper import StackType

stator3D = Airfoil3D(profileArray=[stator_hub,stator_mid,stator_tip],profile_loc=[0.0,0.5,1.0], height = 0.04)
stator3D.stack(StackType.centroid) # Stators are typically stacked with leading edge; rotors with centroid or trailing edge
stator3D.add_sweep(sweep_y=[0,-0.05,0.05], sweep_z=[0.0, 0.5, 1]) # Z =1 is blade tip, Z = 0 is blade hub. The units are in percentage 
stator3D.add_lean(leanX=[0,0.01,-0.02],leanZ=[0,0.5,1])
stator3D.build(nProfiles=20,num_points=160,trailing_edge_points=20)
# stator3D.plot3D()
print('check')
stator3D.export_solidworks('rotor')