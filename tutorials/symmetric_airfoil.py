import numpy as np
from pyturbo.aero import Airfoil2D
from pyturbo.helper import exp_ratio

# Stator
stator_hub = Airfoil2D(alpha1=0,alpha2=70,axial_chord=0.045,stagger=50)
# stator_hub.plot_camber()
stator_hub.add_le_thickness(0.05)
ps_height = [ 0.03,0.04,0.01]
stator_hub.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2,camberPercent=0.95)
ss_height = [ 0.2,0.24,0.18,0.14]
stator_hub.add_ss_thickness(thicknessArray=ss_height,expansion_ratio=1.2,camberPercent=0.8)
stator_hub.match_thickness(location='LE')
stator_hub.te_create(0.001,5,2)
stator_hub.add_ss_flow_guidance_2(0.8,10) # Useful for highspeed flows
stator_hub.plot2D()

stator_tip = Airfoil2D(alpha1=0,alpha2=70,axial_chord=0.045,stagger=50)
stator_tip.add_le_thickness(0.05)
ps_height = [ 0.03,0.04,0.01]
stator_tip.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2,camberPercent=0.95)
ss_height = [ 0.2,0.24,0.18,0.14]
stator_tip.add_ss_thickness(thicknessArray=ss_height,expansion_ratio=1.2,camberPercent=0.8)
stator_tip.match_thickness(location='LE')
stator_tip.te_create(0.001,5,2)
stator_tip.add_ss_flow_guidance_2(0.8,10) # Useful for highspeed flows

# Symmetric Airfoil
rotor_hub = Airfoil2D(alpha1=60,alpha2=60,axial_chord=0.045,stagger=0) # This creates the camberline
rotor_hub.plot_camber()

# Building Leading Edge
rotor_hub.add_le_thickness(0.06)
# Building the Pressure side
ps_height = [0.0300,0.0300,0.0300] # These are thicknesses
rotor_hub.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.0,camberPercent=1)
ss_height=[0.2400, 0.2000, 0.200, 0.2400]
rotor_hub.add_ss_thickness(thicknessArray=ss_height,expansion_ratio=1.0,camberPercent=1.0)
# stator_hub.match_le_thickness()
rotor_hub.match_thickness(location='LE')
rotor_hub.te_create_reversible(0.06)
rotor_hub.match_thickness(location='TE')
rotor_hub.plot2D()


rotor_mid = Airfoil2D(alpha1=58,alpha2=58,axial_chord=0.036,stagger=0) # This creates the camberline
rotor_mid.plot_camber()

# Building Leading Edge
rotor_mid.add_le_thickness(0.05)
# Building the Pressure side
ps_height = [0.0500,0.0500,0.0500] # These are thicknesses
rotor_mid.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.0,camberPercent=1)
ss_height=[0.2400, 0.2000, 0.200, 0.2400]
rotor_mid.add_ss_thickness(thicknessArray=ss_height,expansion_ratio=1.0,camberPercent=1.0)
# stator_hub.match_le_thickness()
rotor_mid.match_thickness(location='LE')
rotor_mid.te_create_reversible(0.05)
rotor_mid.match_thickness(location='TE')
rotor_mid.plot2D()


rotor_tip = Airfoil2D(alpha1=55,alpha2=55,axial_chord=0.035,stagger=0) # This creates the camberline
rotor_tip.plot_camber()

# Building Leading Edge
rotor_tip.add_le_thickness(0.04)
# Building the Pressure side
ps_height = [0.0500,0.0500,0.0500] # These are thicknesses
rotor_tip.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.0,camberPercent=1)
ss_height=[0.2400, 0.2000, 0.200, 0.2400]
rotor_tip.add_ss_thickness(thicknessArray=ss_height,expansion_ratio=1.0,camberPercent=1.0)
# stator_hub.match_le_thickness()
rotor_tip.match_thickness(location='LE')
rotor_tip.te_create_reversible(0.04)
rotor_tip.match_thickness(location='TE')
rotor_tip.plot2D()