from pyturbo.aero import AirfoilWavy, Airfoil2D, Airfoil3D
from pyturbo.helper import exp_ratio, bezier, pw_bezier2D,StackType
import numpy as np 

stator_hub_axial_chord = 0.040
#This creates the camberline
stator_hub = Airfoil2D(alpha1=0,alpha2=72,axial_chord=stator_hub_axial_chord,stagger=52)
stator_hub.add_le_thickness(0.04)

ps_height = [0.0500,0.0200,-0.0100] # These are thicknesses
stator_hub.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
stator_hub.add_ss_thickness(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

stator_hub.match_le_thickness()
stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
stator_hub.add_ss_flow_guidance_2(s_c=0.75,n=10)
# stator_hub.plot2D()

stator_mid = Airfoil2D(alpha1=0,alpha2=70,axial_chord=stator_hub_axial_chord*0.96,stagger=52)
stator_mid.add_le_thickness(0.04)

ps_height = [0.0500,0.0200,-0.0100] # These are thicknesses
stator_mid.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
stator_mid.add_ss_thickness(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

stator_mid.match_le_thickness()
stator_mid.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
stator_mid.add_ss_flow_guidance_2(s_c=0.75,n=10)
# stator_mid.plot2D()

stator_tip = Airfoil2D(alpha1=0,alpha2=68,axial_chord=stator_hub_axial_chord*0.95,stagger=53)
stator_tip.add_le_thickness(0.03)

ps_height = [0.0500,0.0200,-0.0100] # These are thicknesses
stator_tip.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
stator_tip.add_ss_thickness(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

stator_tip.match_le_thickness()
stator_tip.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
stator_tip.add_ss_flow_guidance_2(s_c=0.75,n=10)
# stator_tip.plot2D()


# stator3D = Airfoil3D(profileArray=[stator_hub,stator_mid,stator_tip], profile_loc=[0.0,0.5,1.0], height = 0.04)
# stator3D.stack(StackType.centroid) # stators are typically stacked with leading edge; stators with centroid or trailing edge
# stator3D.add_sweep(sweep_y=[0,-0.05,0.01], sweep_z=[0.0, 0.5, 1]) # Z =1 is blade tip, Z = 0 is blade hub. The units are in percentage
# stator3D.add_lean(leanX=[0,0.1,0.05], leanZ=[0,0.5,1])
# stator3D.build(nProfiles=20,num_points=160,trailing_edge_points=20)
# stator3D.plot3D()

## Suction side scaling
t = np.linspace(0,10*np.pi,100)
ssratio = 0.05*np.sin(t)
leratio = 0.05*np.cos(t/4)
teratio = 0.05*np.cos(t)
psratio = ssratio*0
lewave_angle = 0*ssratio
tewave_angle = 0*ssratio
stator3D_Wavy1 = AirfoilWavy(profileArray=[stator_hub,stator_mid,stator_tip],profile_loc=[0,0.5,1],height=0.04)
stator3D_Wavy1.stack(StackType.centroid) # stators are typically stacked with leading edge; stators with centroid or trailing edge
stator3D_Wavy1.build(nProfiles=100,num_points=160,trailing_edge_points=20)
stator3D_Wavy1.stretch_thickness_chord(ssratio,psratio,leratio,teratio,lewave_angle,tewave_angle)
# stator3D_Wavy1.plot3D()


t = np.linspace(0,10*np.pi,100)
ssratio = 0.05*np.sin(t)
leratio = 0.05*np.cos(t/4)
teratio = 0.05*np.cos(t)
psratio = ssratio*0
lewave_angle = 0*ssratio
tewave_angle = 0*ssratio

stator3D_Wavy1 = AirfoilWavy(profileArray=[stator_hub,stator_mid,stator_tip],profile_loc=[0,0.5,1],height=0.04)
stator3D_Wavy1.stack(StackType.centroid) # stators are typically stacked with leading edge; stators with centroid or trailing edge
stator3D_Wavy1.build(nProfiles=100,num_points=160,trailing_edge_points=20)
stator3D_Wavy1.stretch_thickness_chord(ssratio,psratio,leratio,teratio,lewave_angle,tewave_angle,TE_smooth=0.85)
# stator3D_Wavy1.plot3D()

stator3D_Wavy2 = AirfoilWavy(profileArray=[stator_hub,stator_mid,stator_tip],profile_loc=[0,0.5,1],height=0.04)
stator3D_Wavy2.stack(StackType.centroid) # stators are typically stacked with leading edge; stators with centroid or trailing edge
stator3D_Wavy2.build(nProfiles=100,num_points=160,trailing_edge_points=20)
stator3D_Wavy2.stretch_thickness_chord_te(ssratio,psratio,leratio,teratio,lewave_angle,tewave_angle,TE_smooth=0.85)
# stator3D_Wavy2.plot3D()

stator3D_Wavy3 = AirfoilWavy(profileArray=[stator_hub,stator_mid,stator_tip],profile_loc=[0,0.5,1],height=0.04)
stator3D_Wavy3.stack(StackType.centroid) # stators are typically stacked with leading edge; stators with centroid or trailing edge
stator3D_Wavy3.build(nProfiles=100,num_points=160,trailing_edge_points=20)
stator3D_Wavy3.whisker_blade(leratio,teratio,psratio,lewave_angle,tewave_angle,TE_smooth=0.85)
# stator3D_Wavy3.plot3D()
