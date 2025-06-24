# Import scripts
from typing import List
import numpy as np
from pyturbo.aero import Airfoil2D, Airfoil3D, Passage2D
from pyturbo.helper import exp_ratio, bezier, pw_bezier2D,StackType

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

stator3D = Airfoil3D(profileArray=[stator_hub,stator_mid,stator_tip], profile_loc=[0.0,0.5,1.0], height = 0.04)
stator3D.stack(StackType.leading_edge) # stators are typically stacked with leading edge; stators with centroid or trailing edge
stator3D.add_sweep(sweep_y=[0,-0.05,0.01], sweep_z=[0.0, 0.5, 1]) # Z =1 is blade tip, Z = 0 is blade hub. The units are in percentage
stator3D.add_lean(leanX=[0,0.1,0.05], leanZ=[0,0.5,1])
stator3D.build(nProfiles=20,num_points=160,trailing_edge_points=20)
# stator3D.plot3D()
# Rotor 
### Hub Profile
rotor_axial_chord = 0.030
rotor_hub = Airfoil2D(alpha1=35,alpha2=65,axial_chord=rotor_axial_chord,stagger=38)
rotor_hub.add_le_thickness(0.04)

ps_height = [0.0500,0.0200,-0.0100] # These are thicknesses
rotor_hub.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.2400, 0.2200, 0.2000, 0.1800]
rotor_hub.add_ss_thickness(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

rotor_hub.match_le_thickness()
rotor_hub.te_create(radius=0.001,wedge_ss=3.5,wedge_ps=2.4)
rotor_hub.add_ss_flow_guidance_2(s_c=0.75,n=10)
# rotor_hub.plot2D()

rotor_mid = Airfoil2D(alpha1=30,alpha2=67,axial_chord=0.038,stagger=35)
rotor_mid.add_le_thickness(0.04)

ps_height = [0.0500,0.0200,-0.0100] # These are thicknesses
rotor_mid.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.2400, 0.2200, 0.2000, 0.1800]
rotor_mid.add_ss_thickness(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

rotor_mid.match_le_thickness()
rotor_mid.te_create(radius=0.001,wedge_ss=3.5,wedge_ps=2.4)
rotor_mid.add_ss_flow_guidance_2(s_c=0.75,n=10)
# rotor_mid.plot2D()

rotor_tip = Airfoil2D(alpha1=30,alpha2=65,axial_chord=0.037,stagger=32)
rotor_tip.add_le_thickness(0.03)

ps_height = [0.0500,0.0200,-0.0100] # These are thicknesses
rotor_tip.add_ps_thickness(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.2400, 0.2200, 0.2000, 0.1800]
rotor_tip.add_ss_thickness(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

rotor_tip.match_le_thickness()
rotor_tip.te_create(radius=0.001,wedge_ss=3.5,wedge_ps=2.4)
rotor_tip.add_ss_flow_guidance_2(s_c=0.7,n=10)
# rotor_tip.plot2D()

#%% Rotor 3D
rotor3D = Airfoil3D(profileArray=[rotor_hub,rotor_mid,rotor_tip],profile_loc=[0.0,0.5,1.0], height = 0.04)
rotor3D.stack(StackType.trailing_edge) # stators are typically stacked with leading edge; stators with centroid or trailing edge
rotor3D.add_sweep(sweep_y=[0,-0.05,0.05], sweep_z=[0.0, 0.5, 1]) # Z =1 is blade tip, Z = 0 is blade hub. The units are in percentage
rotor3D.add_lean(leanX=[0,0.01,-0.02],leanZ=[0,0.5,1])
rotor3D.build(nProfiles=20,num_points=60,trailing_edge_points=20)
# rotor3D.plot3D()

def match_end_slope(bezier1:bezier, x:List[float],y:List[float]):
    """Creates another bezier curve that matches the slope at the end of bezier 1

    Args:
        bezier1 (bezier):
        x (List[float]): Bezier control points
        y (List[float]): Bezier control points

    Returns:
        bezier: new bezier curve containing control points x,y with extra points interjected at the 2nd index x[1],y[1] to match the end slope of bezier1
    """
    # Look at the last 2 points of the previous bezier curve. These 2 points control the slope
    x1 = bezier1.x[-2:]
    y1 = bezier1.y[-2:]

    dx = np.diff(x1)[0]    # Find the spacing
    dy = np.diff(y1)[0]
    x2 = x[0]+dx
    y2 = y[0]+dy

    d1 =2; d2=1 # Set these values so loop is executed.
    while d1>d2:    # Loop exists so that new points fall in between existing points
        # Add extra control point (x2,y2) into list, need to adjust control point
        dx = x[0]-x2
        dy = y[0]-y2
        d1 = np.sqrt((dx)**2 + (dy)**2)
        d2 = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)

        if d1>d2:
            x2 += dx*0.2 # Reduce both by 80%, this keeps the slope the same
            y2 += dy*0.2

    x.insert(1,x2)
    y.insert(1,y2)

    b = bezier(x,y)
    return b

# Endwalls Parameters
rtip = 0.25 # meters
hub_tip_ratio = 0.8
rhub = rtip*hub_tip_ratio
stator_rotor_gap = 0.010

rhub_expansion_coeff1 = [1.0,0.98,0.97] # Stator
zhub_expansion_coeff1 = [0.25,0.75]

# This makes the flowpath going from inlet to stator leading edge
rhub_points1 = [rhub, rhub]  # 1.5x Stator Inlet, stator_inlet, stator_mid
zhub_points1 = [-1.5*stator_hub_axial_chord,0]
hub_bezier1 = bezier(zhub_points1,rhub_points1)

# Flowpath from stator leading edge to trailing edge
rhub_points2 = [rhub]
zhub_points2 = [0]
rhub_points2.append(rhub*rhub_expansion_coeff1[0])                      # Mid bezier control point
zhub_points2.append(stator_hub_axial_chord*zhub_expansion_coeff1[0]) # type: ignore

rhub_points2.append(rhub*rhub_expansion_coeff1[1])                      # End bezier control point
zhub_points2.append(stator_hub_axial_chord*zhub_expansion_coeff1[1]) # type: ignore

rhub_points2.append(rhub*rhub_expansion_coeff1[2])                      # End bezier point
zhub_points2.append(stator_hub_axial_chord+stator_rotor_gap*0.5) # type: ignore

rhub_points2 = np.array(rhub_points2)
zhub_points2 = np.array(zhub_points2)

hub_bezier2 = match_end_slope(hub_bezier1,zhub_points2.tolist(),rhub_points2.tolist())

# Mid stator-rotor gap to rotor_te + stator-rotor gap
rhub_expansion_coeff2 = [0.99,0.98,1.0,1.0] # Rotor
zhub_expansion_coeff2 = [0.5,0.80]
rhub_points3 = [rhub_points2[-1]]
zhub_points3 = [zhub_points2[-1]] # This will be adjusted at the end

rhub_points3.append(rhub_points2[-1]*rhub_expansion_coeff2[0]) # Rotor Inlet
zhub_points3.append(zhub_points2[-1]+stator_rotor_gap*0.5)

rhub_points3.append(rhub_points2[-1]*rhub_expansion_coeff2[1]) # Rotor Mid
zhub_points3.append(zhub_points2[-1]+stator_rotor_gap*0.5+rotor_axial_chord*zhub_expansion_coeff2[0])

rhub_points3.append(rhub_points2[-1]*rhub_expansion_coeff2[2]) # Rotor TE
zhub_points3.append(zhub_points2[-1]+stator_rotor_gap*0.5+rotor_axial_chord*zhub_expansion_coeff2[1])

rhub_points3.append(rhub_points2[-1]*rhub_expansion_coeff2[3]) # Rotor TE + stator_rotor_gap
zhub_points3.append(zhub_points2[-1]+stator_rotor_gap*0.5+rotor_axial_chord+stator_rotor_gap*0.5)

rhub_points3 = np.array(rhub_points3)
zhub_points3 = np.array(zhub_points3)

hub_bezier3 = match_end_slope(hub_bezier2,zhub_points3.tolist(),rhub_points3.tolist())

rhub_points4 = [rhub_points3[-1], rhub_points3[-1]]
zhub_points4 = [zhub_points3[-1], zhub_points3[-1]+2.5*rotor_axial_chord]
hub_bezier4 = match_end_slope(hub_bezier3,zhub_points4,rhub_points4)

rshroud_expansion_coeff1 = [1.04,1.05,1.06] # Stator
zshroud_expansion_coeff1 = [0.25,0.9]

# This makes the flowpath going from inlet to stator leading edge
rshroud_points1 = [rtip, rtip]  # 1.5x Stator Inlet, stator_inlet, stator_mid
zshroud_points1 = [-1.5*stator_hub_axial_chord,0]
shroud_bezier1 = bezier(zshroud_points1,rshroud_points1)

# Flowpath from stator leading edge to trailing edge
rshroud_points2 = [rtip]
zshroud_points2 = [0]
rshroud_points2.append(rtip*rshroud_expansion_coeff1[0])                      # Mid bezier control point
zshroud_points2.append(stator_hub_axial_chord*0.5) # type: ignore

rshroud_points2.append(rtip*rshroud_expansion_coeff1[1])                      # End bezier control point
zshroud_points2.append(stator_hub_axial_chord*0.5+stator_hub_axial_chord*0.5*zshroud_expansion_coeff1[0]) # type: ignore

rshroud_points2.append(rtip*rshroud_expansion_coeff1[2])                      # End bezier point
zshroud_points2.append(stator_hub_axial_chord+stator_rotor_gap*0.5) # type: ignore

rshroud_points2 = np.array(rshroud_points2)
zshroud_points2 = np.array(zshroud_points2)

shroud_bezier2 = match_end_slope(shroud_bezier1,zshroud_points2.tolist(),rshroud_points2.tolist())

# Mid stator-rotor gap to rotor_te + stator-rotor gap
rshroud_expansion_coeff2 = [1.0,1.0,1.0,1.0] # Rotor
zshroud_expansion_coeff2 = [0.40,0.8]
rshroud_points3 = [rshroud_points2[-1]]
zshroud_points3 = [zshroud_points2[-1]] # This will be adjusted at the end

rshroud_points3.append(rshroud_points2[-1]*rshroud_expansion_coeff2[0]) # Rotor Inlet
zshroud_points3.append(zshroud_points2[-1]+stator_rotor_gap*0.5)

rshroud_points3.append(rshroud_points2[-1]*rshroud_expansion_coeff2[1]) # Rotor Mid
zshroud_points3.append(zshroud_points2[-1]+stator_rotor_gap*0.5+rotor_axial_chord*zshroud_expansion_coeff2[0])

rshroud_points3.append(rshroud_points2[-1]*rshroud_expansion_coeff2[2]) # Rotor TE
zshroud_points3.append(zshroud_points2[-1]+stator_rotor_gap*0.5+rotor_axial_chord*zshroud_expansion_coeff2[1])

rshroud_points3.append(rshroud_points2[-1]*rshroud_expansion_coeff2[3]) # Rotor TE + stator_rotor_gap
zshroud_points3.append(zshroud_points2[-1]+stator_rotor_gap*0.5+rotor_axial_chord+stator_rotor_gap*0.5)

rshroud_points3 = np.array(rshroud_points3)
zshroud_points3 = np.array(zshroud_points3)

shroud_bezier3 = match_end_slope(shroud_bezier2,zshroud_points3.tolist(),rshroud_points3.tolist())

rshroud_points4 = [rshroud_points3[-1], rshroud_points3[-1]]
zshroud_points4 = [zshroud_points3[-1], zshroud_points3[-1]+2.5*rotor_axial_chord]
shroud_bezier4 = match_end_slope(shroud_bezier3,zshroud_points4,rshroud_points4)

hub_bezier = pw_bezier2D([hub_bezier1,hub_bezier2,hub_bezier3,hub_bezier4])
shroud_bezier = pw_bezier2D([shroud_bezier1,shroud_bezier2,shroud_bezier3,shroud_bezier4])

import copy
stator_adjusted = copy.deepcopy(stator3D)
rotor_adjusted = copy.deepcopy(rotor3D)

stator_adjusted.center_le()
stator_adjusted.flip_x()
stator_adjusted.rotate(cx=0,cy=0,angle=90)

rotor_adjusted.center_le()
rotor_adjusted.flip()
# rotor_adjusted.rotate(cx=0,cy=0,angle=90)

passage = Passage2D([stator_adjusted,rotor_adjusted],[stator_rotor_gap])

zhub,rhub = hub_bezier.get_point(np.linspace(0,1,100))
zshroud,rshroud = shroud_bezier.get_point(np.linspace(0,1,100))
passage.add_endwalls(zhub,rhub,zshroud,rshroud)
passage.blade_fit(0)

passage.plot2D()
passage.plot3D()
print('check')
