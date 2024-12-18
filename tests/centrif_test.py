import numpy as np
from pyturbo.aero import Centrif, CentrifProfile, DefineCircularTE, DefineCutTE
from pyturbo.helper import create_passage

def test_passage():

    hub,shroud,V3,T3,P3,Ma3,eta_ts,eta_now,Power,RPM,Alrel1_deg,Alrel2_deg = create_passage(PR=2.4,phi1=0.7, 
                       M1_rel=0.6, HTR1=0.5,
                       deHaller=1, outlet_yaw=-64, 
                       blade_circulation=0.6, tip_clearance=0.01,
                       P01=1, T01=300, mdot=5)
    cen = Centrif()
    cen.add_hub(hub[:,0],hub[:,1])
    cen.add_shroud(shroud[:,0],shroud[:,1])
    
    LE_Thickness = 0.02
    LE_Metal_Angle = -70
    TE_Metal_Angle = 15
    LE_Metal_Angle_Loc = 0.1
    TE_Metal_Angle_Loc = 0.85
    warp_angle=-30
    warp_displacements = [0.3, -0.3]
    warp_displacement_locs = [0.37,0.64]
    ss_thickness = np.array([0.025,0.02,0.02,0.02,0.03,0.02,0.03])
    ps_thickness = np.array([0.02,0.02,0.02,0.02,0.03,0.02,0.03])
    
    te_props = DefineCircularTE(radius=0.018,major_axis_scale=1,SS_WedgeAngle=5,PS_WedgeAngle=3)
    cen.add_profile(CentrifProfile(percent_span=0,LE_Thickness=LE_Thickness,
                                   trailing_edge_properties=te_props,
                                   LE_Metal_Angle=LE_Metal_Angle,TE_Metal_Angle=TE_Metal_Angle,
                                   LE_Metal_Angle_Loc=LE_Metal_Angle_Loc,TE_Metal_Angle_Loc=TE_Metal_Angle_Loc,
                                   ss_thickness=ss_thickness,
                                   ps_thickness=ps_thickness,
                                   warp_angle=warp_angle,warp_displacements=warp_displacements,
                                   warp_displacement_locs=warp_displacement_locs))
    LE_Thickness = 0.02*0.7
    te_props = DefineCircularTE(radius=0.018*0.5,major_axis_scale=1,SS_WedgeAngle=5,PS_WedgeAngle=3)
    cen.add_profile(CentrifProfile(percent_span=0.5,LE_Thickness=LE_Thickness,
                                   trailing_edge_properties=te_props,
                                   LE_Metal_Angle=LE_Metal_Angle,TE_Metal_Angle=TE_Metal_Angle,
                                   LE_Metal_Angle_Loc=LE_Metal_Angle_Loc,TE_Metal_Angle_Loc=TE_Metal_Angle_Loc,
                                   ss_thickness=ss_thickness*0.5,
                                   ps_thickness=ps_thickness*0.5,   
                                   warp_angle=warp_angle,warp_displacements=warp_displacements,
                                   warp_displacement_locs=warp_displacement_locs))
    scale = 0.4
    LE_Thickness = 0.02*scale
    te_props = DefineCircularTE(radius=0.018*scale,major_axis_scale=1,SS_WedgeAngle=5,PS_WedgeAngle=3)
    cen.add_profile(CentrifProfile(percent_span=1.0,LE_Thickness=LE_Thickness,
                                   trailing_edge_properties=te_props,
                                   LE_Metal_Angle=LE_Metal_Angle,TE_Metal_Angle=TE_Metal_Angle,
                                   LE_Metal_Angle_Loc=LE_Metal_Angle_Loc,TE_Metal_Angle_Loc=TE_Metal_Angle_Loc,
                                   ss_thickness=ss_thickness*scale,
                                   ps_thickness=ps_thickness*scale,
                                   warp_angle=warp_angle,warp_displacements=warp_displacements,
                                   warp_displacement_locs=warp_displacement_locs))
    cen.build()
    cen.plot_rx_profile()
    cen.plot()
    # cen.plot_rx_profile()
    # cen.plot_camber()
if __name__ == "__main__":
    test_passage()