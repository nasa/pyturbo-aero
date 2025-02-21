import numpy as np
from pyturbo.aero import Centrif, CentrifProfile, TrailingEdgeProperties
from pyturbo.helper import create_passage
from copy import deepcopy

def test_blade_and_passage(bSplitter:bool=False,TE_Cut:bool=False,TE_Radius:float=0.015):
    """Creates a centrif blade with or without splitter

    Args:
        bSplitter (bool, optional): Adds splitter. Defaults to False.
        TE_Cut (bool, optional): Cut TE instead of rounded. Defaults to False.
    """
    # Create the passage using code translated from https://whittle.digital/2024/Radial_Compressor_Designer/
    hub,shroud,V3,T3,P3,Ma3,eta_ts,eta_now,Power,RPM,Alrel1_deg,Alrel2_deg = create_passage(PR=2.4,phi1=0.7, 
                       M1_rel=0.6, HTR1=0.5,
                       deHaller=1, outlet_yaw=-64, 
                       blade_circulation=0.6, tip_clearance=0.01,
                       P01=1, T01=300, mdot=5)
    cen = Centrif(blade_position=(0.05,0.96))
    cen.add_hub(hub[:,0],hub[:,1])
    cen.add_shroud(shroud[:,0],shroud[:,1])
    
    LE_Thickness = 0.02
    LE_Metal_Angle = -50
    TE_Metal_Angle = 50
    LE_Metal_Angle_Loc = 0.15
    TE_Metal_Angle_Loc = 0.85
    wrap_angle=-30
    wrap_displacements = [0.1, 0.3]
    wrap_displacement_locs = [0.4,0.8]
    ss_thickness = [0.04,0.03,0.03,0.03,0.03,0.03,0.03]
    ps_thickness = [0.02,0.03,0.03,0.03,0.03,0.03,0.03]
    
    te_props = TrailingEdgeProperties(TE_Cut=TE_Cut,TE_Radius=TE_Radius)
    hub = CentrifProfile(percent_span=0,LE_Thickness=LE_Thickness,
                                   trailing_edge_properties=te_props,
                                   LE_Metal_Angle=LE_Metal_Angle,
                                   TE_Metal_Angle=TE_Metal_Angle,
                                   LE_Metal_Angle_Loc=LE_Metal_Angle_Loc,
                                   TE_Metal_Angle_Loc=TE_Metal_Angle_Loc,
                                   ss_thickness=ss_thickness,
                                   ps_thickness=ps_thickness,
                                   wrap_angle=wrap_angle,
                                   wrap_displacements=wrap_displacements,
                                   wrap_displacement_locs=wrap_displacement_locs)
    mid = deepcopy(hub); mid.percent_span=0.5
    tip = deepcopy(hub); tip.percent_span=1.0
    
    cen.add_profile(hub)
    cen.add_profile(mid)
    cen.add_profile(tip)

    if bSplitter:
        # When creating splitter profiles, only the thicknesses matters. 
        # Percent span doesn't matter
        # Splitter will follow the camber line of the main blade
        ss_thickness = [0.03,0.03,0.03]
        ps_thickness = [0.03,0.03,0.03]
        shub = deepcopy(hub); shub.ss_thickness = ss_thickness; shub.ps_thickness = ps_thickness
        smid = deepcopy(mid); smid.ss_thickness = ss_thickness; shub.ps_thickness = ps_thickness
        stip = deepcopy(tip); stip.ss_thickness = ss_thickness; stip.ps_thickness = ps_thickness

        splitter_profiles = [shub, smid, stip]
        cen.add_splitter(splitter_profiles=splitter_profiles,splitter_starts=[0.5,0.45,0.4])
    cen.build(npts_span=50, npts_chord=150,nblades=6,nsplitters=6)
    cen.plot_mp_profile()
    cen.plot()
    # cen.plot_camber()
    # cen.plot_fullwheel()

if __name__ == "__main__":
    # test_blade_and_passage(bSplitter=False,TE_Cut=False)
    # test_blade_and_passage(bSplitter=True,TE_Cut=False)
    test_blade_and_passage(bSplitter=True,TE_Cut=False,TE_Radius=0.01)