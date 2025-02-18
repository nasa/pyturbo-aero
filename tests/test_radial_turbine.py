import numpy as np
from typing import Tuple, List
from pyturbo.aero import Centrif, CentrifProfile, TrailingEdgeProperties
from pyturbo.helper import create_passage, arc
from copy import deepcopy
import numpy.typing as npt

def build_endwalls(radius:float,inlet_hub_shroud_ratio:float,outlet_hub_shroud_ratio:float,x_stretch_factor:float, rhub_out:float) -> Tuple[npt.NDArray, npt.NDArray]:
    """Build the hub and shroud curve for a general radial blade geometry

    Args:
        radius (float): _description_
        inlet_hub_shroud_ratio (float): _description_
        outlet_hub_shroud_ratio (float): _description_
        x_stretch_factor (float): _description_
        rhub_out (float): _description_

    Returns:
        Tuple containing: 
        - hub (np.ndarray): hub curve
        - shroud (np.ndarray): shroud curve
    """
    shroud = arc(xc=0,yc=0,radius=radius,alpha_start=180,alpha_stop=270)
    hub = arc(xc=0,yc=0,radius=radius/inlet_hub_shroud_ratio,alpha_start=180,alpha_stop=270)

    [xhub,rhub] = hub.get_point(np.linspace(0,1,100))
    [xshroud,rshroud] = shroud.get_point(np.linspace(0,1,100))

    rhub = rhub/outlet_hub_shroud_ratio
    xhub*=x_stretch_factor
    xshroud*=x_stretch_factor
    
    hub = np.vstack([xhub,rhub]).transpose()
    shroud = np.vstack([xshroud, rshroud]).transpose()
    shroud[:,1] += -hub[:,1].min() + rhub_out
    hub[:,1] += -hub[:,1].min() + rhub_out
    xshroud += -xhub.min()
    xhub += -xhub.min()
    return hub,shroud
    
def compute_normals(x:npt.NDArray, y:npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    # Compute first derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Compute normal vectors (perpendicular to tangent)
    length = np.hypot(dx, dy)
    nx = -dy / length
    ny = dx / length
    
    return nx, ny

def offset_curve(x, y, offset_distance):
    nx, ny = compute_normals(x, y)

    # Offset points along the normal direction
    x_offset = x + offset_distance * nx
    y_offset = y + offset_distance * ny
    
    return x_offset, y_offset

    ## Note: To have ADS be able to cut the geometry and create a mesh. We have to make the blades bigger so design it with a slightly larger radius and slightly more inlet_hub_shroud_ratio and more outlet_shroud_ratio 
    
    

def test_blade_and_passage(bSplitter:bool=False,TE_Cut:bool=False,TE_Radius:float=0.015):
    """Creates a centrif blade with or without splitter

    Args:
        bSplitter (bool, optional): Adds splitter. Defaults to False.
        TE_Cut (bool, optional): Cut TE instead of rounded. Defaults to False.
    """
    # Create the passage using code translated from https://whittle.digital/2024/Radial_Compressor_Designer/
    radius = 0.04
    hub1,shroud1 = build_endwalls(radius=radius,
                                inlet_hub_shroud_ratio=0.8,outlet_hub_shroud_ratio=0.8,
                                x_stretch_factor=1.1,rhub_out=0.009)
    hub2 = offset_curve(hub1[:,0],hub1[:,1],-radius*0.02)
    shroud2 = offset_curve(shroud1[:,0],shroud1[:,1],radius*0.02)    
    hub2 = np.vstack(hub2).transpose()
    shroud2 = np.vstack(shroud2).transpose()


    cen = Centrif(blade_position=(0.05,0.96))
    cen.add_hub(hub1[:,0],hub1[:,1])
    cen.add_shroud(shroud1[:,0],shroud1[:,1])
    
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


    mid = CentrifProfile(percent_span=0,LE_Thickness=LE_Thickness,
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
    mid.percent_span=0.5
    cen.add_profile(mid)
   
    cen.build(npts_span=50, npts_chord=150,nblades=6,nsplitters=6)
    cen.plot_mp_profile()
    cen.plot()
    # cen.plot_camber()
    # cen.plot_fullwheel()

if __name__ == "__main__":
    # test_blade_and_passage(bSplitter=False,TE_Cut=False)
    # test_blade_and_passage(bSplitter=True,TE_Cut=False)
    test_blade_and_passage(bSplitter=True,TE_Cut=False,TE_Radius=0.01)