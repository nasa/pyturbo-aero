from pyturbo.aero import Centrif, CentrifProfile, TrailingEdgeProperties
from pyturbo.helper import arc
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from copy import deepcopy


def build_endwalls(radius:float,inlet_hub_shroud_ratio:float,outlet_hub_shroud_ratio:float,x_stretch_factor:float, rhub_out:float, alpha_start:float=180,alpha_stop:float=270):
    shroud = arc(xc=0,yc=0,radius=radius,alpha_start=alpha_start,alpha_stop=alpha_stop)
    hub = arc(xc=0,yc=0,radius=radius/inlet_hub_shroud_ratio,alpha_start=alpha_start,alpha_stop=alpha_stop)

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

def compute_normals(x:npt.NDArray, y:npt.NDArray):
    # Compute first derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Compute normal vectors (perpendicular to tangent)
    length = np.hypot(dx, dy)
    nx = -dy / length
    ny = dx / length

    return nx, ny

def offset_curve(x:npt.NDArray, y:npt.NDArray, offset_distance:float):
    nx, ny = compute_normals(x, y)

    # Offset points along the normal direction
    x_offset = x + offset_distance * nx
    y_offset = y + offset_distance * ny

    return x_offset, y_offset


radius = 0.04 # meters
inlet_hub_shroud_ratio = 0.85
outlet_hub_shroud_ratio = 0.8
x_stretch_factor=1.3
rhub_out = 0.009 # meters
nblades = 6

hub1,shroud1 = build_endwalls(radius=radius,
                                inlet_hub_shroud_ratio=inlet_hub_shroud_ratio,
                                outlet_hub_shroud_ratio=outlet_hub_shroud_ratio,
                                x_stretch_factor=x_stretch_factor,
                                rhub_out=rhub_out,
                                alpha_start=270,alpha_stop=360)

hub2 = offset_curve(hub1[:,0],hub1[:,1],-radius*0.02)
shroud2 = offset_curve(shroud1[:,0],shroud1[:,1],radius*0.02)
hub2 = np.vstack(hub2).transpose()
shroud2 = np.vstack(shroud2).transpose()

# Lets see what it looks like
# plt.figure(num=1,clear=True, dpi=150)
# plt.plot(hub1[:,0],hub1[:,1],'k',linewidth=1.5,label='hub-1')
# plt.plot(shroud1[:,0],shroud1[:,1],'k',linewidth=1.5,label='shroud-1')
# plt.plot(hub2[:,0],hub2[:,1],'m',linewidth=1.5,label='hub-2')
# plt.plot(shroud2[:,0],shroud2[:,1],'m',linewidth=1.5,label='shroud-2')
# plt.xlabel('x-axial')
# plt.ylabel('radius')
# plt.axis('equal')
# plt.legend()
# plt.show()



cen = Centrif(blade_position=(0.0,1.0),use_mid_wrap_angle=True,
                  use_bezier_thickness=False,
                  use_ray_camber=False)
cen.add_hub(hub2[:,0],hub2[:,1])
cen.add_shroud(shroud2[:,0],shroud2[:,1])

TE_Cut = False
te_props = TrailingEdgeProperties(TE_Cut=TE_Cut,TE_Radius=0.05)
hub = CentrifProfile(percent_span=0,LE_Thickness=0.10,
                              trailing_edge_properties=te_props,
                              LE_Metal_Angle=-50,
                              TE_Metal_Angle=-30,
                              LE_Metal_Angle_Loc=0.1,
                              TE_Metal_Angle_Loc=0.9,
                              ss_thickness=[0.07,0.07,0.07,0.07,0.07],
                              ps_thickness=[0.07,0.07,0.07,0.07,0.07],
                              wrap_angle=-25, # this doesn't matter if you set use_mid_angle_wrap=True
                              wrap_displacements=[0.5,0.5,0.5,0.5],
                              wrap_displacement_locs=[0.3,0.4,0.5,0.7])

te_props = TrailingEdgeProperties(TE_Cut=TE_Cut,TE_Radius=0.05)

mid = CentrifProfile(percent_span=0.5,LE_Thickness=0.06,
                              trailing_edge_properties=te_props,
                              LE_Metal_Angle=-50,
                              TE_Metal_Angle=-30,
                              LE_Metal_Angle_Loc=0.1,
                              TE_Metal_Angle_Loc=0.9,
                              ss_thickness=[0.05,0.05,0.05,0.05,0.05],
                              ps_thickness=[0.05,0.05,0.05,0.05,0.05],
                              wrap_angle=-25, # this doesn't matter if you set use_mid_angle_wrap=True
                              wrap_displacements=[0.5,0.5,0.5,0.5],
                              wrap_displacement_locs=[0.3,0.4,0.5,0.7])

tip = CentrifProfile(percent_span=1,LE_Thickness=0.03,
                              trailing_edge_properties=te_props,
                              LE_Metal_Angle=-50,
                              TE_Metal_Angle=-30,
                              LE_Metal_Angle_Loc=0.1,
                              TE_Metal_Angle_Loc=0.9,
                              ss_thickness=[0.05,0.05,0.05,0.05],
                              ps_thickness=[0.05,0.05,0.05,0.05],
                              wrap_angle=-25, # this doesn't matter if you set use_mid_angle_wrap=True
                              wrap_displacements=[0.5,0.5,0.5,0.5],
                              wrap_displacement_locs=[0.3,0.4,0.5,0.7])

cen.add_profile(hub)
cen.add_profile(mid)
cen.add_profile(tip)
cen.add_splitter([hub,mid,tip],splitter_starts=0.55)

cen.build(npts_span=10, npts_chord=100,nblades=nblades)
# s_c_b2b,_ = cen.pitch_to_chord()
# cen.plot_camber()
# cen.plot_mp_profile()
# cen.plot()
cen.plot_fullwheel()
