# 2D Airfoil Design
There are two ways of using this code. 
1. If you need to design from scratch. 
2. If you already have a design

Designing from Scratch - Building the 2D geometry which will later be used to build the 3D Blade.
The code below shows how to construct a stator 
```
import numpy as np
from pyturbo.aero import *
from pyturbo.helper import *

stator_hub = Airfoil2D(alpha1=0,alpha2=72,axial_chord=0.038,stagger=58) # This creates the camberline
stator_hub.le_thickness_add(0.04)
ps_height = [0.0500,0.0200,-0.0100]
ps_height_loc = exp_ratio(1.2,len(ps_height)+2,0.95)
ps_height_loc = np.append(ps_height_loc,[1])
stator_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2)

ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
stator_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)
stator_hub.le_thickness_match()
stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)

stator_hub.flow_guidance2(10)
```

Below is the camberline plotted by calling.
`stator_hub.plot_camber()`

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/2D_design/camber_line.png)

Subsequently the design blade geometry with control points are plotted 
`stator_hub.plot2D()`

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/2D_design/2D_airfoil.png)

Second derivative for both suction and pressure sides
`stator_hub.plot_derivative2()`

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/2D_design/Deriv_2.png)

Plot of the stator with pitch to chord of 0.75
`stator_hub.plot2D_channel(0.75)`

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/2D_design/s_c.png)
