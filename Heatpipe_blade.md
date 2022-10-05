# Design of Branching Paths for Heat Pipes
The idea behind this code is to design something simple in 2D that can be wrapped inside of a 3D blade. 
The steps towards making a heat pipe airfoil are shown below and examples codes are given. More examples can be found in the test folder. 
Step 1: Create the line path
Step 2: Create the cross section that follows the line path
Step 3: Apply it to the blade

## Useful import scripts

```
import numpy as np
from pyturbo.aero import *
from pyturbo.cooling import *
from pyturbo.helper import *
```

## Step 1: Creating a line path
### Periodic Loop
```
b = branch()
b.pattern_periodic_loop(4,152*.95,365.89,4)
b.plot_2D_path()
```
![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/HeatPipe/pathline_periodic_loop.png)

### Close Loop
```
b = branch()
b.pattern_close_loop(4,152*.95,365.89,4,return_gap=0)
b.plot_2D_path()
```

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/HeatPipe/pathline_close_loop_no_return.png)


### Close Loop with return loop
```
b = branch()
b.pattern_close_loop(4,152*.95,365.89,4,return_gap=20)
b.plot_2D_path()
```
![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/HeatPipe/pathline_close_loop_return.png)

        
## Step 2: Creating a simple Cross Section
### A Circle

```
cs = cross_section()
cs.create_circle(0.2)
cs.plot2D()
```
![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/HeatPipe/circle_cross_section.png)

### Rectangle
```
cs = cross_section()
cs.create_rectangle(0.2,0.2)
cs.plot2D()
```
![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/HeatPipe/rectangle_cross_section.png)

### Rectangle with fillets
```
cs = cross_section()
cs.create_rectangle(0.2,0.2,0.01)
cs.plot2D()
```
![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/HeatPipe/rectangle_cross_section_fillet.png)

## Step 3: Create the blade with heat pipe
### Periodic Loop
In this example the branch is wrapped around the entire blade surface.
```
cs = cross_section()
cs.create_circle(2)
# Design the heat pipe
b = branch()
## Note, you can use this one or
# b.pattern_periodic_loop(nloops=3,height=152*0.95,width=365.89,filletR=4)    
## Use this version (This function allows the estimation of the height and width to happen automatically)
b.setup_pattern_periodic_loop(nloops=3,filletR=4)
b.add_cross_section([cs,cs],[0,1]) # adds the same cross section to the start and finish of the branch          

## Code below is for debugging 
# b.plot_2D_path()        
# b.plot_sw_pathline()
# b.plot_pathline_3D()
## Import the airfoil
rotor = airfoil3D.import_geometry(folder='import_7%',axial_chord=124,span=114,ss_ps_split=105)
rotor.rotate(90)
rotor.spanwise_spline_fit()
rotor.spanwise_spline_fit_shell(shell_thickness=-3,smooth=0.1)
        
hp_airfoil = heatpipe_airfoil(rotor)        
# rotor.plot3D()

hp_airfoil.wrap_branch_shell(b,percent_span=[0.05,0.95], wall_dist=-3,rotate=0)
hp_airfoil.plot_heatpipe()
```

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/HeatPipe/periodic_loop.PNG)

### Close Loop Heat Pipe 
In this example the branch pattern is wrapped on either the suction side or pressure side.

```
cs = cross_section()
cs.create_circle(2)
# Suction side heat pipe
b_ss = branch()
## Note, you can use this one or
b_ss.setup_pattern_close_loop(nloops=3,filletR=4,return_gap=0)
b_ss.add_cross_section([cs,cs],[0,1]) # adds the same cross section to the start and finish of the branch  

b_ps = branch()
## Note, you can use this one or
b_ps.setup_pattern_close_loop(nloops=5,filletR=10,return_gap=0)
b_ps.add_cross_section([cs,cs],[0,1]) # adds the same cross section to the start and finish of the branch  

## Code below is for debugging 
# b.plot_2D_path()        
# b.plot_sw_pathline()
# b.plot_pathline_3D()
# Import the airfoil
rotor = airfoil3D.import_geometry(folder='import_7%',axial_chord=124,span=114,ss_ps_split=105)
rotor.rotate(90)
rotor.spanwise_spline_fit()
rotor.spanwise_spline_fit_shell(shell_thickness=-3,smooth=0.1)
rotor.plot3D()

hp_airfoil = heatpipe_airfoil(rotor)        

hp_airfoil.add_branch_shell(b_ss,axial_percent=[0.05, 0.9], percent_span=[0.05,0.95],wall_dist=-3,shell_type=branch_shell_type.suction_side)

hp_airfoil.add_branch_shell(b_ps,axial_percent=[0.12, 0.9], percent_span=[0.05,0.95],wall_dist=-3,shell_type=branch_shell_type.pressure_side)

hp_airfoil.plot_heatpipe()
```
### Close Loop with return

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/HeatPipe/pattern_close_loop_return.PNG)

### Close Loop with no returns
![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/HeatPipe/pattern_close_loop_no_return.PNG)

