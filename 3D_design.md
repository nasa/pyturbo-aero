# 3D Design

The 3D design of the blade begins by stacking 2D Designs together to form a 3D geometry.
## Useful import scripts

```
import numpy as np
from pyturbo.aero import *
from pyturbo.helper import *
```


## Simple 3D Blade 
The code below shows an example of how to build a 3D Blade from 2D designs located at the hub and tip

```
# Hub Geometry
stator_hub = Airfoil2D(alpha1=0,alpha2=72,axial_chord=0.038,stagger=58)
stator_hub.le_thickness_add(0.04)
ps_height = [0.0500,0.0200,-0.0100]
ps_height = [0.0500,0.0200,-0.0100]
stator_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2) 

ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
stator_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
stator_hub.le_thickness_match()
stator_hub.flow_guidance2(10)

# Tip Geometry
stator_tip = Airfoil2D(alpha1=5,alpha2=72,axial_chord=0.036,stagger=56)
stator_tip.le_thickness_add(0.04)
ps_height = [0.0500,0.0200,-0.0100]
ps_height_loc = exp_ratio(1.2,len(ps_height)+2,0.95)
ps_height_loc = np.append(ps_height_loc,[1])
stator_tip.ps_thickness_add(thicknessArray=ps_height,camberPercent=ps_height_loc)

ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
stator_tip.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

stator_tip.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
stator_tip.le_thickness_match()
stator_tip.flow_guidance2(6)

# Begin 3D design
stator3D = airfoil3D([stator_hub,stator_tip],[0,1],0.05)
stator3D.stack(stack_type.trailing_edge)
# stator3D.add_lean([0, 0.05, 1], [0,0.5,1])
stator3D.build(100,80,20)
stator3D.plot3D()
```

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/3D_design/stator_3D.png)

## Wavy Blade
Wavy code from Paht's thesis has been included in the design tools. This allows for optimizing sinusoidal profiles on the surfaces of the blade.
The code is very similar to the 3D blade. A few lines were added to apply the waves in the spanwise direction.
Note: Variables with names LE and TE wave angle control the angle modification of the leading and trailing edges. 
These modifications are done perpendicular to the leading and trailing edge angle.


### Wavy Blade with constant Trailing Edge Radius
For blade designs that require a constant trailing edge radius. The example below is useful.

```
# Wavy Geometry
stator3D_wavy = airfoil_wavy(stator3D)
stator3D_wavy.spanwise_spline_fit()
t = np.linspace(0,1,1000)
ss_ratio = 2*span*np.cos(3*math.pi*t)
ps_ratio = 2*span*np.cos(3*math.pi*t)
te_ratio = 0.5*span*np.cos(3*math.pi*t)
le_ratio = (ss_ratio+ps_ratio)/2.0
ssratio_wave = wave_control(ss_ratio)  # These are spanwise waves
psratio_wave = wave_control(ps_ratio)  
teratio_wave = wave_control(te_ratio)
leratio_wave = wave_control(le_ratio)

LE_wave_angle = leratio_wave*0.0
TE_wave_angle = leratio_wave*0.0

stator3D_wavy.stretch_thickness_chord_te(SSRatio=ssratio_wave.get_wave(0,1),PSRatio=psratio_wave.get_wave(0.2,0.8,True),
    LERatio=leratio_wave.get_wave(0,1),TERatio=teratio_wave.get_wave(0,1),LE_wave_angle=LE_wave_angle,TE_wave_angle=TE_wave_angle,TE_smooth=0.90)

stator3D.plot3D_ly(only_blade=True)
```

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/3D_design/stator3D_wavy_const_te_radius.PNG)


### Wavy Blade without a constant Trailing Edge Radius
If trailing edge radius is not a problem and you want to explore designs that can vary the radius.

```
# Begin 3D design
span = 0.05
stator3D = airfoil3D([stator_hub,stator_tip],[0,1],0.05)
stator3D.stack(stack_type.trailing_edge)
stator3D.add_lean([0, 0.05, 0], [0,0.5,1])
stator3D.build(100,100,20)
# stator3D.plot3D()

# Wavy Geometry
stator3D_wavy = airfoil_wavy(stator3D)
stator3D_wavy.spanwise_spline_fit()
t = np.linspace(0,1,1000)
ss_ratio = 2*span*np.cos(3*math.pi*t)
ps_ratio = 2*span*np.cos(3*math.pi*t)
te_ratio = 0.5*span*np.cos(3*math.pi*t)
le_ratio = (ss_ratio+ps_ratio)/2.0
ssratio_wave = wave_control(ss_ratio)  # These are spanwise waves
psratio_wave = wave_control(ps_ratio)  
teratio_wave = wave_control(te_ratio)
leratio_wave = wave_control(le_ratio)

LE_wave_angle = leratio_wave*0.0
TE_wave_angle = leratio_wave*0.0

stator3D_wavy.stretch_thickness_chord(SSRatio=ssratio_wave.get_wave(0,1),PSRatio=psratio_wave.get_wave(0.2,0.8,True),
    LERatio=leratio_wave.get_wave(0,1),TERatio=teratio_wave.get_wave(0,1),LE_wave_angle=LE_wave_angle,TE_wave_angle=TE_wave_angle,TE_smooth=0.5)

stator3D_wavy.plot3D(only_blade=True)
```

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/3D_design/stator3D_wavy_variable_te_radius.PNG)

### Whisker Blade
This was a challenge from vikram. Whiskers sort maintain their cross sectional area along the span, the area doesn't deviate much. Can a blade do the same thing?
Of course it can. All I have to do is remove one of the parameters, in this case it's the suction side ratio. Removing this and allowing it to vary can create a whisker-like blade geometry.

```
# Begin 3D design
span = 0.05
stator3D = airfoil3D([stator_hub,stator_tip],[0,1],0.05)
stator3D.stack(stack_type.trailing_edge)
stator3D.add_lean([0, 0.05, 0], [0,0.5,1])
stator3D.build(100,100,20)
stator3D.plot3D()

# Wavy Geometry
stator3D_wavy = airfoil_wavy(stator3D)
stator3D_wavy.spanwise_spline_fit()
t = np.linspace(0,1,1000)
ps_ratio = 2*span*np.cos(3*math.pi*t)
te_ratio = 0.5*span*np.cos(3*math.pi*t)
le_ratio = 2*span*np.cos(3*math.pi*t)
psratio_wave = wave_control(ps_ratio)  
teratio_wave = wave_control(te_ratio)
leratio_wave = wave_control(le_ratio)

LE_wave_angle = leratio_wave*0.0
TE_wave_angle = leratio_wave*0.0

stator3D_wavy.whisker_blade(PSRatio=psratio_wave.get_wave(0.2,0.8,True),
    LERatio=leratio_wave.get_wave(0,1),TERatio=teratio_wave.get_wave(0,1),LE_wave_angle=LE_wave_angle,TE_wave_angle=TE_wave_angle,TE_smooth=0.9)

stator3D_wavy.plot3D_ly(only_blade=True)
```

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/3D_design/stator3D_whisker.PNG)

### Creating a Shell of any distance Paht +1 Solidworks 0 
Sometimes you need to create a shell. This is useful for designing internal cooling passages or where to run your heat pipe inside the blade

```
span = 0.05
stator3D = airfoil3D([stator_hub,stator_tip],[0,1],0.05)
stator3D.stack(stack_type.trailing_edge)
stator3D.add_lean([0, 0.05, 0], [0,0.5,1])
stator3D.build(100,100,20)
# stator3D.plot3D()
[ss_x_new,ss_y_new,ps_x_new,ps_y_new] = stator3D.get_shell_2D(percent_span,shell_thickness)
stator3D.plot_shell_2D(0.2,-0.002)
```

![](https://gitlab.grc.nasa.gov/lte-turbo/pyturbo/-/tree/master/pyturbo/wiki/3D_design/stator3D_shell.png)


### Importing a geometry
This is an example of how to import the 7% TE Thickness EEE design
```
rotor = airfoil3D.import_geometry(folder='import_7%',axial_chord=124,span=114,ss_ps_split=105)
rotor.rotate(90)
rotor.spanwise_spline_fit()
rotor.spanwise_spline_fit_shell(shell_thickness=-3,smooth=0.1)
rotor.plot3D()
```

When you import geometries, you can manipulate it: add lean, fit it into a row/channel etc



