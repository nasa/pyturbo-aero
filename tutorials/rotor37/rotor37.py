'''
    Notes: Rotor 37 is already an airfoil designed using a different design philosopy than this code. To use rotor37 we need to import the points into airfoil 3D or airfoil 2D directly. 

    Why use airofil2D or 3D? 
    1. This enables the generation of intermediate profile.
    2. If you wanted to change the hub and shroud curves, this would fit the geometry in between the curves. 
    3. It also allows you to position the blade anywhere along the hub, so if you wanted to increase or decrease the gap between stator and rotor

    Limitations:
    1. When converting to Airfoil3D, you don't have access to stacking. Stacking uses the camberline 
'''
from pyturbo.aero import Airfoil2D,Airfoil3D,Passage2D
import numpy as np
import os 
import matplotlib.pyplot as plt 

hub_pts = np.loadtxt("hub.csv", delimiter=",",skiprows=1)
tip_pts = np.loadtxt("tip.csv", delimiter=",",skiprows=1)

# From the excel file 7in seems to be the radius of the bottom profile (could be mistaken)
x = hub_pts[:,0]
ss_y = hub_pts[:,1]
ps_y = hub_pts[:,2]
new_hub = np.vstack([np.vstack([x[:-1],ps_y[:-1],7+0*ss_y[:-1]]).transpose(), np.flipud(np.vstack([x,ss_y,7+0*ss_y]).transpose())])


plt.figure(1,clear=True)
plt.plot(x,ss_y,x,ps_y)
plt.axis('equal')
plt.show()

# From the excel file 7.2in seems to be the radius of the top profile (could be mistaken)
x = tip_pts[:,0]
ss_y = tip_pts[:,1]
ps_y = tip_pts[:,2]
new_tip = np.vstack([np.vstack([x[:-1],ps_y[:-1],7.2+0*ss_y[:-1]]).transpose(), np.flipud(np.vstack([x,ss_y,7.2+0*ss_y]).transpose())])

os.makedirs('temp',exist_ok=True)
np.savetxt('temp/rotor37_0.txt',new_hub, delimiter=' ')
np.savetxt('temp/rotor37_1.txt',new_tip, delimiter=' ')
rotor37 = Airfoil3D.import_geometry(folder='temp',npoints=100,nspan=50,axial_chord=x.max(),ss_ps_split=len(ps_y)-1)
rotor37.plot3D()

