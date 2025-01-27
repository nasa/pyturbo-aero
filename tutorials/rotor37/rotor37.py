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
#%% R37 Blade by reading from hub and shroud profiles 
# hub_profile = np.loadtxt("hub.csv", delimiter=",",skiprows=1)
# tip_profile = np.loadtxt("tip.csv", delimiter=",",skiprows=1)

# # From the excel file 7in seems to be the radius of the bottom profile (could be mistaken)
# x = hub_profile[:,0]
# ss_y = hub_profile[:,1]
# ps_y = hub_profile[:,2]
# new_hub = np.vstack([np.vstack([x[:-1],ps_y[:-1],7+0*ss_y[:-1]]).transpose(), np.flipud(np.vstack([x,ss_y,7+0*ss_y]).transpose())])

# # plt.figure(1,clear=True)
# # plt.plot(x,ss_y,x,ps_y)
# # plt.axis('equal')
# # plt.show()

# # From the excel file 7.2in seems to be the radius of the top profile (could be mistaken)
# x = tip_profile[:,0]
# ss_y = tip_profile[:,1]
# ps_y = tip_profile[:,2]
# new_tip = np.vstack([np.vstack([x[:-1],ps_y[:-1],7.2+0*ss_y[:-1]]).transpose(), np.flipud(np.vstack([x,ss_y,7.2+0*ss_y]).transpose())])

# os.makedirs('temp',exist_ok=True)
# np.savetxt('temp/rotor37_0.txt',new_hub, delimiter=' ')
# np.savetxt('temp/rotor37_1.txt',new_tip, delimiter=' ')
# rotor37 = Airfoil3D.import_geometry(folder='temp',npoints=100,nspan=50,axial_chord=x.max(),ss_ps_split=len(ps_y)-1)
# rotor37.plot3D()

#%% R37 Blade by reading the 5 profiles from CSV 
# In this example the profiles are well defined and they have a radial component or z value.
files = ['R37_profile01.csv','R37_profile02.csv','R37_profile03.csv','R37_profile04.csv','R37_profile05.csv','R37_profile06.csv']
ss_pts = np.zeros(shape=(len(files),151,3)) # 5 profiles, each one is 300 points, 3 = x,y,z
ps_pts = np.zeros(shape=(len(files),151,3))
data = list()
for i,f in enumerate(files):
    data = np.loadtxt(f,skiprows=1,delimiter=',')
    ss_pts[i,:,0] = data[:151,0]
    ss_pts[i,:,1] = data[:151,1]
    ss_pts[i,:,2] = data[:151,2]
    ps_pts[i,:,0] = np.fliplr(data[150:,0])
    ps_pts[i,:,1] = np.fliplr(data[150:,1])
    ps_pts[i,:,2] = np.fliplr(data[150:,2])
data = np.vstack(data)


# Reading hub and shroud 
hub = np.loadtxt('hub_R37.dat')
shroud = np.loadtxt('hub_R37.dat')
passage = Passage2D(airfoil_array=[rotor37],spacing_array=[0])
passage.add_endwalls(zhub=hub[:,0],rhub=hub[:,1],zshroud=shroud[:,0],rshroud=shroud[:,1])
passage.blade_fit(x.min())
passage.plot2D()