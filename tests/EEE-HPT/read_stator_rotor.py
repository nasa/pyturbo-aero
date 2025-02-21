from typing import List, Tuple
import pyiges,os
import numpy.typing as npt
from pyiges import examples
import numpy as np
import matplotlib.pyplot as plt 
import pickle
from scipy.interpolate import BSpline, splrep, splev

'''
    Read blade into suction and pressure sides 
'''

import matplotlib.pyplot as plt 

def get_ss_ps(points:npt.NDArray):
    """Reads in the points and splits it by suction and pressure sides

    Args:
        points (npt.NDArray): points as an array

    Returns:
        (Tuple) containing:
        
            **Suction** (npt.NDArray): Numpy array of points for the suction side
            **Pressure** (npt.NDArray): Numpy array of points for the pressure side
    """
    
    def look_for_slope_change(dydx:npt.NDArray, start_indx:int, forward_search:bool=True):
        if not forward_search: # backwards search 
            for i in range(start_indx,0,-1):
                if np.sign(dydx[start_indx]) != np.sign(dydx[i]):
                    return i
        else:
            
    dydx = np.gradient(points[:,1], points[:,0])

    # le_indx1 = -1
    # le_indx2 = -1
    te_indx1 = -1 
    te_indx2 = -1 
    
    # min_indx = np.argmin(np.abs(points[:,0] - points[:,0].min()))

    # i = min_indx
    # for i in range(0,min_indx):
    #     if np.sign(dydx[min_indx]) != np.sign(dydx[i]):
    #         le_indx1 = i
    #         break
    
    # for i in range(min_indx,points.shape[0]):
    #     if np.sign(dydx[min_indx]) != np.sign(dydx[i]):
    #         le_indx2 = i
    #         break
    
    # if le_indx1==-1:
    #     le_indx = le_indx2
    # elif le_indx2 == -1:
    #     le_indx = le_indx1
    # else:
    #     if points[le_indx1,0]>points[le_indx2,0]:
    #         le_indx = le_indx2
    #     else:
    #         le_indx = le_indx1
    
    max_indx = np.argmin(np.abs(points[:,0] - points[:,0].max()))
    for i in range(max_indx,0,-1):
        if np.sign(dydx[max_indx]) != np.sign(dydx[i]):
            te_indx1 = i
            break
    
    for i in range(max_indx,points.shape[0]):
        if np.sign(dydx[max_indx]) != np.sign(dydx[i]):
            te_indx2 = i
            break

    if te_indx1==-1:
        te_indx = te_indx2
    elif te_indx2 == -1:
        te_indx = te_indx1
    else:
        if points[te_indx1,0]>points[te_indx2,0]:
            te_indx = te_indx1
        else:
            te_indx = te_indx2

    # Build SS and PS from Indices
    # di = te_indx-le_indx
    # new_pts = np.roll(points,-le_indx,axis=0)
    # ss = new_pts[:di,:]
    # ps = new_pts[di:,:]
    
    n = len(points)
    ss = points[0:te_indx,:]
    ps = points[te_indx:,:]
    return ss,ps 

    
def plot_blade(blade:List[Tuple[npt.NDArray,npt.NDArray]]):
    fig = plt.figure(num=1,clear=True,dpi=150)
    ax = fig.add_subplot(111)
    for ss,ps in blade:
        ax.plot(ss[:,0],ss[:,1],color='red',label='ss')
        ax.plot(ps[:,0],ps[:,1],color='blue',label='ps')
        ax.axis('scaled')
        
    ax.legend()
    ax.set_title("Plot of Suction and pressure sides")
    plt.show()

def Process_HubShroud_IGES():
    iges_case = pyiges.read('case.igs')
    iges_hub = pyiges.read('hub.igs')

    # print an invidiual entity (boring)
    curve = iges_case.items[0].to_geomdl()
    case_pts = np.array(curve.evalpts)
    
    curve = iges_hub.items[0].to_geomdl()
    curve.delta=0.001
    hub_pts = np.array(curve.evalpts)
    os.makedirs('csv',exist_ok=True)
    os.makedirs('plots',exist_ok=True)
    np.savetxt('csv/shroud.csv',case_pts,fmt="%f",delimiter=',',header='x,r,theta')
    np.savetxt('csv/hub.csv',hub_pts,fmt="%f",delimiter=',',header='x,r,theta')

    plt.figure(num=0)
    plt.plot(case_pts[:,0],case_pts[:,1])
    plt.plot(hub_pts[:,0],hub_pts[:,1])
    plt.axis('scaled')
    plt.savefig('plots/flowpath.png',transparent=None,dpi=150)


    pickle.dump({'Hub':hub_pts,'Shroud':case_pts},open('hub_shroud.pkl','wb'))

def Process_StatorRotor_IGES():
    os.makedirs('csv',exist_ok=True)
    os.makedirs('plots',exist_ok=True)
    # load an example impeller
    iges_rotor1 = pyiges.read('hpt_stator1.igs')
    iges_stator1 = pyiges.read('hpt_rotor1.igs')

    iges_rotor2 = pyiges.read('hpt_stator2.igs')
    iges_stator2 = pyiges.read('hpt_rotor2.igs')
    curve_delta=0.001
    # Stage 1 
    stator_pts1 = list(); indx = 1
    plt.figure(num=2,clear=True)
    for i in range(2,7):
        curve = iges_stator1.items[i].to_geomdl()
        curve.delta=curve_delta
        points = np.array(curve.evalpts); n = points.shape[0]
        ss = points[:n,:]; ps = points[n:,:]
        stator_pts1.append(points)
        np.savetxt(f'csv/stator1_{indx}.csv',stator_pts1[-1],fmt="%f",delimiter=',',header='x,rtheta,r')
        plt.plot(ss[:,0],ss[:,1],'.',label='ss')
        plt.plot(ps[:,0],ps[:,1],'.',label='ps')
        # plt.plot(stator_pts1[-1][:,0],stator_pts1[-1][:,1],'.')
        indx+=1
    plt.axis('scaled')
    plt.title('Stator')
    plt.savefig('plots/Stator1.png',transparent=None,dpi=150)
    
    # print an invidiual entity (boring)
    rotor_pts1 = list(); indx = 1
    plt.figure(num=1,clear=True)
    for i in range(2,7):
        curve = iges_rotor1.items[i].to_geomdl()
        curve.delta=curve_delta
        points = np.array(curve.evalpts)
        rotor_pts1.append(points)
        np.savetxt(f'csv/rotor1_{indx}.csv',rotor_pts1[-1],fmt="%f",delimiter=',',header='x,rtheta,r')
        plt.plot(rotor_pts1[-1][:,0],rotor_pts1[-1][:,1],'.')
        indx+=1
    plt.axis('scaled')
    plt.title('Rotor')
    plt.savefig('plots/Rotor1.png',transparent=None,dpi=150)
    

    # Stage 2
    stator_pts2 = list(); indx = 1 
    plt.figure(num=2,clear=True)
    for i in range(2,6):
        curve = iges_stator2.items[i].to_geomdl()
        curve.delta=curve_delta
        points = np.array(curve.evalpts)
        stator_pts2.append(points)
        np.savetxt(f'csv/stator2_{indx}.csv',stator_pts2[-1],fmt="%f",delimiter=',',header='x,rtheta,r')
        plt.plot(stator_pts2[-1][:,0],stator_pts2[-1][:,1],'.')
        indx+=1
    plt.axis('scaled')
    plt.title('Stator')
    plt.savefig('Stator2.png',transparent=None,dpi=150)
    
    # print an invidiual entity (boring)
    rotor_pts2 = list(); indx = 1
    plt.figure(num=1,clear=True)
    for i in range(2,7):
        curve = iges_rotor2.items[i].to_geomdl()
        curve.delta=curve_delta
        points = np.array(curve.evalpts)
        rotor_pts2.append(points)
        np.savetxt(f'csv/rotor2_{indx}.csv',rotor_pts2[-1],fmt="%f",delimiter=',',header='x,rtheta,r')
        plt.plot(rotor_pts2[-1][:,0],rotor_pts2[-1][:,1],'.')
        indx+=1
    plt.axis('scaled')
    plt.title('Rotor')
    plt.savefig('plots/Rotor2.png',transparent=None,dpi=150)
    
    pickle.dump({
                    'Stator1':stator_pts1,
                    'Rotor1':rotor_pts1,
                    'Stator2':stator_pts2,
                    'Rotor2':rotor_pts2,
                 },open('stator_rotor.pkl','wb'))

def ComputeCamberline():
    data = pickle.load(open('stator_rotor.pkl','rb'))
    
    def profiles_to_ss_ps(profile:List[npt.NDArray]):
        blade = []
        for pts in profile:
            ss,ps = get_ss_ps(pts)
            blade.append((ss,ps))
        return blade
    stator1 = profiles_to_ss_ps(data['Stator1'])
    
    rotor1 = profiles_to_ss_ps(data['Rotor1'])
    plot_blade(rotor1)
    stator2 = profiles_to_ss_ps(data['Stator2'])
    rotor2 = profiles_to_ss_ps(data['Rotor2'])
    
     
if __name__ == "__main__":
    # Process_HubShroud_IGES()
    # Process_StatorRotor_IGES()
    ComputeCamberline()