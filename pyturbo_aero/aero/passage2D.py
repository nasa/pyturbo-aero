from typing import List
import numpy as np
from scipy.special import comb
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
import math
import json
from scipy.special import comb
from ..helper import spline_type, pspline
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class passage2D:
    """Passage2D fits 3D blades inside of a channel
    """
    def __init__(self,airfoil_array,spacing_array,periodicity_array):
        '''
        Initialize the passage with airfoils and spacing array
            airfoil_array = array of airfoil3D objects
            spacing_array = space between the each pair of blades
        Airfoils are spaced and then passage is moved to fit the airfoils
        call add_hub and add_shroud to include the hub and shroud definitions
        '''
        self.airfoils=airfoil_array
        self.spacing=spacing_array
        self.periodicity_array = periodicity_array


    def add_endwalls(self,zhub:List[float],rhub:List[float],zshroud:List[float],rshroud:List[float],zhub_control:List[float]=[],rhub_control:List[float]=[],zshroud_control:List[float]=[],rshroud_control:List[float]=[]):
        """Adds the endwalls

        Args:
            zhub (List[float]): points defining the hub axial coordinate
            rhub (List[float]): points defining the hub radial coordinate
            zshroud (List[float]): points defining the shroud axial coordinate
            rshroud (List[float]): points defining the shroud radial coordinate
            zhub_control (List[float], optional): bezier axial control points for the hub. Defaults to [].
            rhub_control (List[float], optional): bezier radial control points for the hub. Defaults to [].
            zshroud_control (List[float], optional):  bezier axial control points for the shroud. Defaults to [].
            rshroud_control (List[float], optional): bezier radial control points for the shroud. Defaults to [].
        """
        self.hub_spline = PchipInterpolator(zhub,rhub)
        self.shroud_spline = PchipInterpolator(zshroud,rshroud)
        self.zhub = zhub
        self.rhub = rhub
        self.zshroud = zshroud
        self.rshroud = rshroud
        self.zhub_control = zhub_control
        self.rhub_control = rhub_control
        self.zshroud_control = zshroud_control
        self.rshroud_control = rshroud_control


    def curve_endwalls(self,curve_zr=[]):
        """Curves the endwalls in the merdional plane. This is used to create radial machines. Distance from shroud to hub will be maintained

        Args:
            curve_zr (List): List of points [[1,2],[3,4]]

        Currently not implemented
        """
        pass


    def blade_fit(self,xBladeStart:float):
        """Fits the blade within the channel

        Args:
            xBladeStart (float): axial location of where the leading edge of the first blade starts within the channel
        """
        
        # Shift the blades within the channel
        ## Take the blade starting location, should be (0,0)
        x_start = self.airfoils[0].shft_xss[0,0] # when rotated the y becomes the x
        # z_endwall_start = self.zhub[0]
        dx = x_start + xBladeStart
        self.airfoils[0].shift(-dx,0)

        # Space out the airfoils from each other
        for i in range(1,len(self.airfoils)):
            x_end = self.airfoils[i-1].shft_xss[0,-1]
            x_start = self.airfoils[i].shft_xss[0,0]
            dx = x_start -x_end - self.spacing[i-1]
            self.airfoils[i].shift(-dx,0)


        # Scale the blade between the endwalls
        for i in range(len(self.airfoils)):
            [_,npoints] = self.airfoils[i].shft_yss.shape            
            zhub_ss = np.zeros(npoints)
            zshroud_ss = np.zeros(npoints)
            zhub_ps = np.zeros(npoints)
            zshroud_ps = np.zeros(npoints)
            for j in range(npoints):
                x = self.airfoils[i].shft_xss[0,j]
                zhub_ss[j] = self.hub_spline(x)

                x = self.airfoils[i].shft_xps[0,j]
                zhub_ps[j] = self.hub_spline(x)

            for j in range(npoints):
                x = self.airfoils[i].shft_xss[-1,j]
                zshroud_ss[j] = self.shroud_spline(x)

                x = self.airfoils[i].shft_xps[-1,j]
                zshroud_ps[j] = self.shroud_spline(x)

            self.airfoils[i].scale_zps(zhub_ps,zshroud_ps)
            self.airfoils[i].scale_zss(zhub_ss,zshroud_ss)            
            # Offset the blade past the hub and shroud curves by adding two additional profiles to hub and tip
            nprofiles,npoints = self.airfoils[i].shft_xss.shape
            shft_xss = self.airfoils[i].shft_xss
            shft_yss = self.airfoils[i].shft_yss
            shft_zss = self.airfoils[i].shft_zss

            shft_xps = self.airfoils[i].shft_xps
            shft_yps = self.airfoils[i].shft_yps
            shft_zps = self.airfoils[i].shft_zps

            self.airfoils[i].shft_xss = np.zeros(shape=(nprofiles+2,npoints))
            self.airfoils[i].shft_yss = np.zeros(shape=(nprofiles+2,npoints))
            self.airfoils[i].shft_zss = np.zeros(shape=(nprofiles+2,npoints))

            self.airfoils[i].shft_xps = np.zeros(shape=(nprofiles+2,npoints))
            self.airfoils[i].shft_yps = np.zeros(shape=(nprofiles+2,npoints))
            self.airfoils[i].shft_zps = np.zeros(shape=(nprofiles+2,npoints))

            self.airfoils[i].shft_xss[1:-1,:] = shft_xss
            self.airfoils[i].shft_yss[1:-1,:] = shft_yss
            self.airfoils[i].shft_zss[1:-1,:] = shft_zss

            self.airfoils[i].shft_xps[1:-1,:] = shft_xps
            self.airfoils[i].shft_yps[1:-1,:] = shft_yps
            self.airfoils[i].shft_zps[1:-1,:] = shft_zps

            self.airfoils[i].shft_xss[0,:] = shft_xss[0,:]
            self.airfoils[i].shft_yss[0,:] = shft_yss[0,:]
            self.airfoils[i].shft_zss[0,:] = shft_zss[0,:] - 0.04

            self.airfoils[i].shft_xps[0,:] = shft_xps[0,:]
            self.airfoils[i].shft_yps[0,:] = shft_yps[0,:]
            self.airfoils[i].shft_zps[0,:] = shft_zps[0,:] - 0.04

            self.airfoils[i].shft_xss[-1,:] = shft_xss[-1,:]
            self.airfoils[i].shft_yss[-1,:] = shft_yss[-1,:]
            self.airfoils[i].shft_zss[-1,:] = shft_zss[-1,:] + 0.04

            self.airfoils[i].shft_xps[-1,:] = shft_xps[-1,:]
            self.airfoils[i].shft_yps[-1,:] = shft_yps[-1,:]
            self.airfoils[i].shft_zps[-1,:] = shft_zps[-1,:] + 0.04
    
    def plot2D_channel(self):     
        """Plot the blades within the channel
            uses plotly 
        """  
        marker=dict(size=0.001, color="red", colorscale='Viridis')
        # Plot the channel
        zhub = np.linspace(min(self.zhub),max(self.zhub),500)
        rhub = self.hub_spline(zhub)

        rshroud = self.shroud_spline(zhub)
        zshroud = zhub
        
        fig = go.Figure(data=go.Scatter3d(x=zhub, y=rhub, z=zhub*0,  marker=marker,line=dict(color='black',width=2)))
        fig.add_trace(go.Scatter3d(x=zshroud, y=rshroud, z=zshroud*0,  marker=marker,line=dict(color='black',width=2)))
        # Plot the control points
        marker=dict(size=0.1, color="red", colorscale='Viridis')

        if (not self.zhub_control):
            fig.add_trace(go.Scatter3d(x=self.zhub_control, y=self.rhub_control, z=self.zhub_control*0,  marker=marker,line=dict(color='red',width=2)))

        if (not self.rshroud_control):
            fig.add_trace(go.Scatter3d(x=self.zshroud_control, y=self.rshroud_control, z=self.zshroud_control*0,  marker=marker,line=dict(color='red',width=2)))

        
        fig.update_layout(showlegend=False,scene= dict(aspectmode='manual',aspectratio=dict(x=1, y=1, z=1)))
        fig.show()

    def check_replace_max(self,max_prev,max_new):
        if max_new>max_prev:
            return max_new
        else:
            return max_prev

    def check_replace_min(self,min_prev,min_new):
        if min_new<min_prev:
            return min_new
        else:
            return min_prev

    def plot3D_ly(self):
        """Plot the blades within the channel 
            uses plotly
        """
        marker=dict(size=0.001, color="red", colorscale='Viridis')

        theta_max=0.0; zmax=0.0; rmax=0.0
        theta_min=0.0; zmin=0.0; rmin=0.0
        fig = go.Figure()
        # Plot the blades 
        for airfoil in self.airfoils:            
            [nprofiles,_] = airfoil.shft_xss.shape
            for i in range(nprofiles):
                fig.add_trace(go.Scatter3d(x=airfoil.shft_xss[i,:], y=airfoil.shft_yss[i,:], z=airfoil.shft_zss[i,:],  marker=marker,line=dict(color='red',width=2)))
                fig.add_trace(go.Scatter3d(x=airfoil.shft_xps[i,:], y=airfoil.shft_yps[i,:], z=airfoil.shft_zps[i,:],  marker=marker,line=dict(color='blue',width=2)))
                
                theta_max = self.check_replace_max(theta_max,np.max(np.append(airfoil.shft_xss[i,:],airfoil.shft_xps[i,:])))
                theta_min = self.check_replace_min(theta_min,np.min(np.append(airfoil.shft_xss[i,:],airfoil.shft_xps[i,:])))

        # Plot the channel
        zhub = np.linspace(min(self.zhub),max(self.zhub),500)
        rhub = self.hub_spline(zhub)

        rshroud = self.shroud_spline(zhub)
        zshroud = zhub
        fig.add_trace(go.Scatter3d(x=zhub, y=zhub*0, z=rhub,  marker=marker,line=dict(color='black',width=2)))
        fig.add_trace(go.Scatter3d(x=zshroud, y=zshroud*0, z=rshroud,  marker=marker,line=dict(color='black',width=2)))
        
        rmax = self.check_replace_max(rmax,np.max(np.append(rhub,rshroud)))
        rmin = self.check_replace_min(rmin,np.min(np.append(rhub,rshroud)))

        zmax = self.check_replace_max(zmax,np.max(np.append(zhub,zshroud)))
        zmin = self.check_replace_min(zmin,np.min(np.append(zhub,zshroud)))

        fig.update_layout(showlegend=False,scene= dict(xaxis = dict(nticks=4, range=[zmin,zmax]),
                                                        yaxis = dict(nticks=4, range=[rmin,rmax]),
                                                        zaxis = dict(nticks=4, range=[theta_min,theta_max]),
                                                        aspectmode='cube',aspectratio=dict(x=1, y=1, z=1)))
        fig.show()

    def plot3D(self):
        """3D plot of the channel and blade using matplotlib 
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        theta_max=0.0; zmax=0.0; rmax=0.0
        theta_min=0.0; zmin=0.0; rmin=0.0

        for airfoil in self.airfoils:            
            [nprofiles,_] = airfoil.shft_xss.shape
            for i in range(nprofiles):
                ax.plot3D(airfoil.shft_xss[i,:],airfoil.shft_yss[i,:],airfoil.shft_zss[i,:],color='red')
                ax.plot3D(airfoil.shft_xps[i,:],airfoil.shft_yps[i,:],airfoil.shft_zps[i,:],color='blue')
                
                theta_max = self.check_replace_max(theta_max,np.max(np.append(airfoil.shft_yss[i,:],airfoil.shft_yps[i,:])))
                theta_min = self.check_replace_min(theta_min,np.min(np.append(airfoil.shft_yss[i,:],airfoil.shft_yps[i,:])))
                # Create cubic bounding box to simulate equal aspect ratio
        
        zhub = np.linspace(min(self.zhub),max(self.zhub),500)
        rhub = self.hub_spline(zhub)

        rshroud = self.shroud_spline(zhub)
        zshroud = zhub
        ax.plot3D(zhub, zhub*0,rhub, color='black')
        ax.plot3D(zshroud, zshroud*0,rshroud, color='black')

        rmax = self.check_replace_max(rmax,np.max(np.append(rhub,rshroud)))
        rmin = self.check_replace_min(rmin,np.min(np.append(rhub,rshroud)))

        zmax = self.check_replace_max(zmax,np.max(np.append(zhub,zshroud)))
        zmin = self.check_replace_min(zmin,np.min(np.append(zhub,zshroud)))
        
        # zmax = check_replace_max(zmax,np.max(np.append(airfoil.shft_zps[i,:],airfoil.shft_zss[i,:])))
        # zmin = check_replace_min(zmin,np.min(np.append(airfoil.shft_zps[i,:],airfoil.shft_zss[i,:])))

        max_range = np.array([rmax-rmin, theta_max-theta_min, zmax-zmin]).max()
        Thetab = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(theta_max+theta_min)
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(zmax+zmin)
        Rb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(rmax+rmin)
        # Comment or uncomment following both lines to test the fake bounding box:
        for thetab, zb, rb in zip(Thetab, Zb, Rb):
            ax.plot([zb],[thetab],[rb], 'w')
        ax.view_init(azim=-90, elev=-90)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def plot2D(self,fig_size=None):
        """2D plot of the channel and blade using matplotlib
        """
        if fig_size:
            fig = plt.figure(figsize=fig_size)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        zmax=0.0; rmax=0.0
        zmin=0.0; rmin=0.0

        for airfoil in self.airfoils:            
            [nprofiles,_] = airfoil.shft_xss.shape
            for i in range(nprofiles):
                ax.plot(airfoil.shft_xss[i,:],airfoil.shft_zss[i,:],color='red')
                ax.plot(airfoil.shft_xps[i,:],airfoil.shft_zps[i,:],color='blue')
        
        zhub = np.linspace(min(self.zhub),max(self.zhub),500)
        rhub = self.hub_spline(zhub)

        rshroud = self.shroud_spline(zhub)
        zshroud = zhub
        ax.plot(zhub, rhub,color='black')
        ax.plot(zshroud, rshroud,color='black')

        # Plot the control points
        if (len(self.zhub_control)>0):
            ax.scatter(self.zhub_control, self.rhub_control,s=10,marker='o',c='red')

        if len(self.rshroud_control)>0:
            ax.scatter(self.zshroud_control, self.rshroud_control,s=10,marker='o',c='red')

        
        ax.set_aspect('equal')
        return fig
    
    def export_json(self,nChannelPoints:int=200,xmin:float=None,xmax:float=None, scale:float=1):
        """Exports the design to a json file

        Args:
            nChannelPoints (int, optional): Number of channel points. Defaults to 200.
            xmin (float, optional): minimum axial value before scale is applied. Defaults to None.
            xmax (float, optional): max axial value before scale is applied. Defaults to None.
            scale (float, optional): value to scale the geometry by. Defaults to 1
        """
        
        if not xmin:
            xmin = self.zhub[0]
        if not xmax:
            xmax = self.zhub[-1]

        x = np.linspace(xmin,xmax,nChannelPoints)
        rhub = self.hub_spline(x)*scale
        xhub = x*scale
        rshroud = self.shroud_spline(x)*scale
        xshroud = x*scale

        data = dict()
        data['channel'] = {'shroud':{'x':xshroud.tolist(),'r':rshroud.tolist()}, 
                            'hub':{'x':xhub.tolist(),'r':rhub.tolist()}}
        
        data['blades'] = list()

        for i in range(len(self.airfoils)):
            blade = self.airfoils[i]

            # blade.cartesian()
            xss = blade.shft_xss*scale
            xps = blade.shft_xps*scale

            yss = blade.shft_yss*scale
            yps = blade.shft_yps*scale

            zss = blade.shft_zss*scale
            zps = blade.shft_zps*scale

            data['blades'].append({'blade_index':i})
            sections = list()
            for j in range(xss.shape[0]):
                suction = {'x':xss[j,:].tolist(), 'rth':yss[j,:].tolist(),'r':zss[j,:].tolist()}
                pressure = {'x':xps[j,:].tolist(), 'rth':yps[j,:].tolist(),'r':zps[j,:].tolist()}
                sections.append({'suction':suction,'pressure':pressure})

            data['blades'][-1]['sections'] = sections

        with open('aero_geometry.json','w') as f:
            json.dump(data,f,indent=4, sort_keys=True)


    def export_dat(self,nChannelPoints=200,xmin:float=None,xmax:float=None, scale:float=1):
        """Exports the channel to a dat file, we recommend using json.

        Args:
            nChannelPoints (int, optional): number of points to define the channel. Defaults to 200.
            xmin (float, optional): minimum axial value. Defaults to None.
            xmax (float, optional): maximum axial value. Defaults to None.
            scale (float, optional): value to scale the geometry by. Defaults to 1
            
        """
        
        with open('aero_geometry.dat','w') as f:
            f.write('NBlades {0:d}\n'.format(len(self.airfoils)))
            
            for i in range(len(self.airfoils)):
                blade = self.airfoils[i]
                # blade.cartesian()
                xss = blade.shft_xss*scale
                xps = blade.shft_xps*scale

                yss = blade.shft_yss*scale
                yps = blade.shft_yps*scale

                zss = blade.shft_zss*scale
                zps = blade.shft_zps*scale

                f.write('Blade {0:d}\n'.format(i))
                f.write('\tnpoints {0:d}\n'.format(xss.shape[1]))
                f.write('\tnsections {0:d}\n'.format(xss.shape[0]))                
                for j in range(xss.shape[0]):
                    f.write('\tsection {0:d}\n'.format(j))
                    f.write('\t\txss rth_ss r xps rth_ps r\n')
                    for k in range(xss.shape[1]):
                        f.write('\t\t{:5.6f} {:5.6f} {:5.6f}'.format(xss[j,k],yss[j,k],zss[j,k]))
                        f.write(' {:5.6f} {:5.6f} {:5.6f}\n'.format(xps[j,k],yps[j,k],zps[j,k]))
                        
            if not xmin:
                xmin = self.zhub[0]
            if not xmax:
                xmax = self.zhub[-1]

            x = np.linspace(xmin,xmax,nChannelPoints)
            rhub = self.hub_spline(x)*scale
            xhub = x*scale
            rshroud = self.shroud_spline(x)*scale
            xshroud = x*scale

            f.write('2D Passage\n')
            f.write('Points {0:d}'.format(len(xshroud)))
            f.write('\thub\n')
            for i in range(len(xhub)):
                f.write('\t\t{:5.6f} {:5.6f}\n'.format(xhub[i],rhub[i]))
            f.write('\tshroud\n')
            for i in range(len(xshroud)):
                f.write('\t\t{:5.6f} {:5.6f}\n'.format(xshroud[i],rshroud[i]))
            
        