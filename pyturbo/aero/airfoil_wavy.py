from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from .airfoil3D import *
from ..helper import derivative_1, bezier,cosd,sind
import math

class airfoil_wavy(airfoil3D):
    """Makes the surface of the airfoil: LE, TE, SS, PS wavy
    """
    def __init__(self,profileArray,profile_loc,height):
        """Initializes a wavy airfoil from an array of profiles, location and height, same as airfoil3D

        Args:
            profileArray ([airfoil2D]): Array of 2D airfoil profiles 
            profile_loc ([List[float]]): Locations in terms of percent span where these 2D airfoils are located 
            height ([float]): height of the airfoil
        """
        super(airfoil_wavy, self).__init__(profileArray,profile_loc,height)
   

    def stretch_thickness_chord(self,SSRatio,PSRatio,LERatio,TERatio,LE_wave_angle,TE_wave_angle,TE_smooth=0.85):
        """Changes the thickness to chord ratio along the span of the blade 

        Args:
            SSRatio (List[float]): list of float values for example sin(x)+1 where the 1 means no scaling
            PSRatio (List[float]): list of float values for example sin(x)+1 where the 1 means no scaling
            LERatio (List[float]): list of float values for example sin(x)+1 where the 1 means no scaling
            TERatio (List[float]): list of float values for example sin(x)+1 where the 1 means no scaling
            LE_wave_angle (List[float]): 0 to 90, 90 means perpendicular to metal angle.
            TE_wave_angle (List[float]): 0 to 90, 90 means perpendicular to metal angle.
            TE_smooth (float, optional): This is the percntage along the surface of the blade to start smoothing the trailing edge. Defaults to 0.85.
        """
        
        LERatio = convert_to_ndarray(LERatio)
        SSRatio = convert_to_ndarray(SSRatio)
        PSRatio = convert_to_ndarray(PSRatio)
        TERatio = convert_to_ndarray(TERatio)

        LE_wave_angle = np.radians(convert_to_ndarray(LE_wave_angle))
        TE_wave_angle = np.radians(convert_to_ndarray(TE_wave_angle))
        
        [nprofiles,npointsPerProfile] = self.shft_xss.shape

        t = np.linspace(0,1,nprofiles)
        te_center_x = copy.deepcopy(self.te_center_x) #Create a copy showing the previous value
        te_center_y = copy.deepcopy(self.te_center_y)
        self.spanw_leratio_fn = PchipInterpolator(np.linspace(0,1,len(LERatio)),LERatio)
        self.spanw_ssratio_fn = PchipInterpolator(np.linspace(0,1,len(SSRatio)),SSRatio)
        self.spanw_psratio_fn = PchipInterpolator(np.linspace(0,1,len(PSRatio)),PSRatio)
        self.spanw_teratio_fn = PchipInterpolator(np.linspace(0,1,len(TERatio)),TERatio)

        self.spanw_lewave_fn = PchipInterpolator(np.linspace(0,1,len(LE_wave_angle)),np.radians(LE_wave_angle))
        self.spanw_tewave_fn = PchipInterpolator(np.linspace(0,1,len(LE_wave_angle)),np.radians(TE_wave_angle))
        
        vibrissaeIndx = np.zeros(nprofiles)
        ## Build a shift matrix             
        cyss = []
        cxss = []
        cyps = []
        cxps = []    
        
        # Spline fit 
        for i in range(nprofiles):           # TE               # LE                   
            # camber line
            xcam = (self.shft_xss[i,:]+self.shft_xps[i,:])/2.0
            ycam = (self.shft_yss[i,:]+self.shft_yps[i,:])/2.0
            dydx = derivative_1(ycam,xcam)
            # Find where camber line changes signs
            vibrissaeIndx[i] = -1
            s1 = np.sign(dydx[1])
            for j in range(2,len(dydx)):
                if (np.sign(dydx[j]) !=s1):
                    vibrissaeIndx[i] = j
                    break
            
            if (vibrissaeIndx[i]==-1):
                vibrissaeIndx[i] = math.ceil(0.2*npointsPerProfile)
            
            # use self for wavy TE

            # These are scaling factors used to scale dx and dy
            t = np.array([0,vibrissaeIndx[i],math.floor(npointsPerProfile*TE_smooth),npointsPerProfile-1])
            percent_span = i/(nprofiles-1)
            profile_scale = convert_to_ndarray([self.spanw_leratio_fn([percent_span])[0], self.spanw_ssratio_fn([percent_span])[0], 
                self.spanw_teratio_fn([percent_span])[0],self.spanw_teratio_fn([percent_span])[0]])
            cxss.append(PchipInterpolator(t,1+profile_scale))
            cyss.append(PchipInterpolator(t,1+profile_scale)) # Y is axial    

            profile_scale = convert_to_ndarray([self.spanw_leratio_fn([percent_span])[0], self.spanw_psratio_fn([percent_span])[0], 
                self.spanw_teratio_fn([percent_span])[0],self.spanw_teratio_fn([percent_span])[0]])
            cxps.append(PchipInterpolator(t,1+profile_scale))
            cyps.append(PchipInterpolator(t,1+profile_scale)) # Y is axial
        
        # Convert to int
        vibrissaeIndx = vibrissaeIndx.astype(np.int)

        chord = np.zeros(nprofiles)
        centerPointX = np.zeros(nprofiles)
        centerPointY = np.zeros(nprofiles)

        camberX = np.add(self.shft_xss,self.shft_xps)/2.0
        camberY = np.add(self.shft_yss,self.shft_yps)/2.0
        t = np.linspace(0,1,npointsPerProfile)

        # Apply thickness chord ratio
        for i in range(nprofiles):
            ps = i/(nprofiles-1)# percent span 

            chord[i] = np.sqrt((self.shft_yss[i,-1]-self.shft_yss[i,-1])**2+(self.shft_xss[i,-1]-self.shft_xss[i,-1])**2)  
            centerPointX[i] = camberX[i,vibrissaeIndx[i]] # Stretch will be based around self point
            centerPointY[i] = camberY[i,vibrissaeIndx[i]]
            
            # LE Rotation Angle
            rot_ang1 = self.spanw_lewave_fn([ps])[0]*(0.5*(1+np.cos(math.pi*t))) 
            # TE Rotation Angle
            rot_ang2 = self.spanw_tewave_fn([ps])[0]*(0.5*(1+np.cos(math.pi*np.flip(t))))
            # Blend the rotation angles into a vector
            rot_angl = rot_ang1+rot_ang2
            rotation_vector = rot_angl
            
            # Find thickness for each camber point
            for j in range(npointsPerProfile):                      
                # Suction Side
                dx = self.shft_xss[i,j]-centerPointX[i]
                dy = self.shft_yss[i,j]-centerPointY[i]
                # theta = 0 + rotation_vector[j] # Rotation angle 
                # Scale dx and dy by thickness to chord
                dx_scaled = dx*cxss[i]([j])[0] # dx from centerPoint
                dy_scaled = dy*cyss[i]([j])[0] # dy from centerPoint
                # obtain the shift relative the point
                xnew = centerPointX[i] + dx_scaled
                ynew = centerPointY[i] + dy_scaled
                
                dx = xnew - self.shft_xss[i,j]
                dy = ynew - self.shft_yss[i,j]

                rot = np.array([ [cosd(rotation_vector[j]), -sind(rotation_vector[j])], [sind(rotation_vector[j]), cosd(rotation_vector[j])] ]) 
                rot = np.matmul(rot, np.array([[dx],[dy]]))

                dx_rot = rot[0]
                dy_rot = rot[1]
                
                self.shft_xss[i,j] = self.shft_xss[i,j] + dx_rot
                self.shft_yss[i,j] = self.shft_yss[i,j] + dy_rot
                
                # Pressure Side
                dx = self.shft_xps[i,j]-centerPointX[i]
                dy = self.shft_yps[i,j]-centerPointY[i]
                
                # Scale dx and dy by thickness to chord
                dx_scaled = dx*cxps[i]([j])[0]
                dy_scaled = dy*cyps[i]([j])[0]
                # obtain the shift relative to the point
                xnew = centerPointX[i] + dx_scaled
                ynew = centerPointY[i] + dy_scaled
                dx = xnew - self.shft_xps[i,j]
                dy = ynew - self.shft_yps[i,j]
                
                rot = np.array([[cosd(rotation_vector[j]), -sind(rotation_vector[j])], [sind(rotation_vector[j]), cosd(rotation_vector[j])]])
                rot = np.matmul(rot, np.array([[dx],[dy]]))

                dx_rot = rot[0]
                dy_rot = rot[1]
                self.shft_xps[i,j] = dx_rot + self.shft_xps[i,j] 
                self.shft_yps[i,j] = dy_rot + self.shft_yps[i,j]
                
            
            # Shift the trailing edge center
            j = npointsPerProfile-1
            dx = te_center_x[i]-centerPointX[i]
            dy = te_center_y[i]-centerPointY[i] 
            dx_scaled = dx*(cxps[i]([j])[0] + cxss[i]([j])[0])/2
            dy_scaled = dy*(cyps[i]([j])[0] + cyss[i]([j])[0])/2

            xnew = centerPointX[i] + dx_scaled
            ynew = centerPointY[i] + dy_scaled
            dx = xnew - te_center_x[i]
            dy = ynew - te_center_y[i]

            rot = np.matmul(np.array([ 
                        [cosd(rotation_vector[j]), -sind(rotation_vector[j])],
                        [sind(rotation_vector[j]), cosd(rotation_vector[j])]
                        ]), np.array([ [dx], [dy] ]))
            dx_rot = rot[0]
            dy_rot = rot[1]
            te_center_x[i] = dx_rot + te_center_x[i]
            te_center_y[i] = dy_rot + te_center_y[i]
                    
        self.te_center_x = te_center_x
        self.te_center_y = te_center_y
        
    

    def stretch_thickness_chord_te(self,SSRatio,PSRatio,LERatio,TERatio,LE_wave_angle,TE_wave_angle,TE_smooth=0.90):
        """Stretches blade but keeps the trailing edge constant

            Args:
                SSRatio (List[float]): list of float values for example sin(x)+1 where the 1 means no scaling
                PSRatio (List[float]): list of float values for example sin(x)+1 where the 1 means no scaling
                LERatio (List[float]): list of float values for example sin(x)+1 where the 1 means no scaling
                TERatio (List[float]): list of float values for example sin(x)+1 where the 1 means no scaling
                LE_wave_angle (List[float]): 0 to 90, 90 means perpendicular to metal angle.
                TE_wave_angle (List[float]): 0 to 90, 90 means perpendicular to metal angle.
                TE_smooth (float, optional): This is the percntage along the surface of the blade to start smoothing the trailing edge. Defaults to 0.85.

        """
        LERatio = convert_to_ndarray(LERatio)
        SSRatio = convert_to_ndarray(SSRatio)
        PSRatio = convert_to_ndarray(PSRatio)
        TERatio = convert_to_ndarray(TERatio)

        LE_wave_angle = np.radians(convert_to_ndarray(LE_wave_angle))
        TE_wave_angle = np.radians(convert_to_ndarray(TE_wave_angle))
     
        [nprofiles,npointsPerProfile] = self.shft_xss.shape

        t = np.linspace(0,1,nprofiles)
        te_center_x = copy.deepcopy(self.te_center_x) #Create a copy showing the previous value
        te_center_y = copy.deepcopy(self.te_center_y)
        # Create a spline that represents the SSRatio, PSRatio, LERatio, TERatio
        # Check to see if variable exists and if there are differences between the inputs
        self.spanw_leratio_fn = PchipInterpolator(np.linspace(0,1,len(LERatio)),LERatio)
        self.spanw_ssratio_fn = PchipInterpolator(np.linspace(0,1,len(SSRatio)),SSRatio)
        self.spanw_psratio_fn = PchipInterpolator(np.linspace(0,1,len(PSRatio)),PSRatio)
        self.spanw_teratio_fn = PchipInterpolator(np.linspace(0,1,len(TERatio)),TERatio)

        self.spanw_lewave_fn = PchipInterpolator(np.linspace(0,1,len(LE_wave_angle)),LE_wave_angle)
        self.spanw_tewave_fn = PchipInterpolator(np.linspace(0,1,len(LE_wave_angle)),TE_wave_angle)
        
        vibrissaeIndx = np.zeros(nprofiles)
        # Build a shift matrix             
        npoints_no_TE = npointsPerProfile - self.nte+1 # number of points neglecting the Trailing Edge, trailing edge diameter remains untouched
        
        cyss = []
        cxss = []
        cyps = []
        cxps = []
        # Spline fit for each profile
        for i in range(nprofiles): #           % TE               % LE                   
            # Find the inflection point for each profile 
            xcam = (self.shft_xss[i,:]+self.shft_xps[i,:])/2.0
            ycam = (self.shft_yss[i,:]+self.shft_yps[i,:])/2.0
            dydx = derivative_1(ycam,xcam)
            # Find where camber line changes signs
            vibrissaeIndx[i] = -1
            s1 = np.sign(dydx[1])
            for j in range(2,len(dydx)):
                if (np.sign(dydx[j]) !=s1):
                    vibrissaeIndx[i] = j
                    break

            if (vibrissaeIndx[i]==-1):
                vibrissaeIndx[i] = ceil(0.2*npointsPerProfile)
            
            # use self for wavy TE

            # These are scaling factors used to scale dx and dy
            # There is no scaling near the trailing edge
            """
                Define the scaling factors for each profile as a function of the profile and which point
            """
            t = [0, vibrissaeIndx[i], math.floor((npoints_no_TE-2)*TE_smooth), npoints_no_TE-2]
            percent_span = i/(nprofiles-1)
            profile_scale = convert_to_ndarray([self.spanw_leratio_fn([percent_span])[0], self.spanw_ssratio_fn([percent_span])[0], 
                self.spanw_teratio_fn([percent_span])[0] ,self.spanw_teratio_fn([percent_span])[0]])
            cxss.append(PchipInterpolator(t,1+profile_scale))
            cyss.append(PchipInterpolator(t,1+profile_scale)) # Y is axial 
            
            profile_scale = convert_to_ndarray([self.spanw_leratio_fn([percent_span])[0], self.spanw_psratio_fn([percent_span])[0], 
                self.spanw_teratio_fn([percent_span])[0],self.spanw_teratio_fn([percent_span])[0]])
            cxps.append(PchipInterpolator(t,1+profile_scale))
            cyps.append(PchipInterpolator(t,1+profile_scale)) # Y is axial
        
        # Convert to int
        vibrissaeIndx = vibrissaeIndx.astype(np.int)

        chord = np.zeros(nprofiles)
        centerPointX = np.zeros(nprofiles)
        centerPointY = np.zeros(nprofiles)

        camberX = np.add(self.shft_xss,self.shft_xps)/2.0
        camberY = np.add(self.shft_yss,self.shft_yps)/2.0
        
        TE_Radius_ss = np.zeros(nprofiles)
        TE_Radius_ps = np.zeros(nprofiles)
        # Calculate the trailing edge radius for each profile
        for i in range(0,nprofiles):
            TE_Radius_ss[i] = math.sqrt((self.te_center_x[i]-self.shft_xss[i,-1])**2 + (self.te_center_y[i]-self.shft_yss[i,-1])**2)
            TE_Radius_ps[i] = math.sqrt((self.te_center_x[i]-self.shft_xps[i,-1])**2 + (self.te_center_y[i]-self.shft_yps[i,-1])**2)
        
        
        # Apply the thickness to chord ratio - Make the geometry wavy

        t_no_te = np.linspace(0,1,npoints_no_TE) # points up until where TRAILING EDGE ENDS
        for i in range(nprofiles):                              # Apply thickness chord ratio
            ps = i/(nprofiles-1)# percent span 
            chord[i] = np.sqrt((self.shft_yss[i,-1]-self.shft_yss[i,-1])**2+(self.shft_xss[i,-1]-self.shft_xss[i,-1])**2)  
            
            centerPointX[i] = camberX[i,vibrissaeIndx[i]] # Stretch will be based around self point
            centerPointY[i] = camberY[i,vibrissaeIndx[i]]
            
            # Apply the rotation angle using a Hann Windowing function 
            # LE Rotation Angle
            rot_ang1 = self.spanw_lewave_fn([ps])[0]*(0.5*(1+np.cos(math.pi*t_no_te))) 
            # TE Rotation Angle
            rot_ang2 = self.spanw_tewave_fn([ps])[0]*(0.5*(1+np.cos(math.pi*np.flip(t_no_te))))
            # Blend the rotation angles into a vector
            rot_angl = rot_ang1+rot_ang2
            rotation_vector = rot_angl
        
            for j in range(npoints_no_TE):                       # Find thickness for each camber point              
                # Suction Side
                dx = self.shft_xss[i,j]-centerPointX[i]
                dy = self.shft_yss[i,j]-centerPointY[i]
                # theta = 0 + rotation_vector(j); % Rotation angle 
                # Scale dx and dy by thickness to chord
                dx_scaled = dx*cxss[i]([j])[0] # dx from centerPoint
                dy_scaled = dy*cyss[i]([j])[0] # dy from centerPoint
                # obtain the shift relative the point
                xnew = centerPointX[i] + dx_scaled
                ynew = centerPointY[i] + dy_scaled
                
                dx = xnew - self.shft_xss[i,j]
                dy = ynew - self.shft_yss[i,j]

                rot = np.array([ [cosd(rotation_vector[j]), -sind(rotation_vector[j])], [sind(rotation_vector[j]), cosd(rotation_vector[j])] ]) 
                rot = np.matmul(rot, np.array([[dx],[dy]]))
                
                dx_rot = rot[0]
                dy_rot = rot[1]
                
                self.shft_xss[i,j] = self.shft_xss[i,j] + dx_rot
                self.shft_yss[i,j] = self.shft_yss[i,j] + dy_rot 
                
                # Pressure Side
                dx = self.shft_xps[i,j]-centerPointX[i]
                dy = self.shft_yps[i,j]-centerPointY[i]    
                
                # Scale dx and dy by thickness to chord
                dx_scaled = dx*cxps[i]([j])[0]
                dy_scaled = dy*cyps[i]([j])[0]
                # obtain the shift relative to the point
                xnew = centerPointX[i] + dx_scaled
                ynew = centerPointY[i] + dy_scaled
                dx = xnew - self.shft_xps[i,j]
                dy = ynew - self.shft_yps[i,j]
                
                rot = np.array([[cosd(rotation_vector[j]), -sind(rotation_vector[j])], [sind(rotation_vector[j]), cosd(rotation_vector[j])]])
                rot = np.matmul(rot, np.array([[dx],[dy]]))

                dx_rot = rot[0]
                dy_rot = rot[1]
                self.shft_xps[i,j] = dx_rot + self.shft_xps[i,j] 
                self.shft_yps[i,j] = dy_rot + self.shft_yps[i,j]
                
                            
            # Shift the trailing edge center
            
            j = npoints_no_TE-1 # Take the final shift
            dx = te_center_x[i]-centerPointX[i]
            dy = te_center_y[i]-centerPointY[i] 
            dx_scaled = dx*(cxps[i]([j])[0] + cxss[i]([j])[0])/2
            dy_scaled = dy*(cyps[i]([j])[0] + cyss[i]([j])[0])/2

            xnew = centerPointX[i] + dx_scaled
            ynew = centerPointY[i] + dy_scaled
            dx = xnew - te_center_x[i]
            dy = ynew - te_center_y[i]

            rot = np.array([
                            [cosd(rotation_vector[j]), -sind(rotation_vector[j])],
                            [sind(rotation_vector[j]), cosd(rotation_vector[j])]
                            ])
            rot = np.matmul(rot,np.array([[dx],[dy]]))
            dx_rot = rot[0]
            dy_rot = rot[1]
            te_center_x[i] = dx_rot + te_center_x[i]
            te_center_y[i] = dy_rot + te_center_y[i]
            
                
            # Shift the trailing edge by the dx, dy of the camber 
            # Find new last camber point
            dx = te_center_x[i] - self.te_center_x[i] # new te_center_x subtract the old one before the wavy
            dy = te_center_y[i] - self.te_center_y[i]
            
            # Shift the TE Points by common dx and dy
            te_start = npoints_no_TE
            self.shft_xss[i,te_start:] = self.shft_xss[i,te_start:]+dx # self is to maintain the same Trailing edge diameter
            self.shft_yss[i,te_start:] = self.shft_yss[i,te_start:]+dy
            self.shft_xps[i,te_start:] = self.shft_xps[i,te_start:]+dx
            self.shft_yps[i,te_start:] = self.shft_yps[i,te_start:]+dy
            
            TE_Radius_ss[i] = np.sqrt((te_center_x[i]-self.shft_xss[i,-1])**2+ (te_center_y[i]-self.shft_yss[i,-1])**2)
            TE_Radius_ps[i] = np.sqrt((te_center_x[i]-self.shft_xps[i,-1])**2+ (te_center_y[i]-self.shft_yps[i,-1])**2)
        

        self.te_center_x = te_center_x
        self.te_center_y = te_center_y
        # Fix the start and end points 
        # Make sure LE and TE start and end at the same point. 
        startPointX = (self.shft_xss[:,0] + self.shft_xps[:,0])/2.0 # First X coordinate of all the profiles
        startPointY = (self.shft_yss[:,0] + self.shft_yps[:,0])/2.0 # First Y coordinate of all the profiles 

        endPointX = (self.shft_xss[:,-1] + self.shft_xps[:,-1])/2.0 # last x-coordinate of all the profiles
        endPointY = (self.shft_yss[:,-1] + self.shft_yps[:,-1])/2.0 # last y-coordinate of all the profiles
        
        self.shft_xss[:,0] = startPointX
        self.shft_yss[:,0] = startPointY
        self.shft_xps[:,0] = startPointX
        self.shft_yps[:,0] = startPointY

        self.shft_xss[:,-1] = endPointX
        self.shft_yss[:,-1] = endPointY
        self.shft_xps[:,-1] = endPointX
        self.shft_yps[:,-1] = endPointY

        # self.shft_xss = np.array([startPointX, self.shft_xss[:,1:-2], endPointX])
        # self.shft_xps = [startPointX, self.shft_xps[:,1:-2], endPointX]
        # self.shft_yss = [startPointY, self.shft_yss[:,1:-2], endPointY]
        # self.shft_yps = [startPointY, self.shft_yps[:,1:-2], endPointY]

        self.te_radius_ss = TE_Radius_ss
        self.te_radius_ps = TE_Radius_ps
    
    def whisker_blade(self,LERatio,TERatio,PSRatio,LE_wave_angle,TE_wave_angle,TE_smooth=0.9):
        """Stretches blade but keeps the trailing edge constant. This code maintains the same cross sectional profile area. 
            The way it does this is by contracting or stretching the suction side

            The following inputs are functions ranging from 0% span to 100% span, it can be any function

        Args:
            LERatio ([type]): list of float values for example sin(x)+1 where the 1 means no scaling
            TERatio ([type]): list of float values for example sin(x)+1 where the 1 means no scaling
            PSRatio ([type]): list of float values for example sin(x)+1 where the 1 means no scaling
            LE_wave_angle (List[float]): 0 to 90, 90 means perpendicular to metal angle.
            TE_wave_angle (List[float]): 0 to 90, 90 means perpendicular to metal angle.
            TE_smooth (float, optional): [description]. Defaults to 0.9.
        """

        LERatio = convert_to_ndarray(LERatio)
        PSRatio = convert_to_ndarray(PSRatio)
        TERatio = convert_to_ndarray(TERatio)

        LE_wave_angle = np.radians(convert_to_ndarray(LE_wave_angle))
        TE_wave_angle = np.radians(convert_to_ndarray(TE_wave_angle))

        [nprofiles,npointsPerProfile] = self.shft_xss.shape

        # Create a spline that represents the SSRatio, PSRatio, LERatio, TERatio
        # Check to see if variable exists and if there are differences between the inputs
        self.spanw_leratio_fn = PchipInterpolator(np.linspace(0,1,len(LERatio)),LERatio)
        self.spanw_psratio_fn = PchipInterpolator(np.linspace(0,1,len(PSRatio)),PSRatio)
        self.spanw_teratio_fn = PchipInterpolator(np.linspace(0,1,len(TERatio)),TERatio)
        self.spanw_lewave_fn = PchipInterpolator(np.linspace(0,1,len(LE_wave_angle)),LE_wave_angle)
        self.spanw_tewave_fn = PchipInterpolator(np.linspace(0,1,len(LE_wave_angle)),TE_wave_angle)

        # Build a shift matrix             
        npoints_no_TE = npointsPerProfile - self.nte # number of points neglecting the Trailing Edge, trailing edge diameter remains untouched                
        
        chord = np.zeros(nprofiles)
        centerPointX = np.zeros(nprofiles)
        centerPointY = np.zeros(nprofiles)

        vibrissaeIndx = np.zeros(nprofiles)

        TE_Radius_ss = np.zeros(nprofiles)
        TE_Radius_ps = np.zeros(nprofiles)
        te_center_x = copy.deepcopy(self.te_center_x)
        te_center_y = copy.deepcopy(self.te_center_y)

        camberX = np.add(self.shft_xss,self.shft_xps)/2.0
        camberY = np.add(self.shft_yss,self.shft_yps)/2.0

        # Grab the center point
        profileArea= np.zeros(nprofiles)
        profileAreaNew = np.zeros(nprofiles)

        t_no_te = np.linspace(0,1,npoints_no_TE)
        def CreateWhisker(SSRatio,profile_indx):                                
            '''
                These are scaling factors used to scale dx and dy
                There is no scaling near the trailing edge
            ''' 
            percentSpan = profile_indx/(nprofiles-1)
            t = [0,vibrissaeIndx[profile_indx],floor((npoints_no_TE-2)*TE_smooth),npoints_no_TE-2]

            profile_scale = convert_to_ndarray([self.spanw_leratio_fn(percentSpan),SSRatio,
                self.spanw_teratio_fn(percentSpan),self.spanw_teratio_fn(percentSpan)])
            cxss = PchipInterpolator(t,1+profile_scale)
            cyss = PchipInterpolator(t,1+profile_scale) # Y is axial     

            profile_scale = convert_to_ndarray([self.spanw_leratio_fn(percentSpan),self.spanw_psratio_fn(percentSpan),
                self.spanw_teratio_fn(percentSpan),self.spanw_teratio_fn(percentSpan)])

            cxps = PchipInterpolator(t,1+profile_scale)
            cyps = PchipInterpolator(t,1+profile_scale) # Y is axial

            return cxss,cyss,cxps,cyps
            
        def ScaleBlade(indx_profile,indx_point,xss,yss,xps,yps,cxss,cxps,cyss,cyps,rotation_vector):
            # Suction Side
            dx = xss[indx_point]-centerPointX[indx_profile]
            dy = yss[indx_point]-centerPointY[indx_profile]

            # Scale dx and dy by thickness to chord
            dx_scaled = dx*cxss(indx_point) # dx from centerPoint
            dy_scaled = dy*cyss(indx_point) # dy from centerPoint
            # obtain the shift relative the point
            xnew = centerPointX[indx_profile] + dx_scaled
            ynew = centerPointY[indx_profile] + dy_scaled

            dx = xnew - xss[indx_point]
            dy = ynew - yss[indx_point]
            
            rot = np.matmul(np.array([
                [cosd(rotation_vector[indx_point]),-sind(rotation_vector[indx_point])], 
                    [sind(rotation_vector[indx_point]), cosd(rotation_vector[indx_point])]
                    ]), np.array([[dx], [dy]]))
            dx_rot = rot[0]
            dy_rot = rot[1]

            xss[indx_point] = xss[indx_point] + dx_rot
            yss[indx_point] = yss[indx_point] + dy_rot

            # Pressure Side
            dx = xps[indx_point]-centerPointX[indx_profile]
            dy = yps[indx_point]-centerPointY[indx_profile]  

            # Scale dx and dy by thickness to chord
            dx_scaled = dx*cxps(indx_point)
            dy_scaled = dy*cyps(indx_point)
            # obtain the shift relative to the point
            xnew = centerPointX[indx_profile] + dx_scaled
            ynew = centerPointY[indx_profile] + dy_scaled
            dx = xnew - xps[indx_point]
            dy = ynew - yps[indx_point]

            rot = np.matmul(np.array([
                [cosd(rotation_vector[indx_point]),-sind(rotation_vector[indx_point])], 
                    [sind(rotation_vector[indx_point]), cosd(rotation_vector[indx_point])]
                    ]), np.array([[dx], [dy]]))
            dx_rot = rot[0]
            dy_rot = rot[1]
            xps[indx_point] = dx_rot + xps[indx_point]
            yps[indx_point] = dy_rot + yps[indx_point]
            return xss,yss,xps,yps
        
        def ApplyWave(SSRatio,kwargs):
            profile_indx = kwargs['profile_indx']
            xss = copy.deepcopy(kwargs['xss'])
            yss = copy.deepcopy(kwargs['yss'])
            xps = copy.deepcopy(kwargs['xps'])
            yps = copy.deepcopy(kwargs['yps'])
            profileArea = kwargs['profileArea']
            bMinimize = kwargs['bMinimize']
            [cxss,cyss,cxps,cyps] = CreateWhisker(SSRatio[0],profile_indx)
            
            # LE Rotation Angle
            t = profile_indx/nprofiles
            rot_ang1 = self.spanw_lewave_fn([t])[0]*(0.5*(1+np.cos(math.pi*t_no_te)))
            # TE Rotation Angle
            rot_ang2 = self.spanw_tewave_fn([t])[0]*(0.5*(1+np.cos(math.pi*np.flip(t_no_te))))
            # Blend the rotation angles into a vector
            rotation_vector = rot_ang1+rot_ang2
            
            for point_indx in range(npoints_no_TE):
                [xss,yss,xps,yps] = ScaleBlade(profile_indx,point_indx,xss,yss,xps,yps,cxss,cxps,cyss,cyps,rotation_vector)
            
            # Move the Trailing Edge
            j = npoints_no_TE-1 # Take the final shift

            dx = te_center_x[profile_indx]-centerPointX[profile_indx]
            dy = te_center_y[profile_indx]-centerPointY[profile_indx] 
            dx_scaled = dx*cxss([j])[0] # dx from centerPoint
            dy_scaled = dy*cyss([j])[0] # dy from centerPoint

            xnew = centerPointX[profile_indx] + dx_scaled
            ynew = centerPointY[profile_indx] + dy_scaled
            dx = xnew - te_center_x[profile_indx]
            dy = ynew - te_center_y[profile_indx]

            rot = np.matmul(np.array([
                    [cosd(rotation_vector[j]), -sind(rotation_vector[j])], 
                    [sind(rotation_vector[j]), cosd(rotation_vector[j])]
                    ]),np.array([[dx], [dy]]))
            dx_rot = rot[0]
            dy_rot = rot[1]
            te_center_x_temp = dx_rot + te_center_x[profile_indx]
            te_center_y_temp = dy_rot + te_center_y[profile_indx]
            
                
            # Shift the trailing edge by the dx, dy of the camber 
            # Find new last camber point
            dx = te_center_x_temp - self.te_center_x[profile_indx] # new te_center_x subtract the old one before the wavy
            dy = te_center_y_temp - self.te_center_y[profile_indx]
            
            # Shift the TE Points by common dx and dy
            xss[npoints_no_TE:] = xss[npoints_no_TE:]+dx # self is to maintain the same Trailing edge diameter
            yss[npoints_no_TE:] = yss[npoints_no_TE:]+dy
            
            xps[npoints_no_TE:] = xps[npoints_no_TE:]+dx
            yps[npoints_no_TE:] = yps[npoints_no_TE:]+dy
            
            
            area_ss = np.trapz(yss,xss)
            area_ps = np.trapz(yps,xps)
            area = (area_ps-area_ss)  
            err = (area-profileArea)/profileArea
            if bMinimize:
                return  abs(err) 
            else:
                return xss,yss,xps,yps,te_center_x_temp,te_center_y_temp


        for i in range(nprofiles):
            print("Evaluating Profile " + str(i+1) + " out of " + str(nprofiles))
            TE_Radius_ss[i] = math.sqrt((self.te_center_x[i]-self.shft_xss[i,-1])**2 + (self.te_center_y[i]-self.shft_yss[i,-1])**2)
            TE_Radius_ps[i] = math.sqrt((self.te_center_x[i]-self.shft_xps[i,-1])**2 + (self.te_center_y[i]-self.shft_yps[i,-1])**2)
            
            # camber line
            xcam = (self.shft_xss[i,:]+self.shft_xps[i,:])/2.0
            ycam = (self.shft_yss[i,:]+self.shft_yps[i,:])/2.0
            dydx = derivative_1(ycam,xcam)
            # Find where camber line changes signs
            vibrissaeIndx[i] = -1
            s1 = np.sign(dydx[1])

            for indx in range(2,len(dydx)):
                if (np.sign(dydx[indx])!=s1):
                    vibrissaeIndx[i] = indx
                    break

            if (vibrissaeIndx[i]==-1):
                vibrissaeIndx[i] = ceil(0.2*npointsPerProfile)

            chord[i]=math.sqrt((self.shft_yss[i,-1]-self.shft_yss[i,0])**2+(self.shft_xss[i,-1]-self.shft_xss[i,0])**2)
            
            centerPointX[i] = camberX[i,int(vibrissaeIndx[i])] # Stretch will be based around self point
            centerPointY[i] = camberY[i,int(vibrissaeIndx[i])]
            # Find thickness for each camber point  
            
            # Compute Area of the blade
            area_ss = np.trapz(self.shft_yss[i,:],self.shft_xss[i,:])
            area_ps = np.trapz(self.shft_yps[i,:],self.shft_xps[i,:])
            profileArea[i] = abs(area_ps-area_ss)  
            # Get the SS ratio that maintains the area
            xss = self.shft_xss[i,:]
            yss = self.shft_yss[i,:]
            xps = self.shft_xps[i,:]
            yps = self.shft_yps[i,:]
            
            # Find the SS Ratio that maintains the same area as
            # original profile 
            bnds = ((-1,5),)
            res = minimize(ApplyWave,x0=(0),bounds=bnds,tol=1E-4,args=dict({"profile_indx":i,"xss":xss,"yss":yss,"xps":xps,"yps":yps,"profileArea":profileArea[i],"bMinimize":True}))
            SSRatio = res.x[0]
            [xss,yss,xps,yps,te_center_x_temp,te_center_y_temp] = ApplyWave([0],dict({"profile_indx":i,"xss":xss,"yss":yss,"xps":xps,"yps":yps,"profileArea":profileArea[i],"bMinimize":False}))
            self.shft_xss[i,:] = xss
            self.shft_yss[i,:] = yss
            self.shft_xps[i,:] = xps
            self.shft_yps[i,:] = yps
            # self.te_center_x[i] = te_center_x_temp
            # self.te_center_y[i] = te_center_y_temp
            
            area_ss = np.trapz(self.shft_yss[i,:],self.shft_xss[i,:])
            area_ps = np.trapz(self.shft_yps[i,:],self.shft_xps[i,:])
            profileAreaNew[i] = abs(area_ps-area_ss)
                            
            TE_Radius_ss[i] = math.sqrt((te_center_x[i]-self.shft_xss[i,-1])**2+ (te_center_y[i]-self.shft_yss[i,-1])**2)
            TE_Radius_ps[i] = math.sqrt((te_center_x[i]-self.shft_xps[i,-1])**2+ (te_center_y[i]-self.shft_yps[i,-1])**2)
        

        # Fix the start and end points 
        # Make sure LE and TE start and end at the same point. 
        startPointX = (self.shft_xss[:,0] + self.shft_xps[:,0])/2.0 # First X coordinate of all the profiles
        startPointY = (self.shft_yss[:,0] + self.shft_yps[:,0])/2.0 # First Y coordinate of all the profiles 

        endPointX = (self.shft_xss[:,-1] + self.shft_xps[:,-1])/2.0 # last x-coordinate of all the profiles
        endPointY = (self.shft_yss[:,-1] + self.shft_yps[:,-1])/2.0 # last y-coordinate of all the profiles
        
        self.shft_xss[:,0] = startPointX
        self.shft_yss[:,0] = startPointY
        self.shft_xps[:,0] = startPointX
        self.shft_yps[:,0] = startPointY

        self.shft_xss[:,-1] = endPointX
        self.shft_yss[:,-1] = endPointY
        self.shft_xps[:,-1] = endPointX
        self.shft_yps[:,-1] = endPointY

