import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

class wave_control:
    """
        This is used to control a wave's starting ending location as well as invert the wave
        Invert - wave starts on the outside of the end points
                wave_____________wave
        NoInvert - wave starts between the bounds
                ___wave_wave___
    """
    
    def __init__(self,wave_array):            
        self.wave_vector=wave_array
        
    def get_wave(self,percent_start=0,percent_end=1,bInverse=False):
        """
            Returns the wave back as a percent
        """
        n = len(self.wave_vector)
        indx20 = percent_start*n
        indx80 = percent_end*n
        factor = self.wave_vector*0
        for i in range(n):
            if (i<math.ceil(indx20)):
                factor[i] = 0.5*(1-np.cos(2*math.pi*(i-1)/((indx20-1)*2)))
            elif (i>math.floor(indx80)):
                factor[i] = 0.5*(1-np.cos(2*math.pi*(n-(i))/((n-indx80+1)*2))) # Hann Window smoothing pressure side
            else:
                factor[i]=1

        if (bInverse):
            wave_vector_mod = self.wave_vector*(1-factor)
        else:
            wave_vector_mod = self.wave_vector*factor
        return wave_vector_mod
    
    def plot(self,percent_start=0,percent_end=1,bInverse=False):        
        wave_vector_mod = self.get_wave(percent_start,percent_end,bInverse)
        t = np.linspace(0,1,len(wave_vector_mod))
        
        _, ax1 = plt.subplots()
        ax1.plot(t, wave_vector_mod)
        ax1.set_xlabel("x-label")
        ax1.set_ylabel("y-label")
        plt.show()
    

    """
        Overload multiplication operator
    """
    def __mul__(self, other):
        return self.wave_vector*other
        
    def __rmul__(self, other):
        return self.wave_vector*other