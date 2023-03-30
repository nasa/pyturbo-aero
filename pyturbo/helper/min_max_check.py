import numpy as np

def check_replace_max(max_prev,max_new):
    if max_new>max_prev:
        return max_new
    else:
        return max_prev
def check_replace_min(min_prev,min_new):
    if min_new<min_prev:
        return min_new
    else:
        return min_prev

def create_cubic_bounding_box(xmax,xmin,ymax,ymin,zmax,zmin):  
    '''
        Creates a cubic bounding box around a blade
        This is useful for plotting purposes.

        Inputs:
            xmax (float)
            xmin (float)
            ymax (float)
            ymin (float)
            zmax (float)
            zmin (float)
        
        Returns:
            Xb - numpy matrix
            Yb - numpy matrix
            Zb - numpy matrix
    '''
    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    return Xb,Yb,Zb