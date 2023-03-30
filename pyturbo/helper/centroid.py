import numpy as np
def centroid(x,y):
    xc = np.sum(x)/len(x)
    yc = np.sum(y)/len(y)
    return xc,yc 