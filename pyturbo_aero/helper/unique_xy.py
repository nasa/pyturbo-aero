import copy
import numpy as np

def uniqueXY(x,y):
    x2 = copy.deepcopy(x)*0; y2=copy.deepcopy(y)*0
    indx = 0
    for i in range(0,len(x)-1):
        if (x[i] !=x[i+1] and y[i] != y[i+1]):
            x2[indx] = x[i]
            y2[indx] = y[i]
            indx+=1
        

    x=np.append(x2[0:indx-1], x2[0])
    y=np.append(y2[0:indx-1], y2[0])
    return x,y