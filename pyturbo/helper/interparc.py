import numpy as np 

def interpcurve(N,pX,pY):
    #equally spaced in arclength
    N=np.transpose(np.linspace(0,1,N))

    #how many points will be uniformly interpolated?
    nt=N.size

    #number of points on the curve
    n=pX.size
    pxy=np.array((pX,pY)).T
    p1=pxy[0,:]
    pend=pxy[-1,:]
    last_segment= np.linalg.norm(np.subtract(p1,pend))
    epsilon= 10*np.finfo(float).eps

    #IF the two end points are not close enough lets close the curve
    if last_segment > epsilon*np.linalg.norm(np.amax(abs(pxy),axis=0)):
        pxy=np.vstack((pxy,p1))
        nt = nt + 1
    else:
        print('Contour already closed')

    pt=np.zeros((nt,2))

    #Compute the chordal arclength of each segment.
    chordlen = (np.sum(np.diff(pxy,axis=0)**2,axis=1))**(1/2)
    #Normalize the arclengths to a unit total
    chordlen = chordlen/np.sum(chordlen)
    #cumulative arclength
    cumarc = np.append(0,np.cumsum(chordlen))

    tbins= np.digitize(N,cumarc) # bin index in which each N is in

    #catch any problems at the ends
    tbins[np.where(tbins<=0 | (N<=0))]=1
    tbins[np.where(tbins >= n | (N >= 1))] = n - 1      

    s = np.divide((N - cumarc[tbins]),chordlen[tbins-1])
    pt = pxy[tbins,:] + np.multiply((pxy[tbins,:] - pxy[tbins-1,:]),(np.vstack([s]*2)).T)

    return pt 