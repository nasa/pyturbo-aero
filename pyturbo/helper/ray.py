import numpy as np
import math
import matplotlib.pyplot as plt

class ray2D(object):
    def __init__(self,x,y,dx,dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def get_point(self,t):
        x = self.x+t*self.dx
        y = self.y+t*self.dy
        return x,y


    def set_length(self,ray_length):
        t = ray_length/math.sqrt(self.dx**2+self.dy**2)

    def plot(self,t):
        [x,y] = self.get_point(t)
        plt.figure()
        plt.plot(x,y)
        plt.draw()
        plt.show()

class ray3D(ray2D):
    def __init__(self,x,y,z,dx,dy,dz):
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def get_point(self,t):
        x = self.x+t*self.dx
        y = self.y+t*self.dy
        z = self.z+t*self.dz
        return x,y,z


    def set_length(self,ray_length):
        t = ray_length/math.sqrt(self.dx**2+self.dy**2+self.dz**2)

    def plot(self,t):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        [x,y] = self.get_point(t)
        plt.figure()
        plt.plot(x,y)
        plt.draw()
        plt.show()

def ray2D_intersection(ray1,ray2):
    A =np.array([[ray1.dx, -ray2.dx],[ray1.dy, -ray2.dy]])
    b = np.array([[ray2.x-ray1.x], [ray2.y-ray1.y]])

    T = np.linalg.solve(A,b)
    t1 = T[0,0]
    t2 = T[1,0]

    [x, y] = ray1.get_point(t1)
    [x2, y2] = ray2.get_point(t2)
    if (np.sqrt((x-x2)**2 + (y-y2)**2) > 0.001):
        return []
    else:
        return x,y,t1,t2