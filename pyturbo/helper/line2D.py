from typing import Tuple, Union
from .ray import *
import math
import numpy as np
from .ray import ray2D
from .bezier import bezier

class line2D:
    x:npt.NDArray
    y:npt.NDArray
    p:Tuple[float,float]
    q:Tuple[float,float]
    
    def __init__(self,pt1:Tuple[float,float],pt2:Tuple[float,float]):
        """Intializes a line

        Args:
            pt1 (Tuple[float,float]): start point in x,y
            pt2 (Tuple[float,float]): end point in x,y

        """
        x = np.array([pt1[0], pt2[0]])
        y = np.array([pt1[1], pt2[1]])
        self.x = x
        self.y = y
        self.p = (x[0], y[0]); self.q=(x[1], y[1])
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.length = np.sqrt(self.dx*self.dx+self.dy*self.dy)
        self.angle = np.arctan2(self.dy,self.dx)


    def to_bezier(self) -> bezier:
        """returns bezier curve

        Returns:
            bezier: returns a bezier curve
        """
        b = bezier(self.x.tolist(),self.y.tolist()) # type: ignore
        return b

    def ray(self) -> ray2D:
        """returns a ray 

        Returns:
            ray2D: ray
        """
        return ray2D(self.x[0],self.y[0],self.x[1]-self.x[0],self.y[1]-self.y[0])
        
    def intersect_ray(self,r2:ray2D):
        """Check if ray will intersect 

        Args:
            r2 (ray2D): ray

        Returns:
            Tuple containing: 
            
                **t** (float): t value of ray
                **bIntersect** (bool): True = intersect, False, no intersection
        """
        r1 = self.ray()
        A = np.array([ [r1.dx, -r2.dx],[r1.dy, -r2.dy] ])
        b = np.array([ [r2.x-r1.x], [r2.y-r1.y] ])
        T = np.linalg.solve(A,b)
        t = T[0,0]
        u = T[1,0] # time, ray

        # t2 = (r1.y*r2.dx + r2.dy*r2.x - r2.y*r2.dx - r2.dy*r1.x ) / (r1.dx*r2.dy - r1.dy*r2.dx);     
        # t1 = (r1.x+r1.dx*t2-r2.x)/r2.dx
        [x, y] = r1.get_point(t)
        [x2, y2] = r2.get_point(u)

        if (math.sqrt((x-x2)**2 + (y-y2)**2) > 0.001):
            return [-1],False
        
        if (t>1 or t<0): # if time is more than 1, the lines wont intersect 
            return t,False
        
        if (u<0):
            return t,False

        return t,True
    
    def add_length(self,len:float):
        """_Increase the length of the line

        Args:
            len (float): new length
        """
        self.x[1]=self.x[0]+len*self.dx
        self.y[1]=self.y[0]+len*self.dy
        self = line2D((self.x[0],self.y[0]),(self.x[1],self.y[1]))

    
    def __eq__(self,line2):
        """Check if two lines are equal

        Args:
            line2 (line2D): second line
        """
        if (self.x == line2.x and self.y == line2.y):
            return True
        elif (self.x == np.flip(line2.x) and self.y == np.flip(line2.y)):
            return True
        else:
            return False
    
    def set_length(self,len:float):
        """Sets the length of a line

        Args:
            len (float): new length
        """
        self.x[1]=self.x[0]+len*self.dx/math.sqrt(self.dx**2+self.dy**2)
        self.y[1]=self.y[0]+len*self.dy/math.sqrt(self.dx**2+self.dy**2)
        self = line2D((self.x[0],self.y[0]),(self.x[1],self.y[1]))
    
    
    def get_point(self,t:Union[np.ndarray,float]):
        """Gets the point given a value between 0 and 1

        Args:
            t (Union[np.ndarray,float]): numpy array linspace(0,1,10) or any float value between 0 and 1

        Returns:
            Union[np.ndarray,float]: either a single value or an array of values
        """
        x2=self.dx*t+self.x[0]
        y2=self.dy*t+self.y[0]
        return x2,y2
    
    def get_points2(self,n:int):
        """Get `n` number of points along a line

        Args:
            n (int): _description_

        Returns:
            Union[np.ndarray,float]: either a single value or an array of values
        """
        t = np.linspace(0,1,n)
        x2=self.dx*t+self.x[0]
        y2=self.dy*t+self.y[0]
        return x2,y2
    
    def get_y(self,x:Union[np.ndarray,float]):
        """Given an x value, output a y value

        Args:
            x (Union[np.ndarray,float]): any value

        Returns:
            Union[np.ndarray,float]: either a single value or an array of values
        """
        return self.dy/self.dx * (x-self.x[0]) + self.y[0]
    
    def average(self,new_line):
        """Averages the line with another line 

        Args:
            new_line (line2D): new line 
        """
        self.x = (new_line.x+self.x)/2
        self.y = (new_line.y+self.y)/2
        self.p = (self.x[0],self.y[0]); self.q=(self.x[1],self.y[1])
        self.dx = self.x[1]-self.x[0]; self.dy = self.y[1]-self.y[0]

        self.length = math.sqrt((self.x[1]-self.x[0])*(self.x[1]-self.x[0]) + (self.y[1]-self.y[0])*(self.y[1]-self.y[0]))
        self.angle = math.atan2(self.y[1]-self.y[0],self.x[1]-self.x[0])
    
    def get_t(self,length:float) -> Union[float, npt.NDArray]:
        """Returns the t that give a certain length 

        Args:
            length (float): length of line

        Returns:
            Union[float, npt.NDArray]: time
        """
        return np.sqrt(length**2/(self.dx**2 + self.dy**2))
                    
    def plot2D(self):
        """Plot the line
        """
        _, ax1 = plt.subplots()
        ax1.plot(self.x, self.y,'or')
        ax1.plot(self.x, self.y,'-b')
        ax1.set_xlabel("x-label")
        ax1.set_ylabel("y-label")
        plt.show()


    def intersect_check(self,line2) -> bool:
        """Checks for intersection with another line

        Args:
            line2 (line2D): another line

        Returns:
            bool: True - intersects, False - no intersection
        """
        p1 = self.p; q1 = self.q
        p2 = line2.p; q2 = line2.q
        # Find the four orientations needed for general and special cases
        o1 = self.__orientation(p1, q1, p2)
        o2 = self.__orientation(p1, q1, q2)
        o3 = self.__orientation(p2, q2, p1)
        o4 = self.__orientation(p2, q2, q1)
        #  General case
        if (o1 != o2 and o3 != o4):
            return True
        
        # Special Cases
        # p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if (o1 == 0 and self.__onSegment(p1, p2, q1)):
            return True
        
        # p1, q1 and p2 are colinear and q2 lies on segment p1q1
        if (o2 == 0 and self.__onSegment(p1, q2, q1)):
            return True
        
        # p2, q2 and p1 are colinear and p1 lies on segment p2q2
        if (o3 == 0 and self.__onSegment(p2, p1, q2)):
            return True
        

        # p2, q2 and q1 are colinear and q1 lies on segment p2q2
        if (o4 == 0 and self.__onSegment(p2, q1, q2)):
            return True

        return False

    def flip(self):
        """
            reverses the direction of the line
        """
        return line2D(self.q,self.p)

    def line_intersect(self,line2) -> bool:
        """Checks if two lines intersect

        Args:
            line2 (line2D): another line

        Returns:
            (bool): True if intersect
        """
        return self.intersect_check(line2) # type: ignore
    
    def mag(self) -> float:
        """returns the magnitude

        Returns:
            float: magnitude
        """
        return math.sqrt(self.dx**2+self.dy**2)
    
    
    def angle_between(self,line2)-> float:
        """Returns the angle between two lines 

        Args:
            line2 (line2D): another line

        Returns:
            float: angle between two lines in degrees
        """
        return math.degrees(math.acos((self.dx*line2.dx + self.dy*line2.dy)/(self.mag()*line2.mag())))

    def __onSegment(self,p,q,r):
        """ Private Functions 
        Given three colinear points p, q, r, the function checks if point q lies on line segment 'pr'
        """
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False
    

    def __orientation(self,p,q,r):
        """
            To find orientation of ordered triplet (p, q, r).
            The function returns following values
                0 --> p, q and r are colinear
                1 --> Clockwise
                2 --> Counterclockwise
        """
        orientation = 0
        # See http://www.geeksforgeeks.org/orientation-3-ordered-points/ for details of below formula.
        val = (q[1] - p[1])*(r[0]-q[0]) - (q[0] - p[0])*(r[1] - q[1])
        if (val == 0):
            orientation = 0; # colinear
        elif (val>0):
            orientation = 1
        elif (val<0):
            orientation = 2
        
        # orientation = (val > 0)? 1: 2; % clock or counterclock wise
        return orientation
    
    def shrink_start(self,shrink_len:float):
        """calculates new starting point by shrinking the line

        Args:
            shrink_len (float): amount to shrink the line by
        """
        self.x[0] = self.x[1]-self.dx*(self.length-shrink_len)/self.length
        self.y[0] = self.y[1]-self.dy*(self.length-shrink_len)/self.length
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.length = math.sqrt((self.dx)*(self.dx) + (self.dy)*(self.dy))
        self.p = (self.x[0],self.y[0])
        self.q = (self.x[1],self.y[1])

    def shrink_end(self,shrink_len:float):        
        """Calculates new end point by shrinking hte line 

        Args:
            shrink_len (float): amount to shrink the line by
        """
        self.x[1] = self.x[0]+self.dx*(self.length-shrink_len)/self.length
        self.y[1] = self.y[0]+self.dy*(self.length-shrink_len)/self.length
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.length = math.sqrt((self.dx)*(self.dx) + (self.dy)*(self.dy))
        self.p = (self.x[0],self.y[0])
        self.q = (self.x[1],self.y[1])
    
    def fillet(self,prev_line,filletR:float) -> Tuple[bezier,bezier]:
        """Creates a fillet with the previous line, how the line terminates doesn't matter

        Args:
            prev_line (line2D): previous line
            filletR (float): fillet radius

        Returns:
            bezier: bezier curve describing the fillet
        """
        # Check the start and end points see where it intersects
        if (self.p == prev_line.p): 
            # Beginning of first line and beginning of 2nd line, intersections happen at start
            # shrink self(first line) at the beginning
            # shrink prev line at the end 
            ix = self.x[0]
            iy = self.y[0]
            self.shrink_start(filletR)
            prev_line.shrink_start(filletR)
            fillet = bezier([prev_line.x[0],ix,self.x[0]], [prev_line.y[0],iy,self.y[0]])
            return prev_line,fillet
        elif (self.p == prev_line.q):
            # Beginning of first line and end of 2nd line
            # shrink self(first line) at the end
            # shrink prev line at the beginning
            ix = self.x[0]
            iy = self.y[0]
            self.shrink_start(filletR)
            prev_line.shrink_end(filletR)
            fillet = bezier([self.x[0],ix,prev_line.x[1]], [self.y[0],iy,prev_line.y[1]])
            fillet = fillet.flip()
            return prev_line,fillet
        elif (self.q==prev_line.p):
            # End of first line and beginning of 2nd line
            ix = self.x[1]
            iy = self.y[1]
            self.shrink_end(filletR)
            prev_line.shrink_start(filletR)
            fillet = bezier([self.x[1],ix,prev_line.x[0]], [self.y[1],iy,prev_line.y[0]])
        else: #  (self.q==prev_line.q)
            # End of first line and end of 2nd line
            # Reverse the current line
            ix = self.x[1]
            iy = self.y[1]
            self.shrink_end(filletR)
            prev_line.shrink_end(filletR)
            fillet = bezier([self.x[1],ix,prev_line.x[1]], [self.y[1],iy,prev_line.y[1]])
        
        return prev_line,fillet