import unittest
import sys
sys.path.insert(0, "../")
import numpy as np
from pyturbo.aero import *
from pyturbo.helper import *


class TestLibrary(unittest.TestCase):
    def test_pspline(self):
        x = np.linspace(0,10,10)
        y = np.sqrt(x**2+2)

        p = pspline(x,y)
        [pt,dudt] = p.get_point(0.5)
    def test_bezier3(self):
        x = np.linspace(0,10,10)
        y = x*2
        z = x

        p = bezier3(x,y,z)
        [bx,by,bz] = p.get_point(1)
        print('done')
    

if __name__ == '__main__':
    # test2D = Test_Bezier()
    # test2D.build_HPT_stator()
    unittest.main()
