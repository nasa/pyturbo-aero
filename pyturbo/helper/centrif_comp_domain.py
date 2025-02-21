from pyturbo.aero.airfoil_mp import Centrif
from pyturbo.helper import line2D
import numpy as np 

def extend_domain(cen:Centrif, extension_len1:float, extension_len2:float):
    func_drhub = cen.func_rhub.derivative(1)
    func_dxhub = cen.func_xhub.derivative(1)
    func_drshroud = cen.func_rhub.derivative(1)
    func_dxshroud = cen.func_xhub.derivative(1)

    dr = func_drhub(0)
    dx = func_dxhub(0)
    inlet_hub = np.array([[dx*extension_len1, dr*extension_len1],
           [cen.func_xhub(0),cen.func_rhub(0) ]])
    
    dr = func_drshroud(0)
    dx = func_dxshroud(0)
    inlet_shroud = 
    

