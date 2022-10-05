import unittest
import sys, copy
sys.path.insert(0, "../../")
import numpy as np
from pyturbo.aero import *
from pyturbo.cooling import *
from pyturbo.helper import *

class TestDesign(unittest.TestCase):

    def test_cross_section_circle(self):
        # cs = cross_section()
        # cs.create_circle(0.2)
        # cs.plot2D()
        pass
    
    def test_cross_section_rectangle(self):
        # cs = cross_section()
        # cs.create_rectangle(0.2,0.2)
        # cs.plot2D()
        pass
    
    def test_cross_section_rectangle_fillet(self):
        # cs = cross_section()
        # cs.create_rectangle(0.2,0.2,0.01)
        # cs.plot2D()
        pass

    def test_branch_periodic_loop(self):
        # b = branch()
        # b.pattern_periodic_loop(4,150,800,4)
        # b.plot_2D_path()
        pass

    def test_branch_close_loop_no_return(self):    
        # b = branch()
        # b.pattern_close_loop(8,150,800,4,return_gap=0)
        # b.plot_2D_path()
        pass

    def test_branch_close_loop_return(self):    
        # b = branch()
        # b.pattern_close_loop(3,height=150,width=800,filletR=4,return_gap=20)
        # b.plot_2D_path()
        pass

    def test_heat_pipe_periodic_loop(self):
        '''
            Test case for wrapping a branch around the inside of the blade
        '''
        cs = cross_section()
        cs.create_circle(2)
        # Design the heat pipe
        b = branch()
        ## Note, you can use this one or
        # b.pattern_periodic_loop(nloops=3,height=152*0.95,width=365.89,filletR=4)    
        ## Use this version (This function allows the estimation of the height and width to happen automatically)
        b.setup_pattern_periodic_loop(nloops=3,filletR=4)
        b.add_cross_section([cs,cs],[0,1]) # adds the same cross section to the start and finish of the branch          
        #b.create_pathline()
        ## Code below is for debugging 
        # b.plot_2D_path()        
        # b.plot_sw_pathline()
        # b.plot_pathline_3D()
        ## Import the airfoil
        rotor = airfoil3D.import_geometry(folder='import_7%',axial_chord=130,span=[0,114],ss_ps_split=105)
        # rotor.plot_shell_2D(percent_span=0,shell_thickness=-5)
        rotor.spanwise_spline_fit()
        rotor.spanwise_spline_fit_shell(shell_thickness=-3,smooth=0.1)
        #rotor.rotate(90) # This only rotates the blade 

        hp_airfoil = heatpipe_airfoil(rotor)        
        # rotor.plot3D()

        hp_airfoil.wrap_branch_shell(b,percent_span=[0.05,0.95], wall_dist=-3,rotate=0)
        # hp_airfoil.plot_heatpipe()
        hp_airfoil.export_solidworks('hp_loop')
        pass

    def test_heat_pipe_close_loop(self):
        # cs = cross_section()
        # cs.create_circle(2)
        # # Suction side heat pipe
        # b_ss = branch()
        # ## Note, you can use this one or
        # b_ss.setup_pattern_close_loop(nloops=3,filletR=4,return_gap=0)
        # b_ss.add_cross_section([cs,cs],[0,1]) # adds the same cross section to the start and finish of the branch  

        # b_ps = branch()
        # ## Note, you can use this one or
        # b_ps.setup_pattern_close_loop(nloops=5,filletR=10,return_gap=0)
        # b_ps.add_cross_section([cs,cs],[0,1]) # adds the same cross section to the start and finish of the branch  

        # ## Code below is for debugging 
        # # b.plot_2D_path()        
        # # b.plot_sw_pathline()
        # # b.plot_pathline_3D()
        # ## Import the airfoil
        # rotor = airfoil3D.import_geometry(folder='import_7%',axial_chord=124,span=114,ss_ps_split=105)
        # rotor.rotate(90)
        # rotor.spanwise_spline_fit()
        # rotor.spanwise_spline_fit_shell(shell_thickness=-3,smooth=0.1)
        # # rotor.plot3D()

        # hp_airfoil = heatpipe_airfoil(rotor)        

        # hp_airfoil.add_branch_shell(b_ss,axial_percent=[0.05, 0.9], percent_span=[0.05,0.95],wall_dist=-3,shell_type=branch_shell_type.suction_side)

        # hp_airfoil.add_branch_shell(b_ps,axial_percent=[0.12, 0.9], percent_span=[0.05,0.95],wall_dist=-3,shell_type=branch_shell_type.pressure_side)

        # hp_airfoil.plot_heatpipe()
        pass

    def test_heat_pipe_close_loop_return(self):
        # cs = cross_section()
        # cs.create_circle(2)
        # # Suction side heat pipe
        # b_ss = branch()
        # ## Note, you can use this one or
        # b_ss.setup_pattern_close_loop(nloops=3,filletR=4,return_gap=15)
        # b_ss.add_cross_section([cs,cs],[0,1]) # adds the same cross section to the start and finish of the branch  

        # b_ps = branch()
        # ## Note, you can use this one or
        # b_ps.setup_pattern_close_loop(nloops=5,filletR=10,return_gap=15)
        # b_ps.add_cross_section([cs,cs],[0,1]) # adds the same cross section to the start and finish of the branch  

        # ## Code below is for debugging 
        # b_ss.plot_2D_path()        
        # # b.plot_sw_pathline()
        # # b.plot_pathline_3D()
        # ## Import the airfoil
        # rotor = airfoil3D.import_geometry(folder='import_7%',axial_chord=124,span=114,ss_ps_split=105)
        # rotor.rotate(90)
        # rotor.spanwise_spline_fit()
        # rotor.spanwise_spline_fit_shell(shell_thickness=-3,smooth=0.1)
        # # rotor.plot3D()

        # hp_airfoil = heatpipe_airfoil(rotor)        

        # hp_airfoil.add_branch_shell(b_ss,axial_percent=[0.05, 0.9], percent_span=[0.05,0.95],wall_dist=-3,shell_type=branch_shell_type.suction_side)

        # hp_airfoil.add_branch_shell(b_ps,axial_percent=[0.12, 0.9], percent_span=[0.05,0.95],wall_dist=-3,shell_type=branch_shell_type.pressure_side)

        # hp_airfoil.plot_heatpipe()
        pass

    
if __name__ == '__main__':
    unittest.main()