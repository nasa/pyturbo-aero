import unittest
import sys, copy
sys.path.insert(0, "../../")
import numpy as np
from pyturbo.aero import airfoil2D,airfoil3D,airfoil_wavy,stack_type, passage2D
from pyturbo.helper import *
import matplotlib.pyplot as plt
from math import pi

class TestDesign(unittest.TestCase):
    def test_2D_stator(self):
        # stator_hub = airfoil2D(alpha1=0,alpha2=72,axial_chord=0.038,stagger=58)
        
        # stator_hub.le_thickness_add(0.08)
        # ps_height = [0.0500,0.0200,-0.0100]
        # stator_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2)

        # ss_height=[0.2400, 0.2600, 0.2200, 0.1800]
        # stator_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)
        # stator_hub.le_thickness_match()
        # stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)

        # stator_hub.flow_guidance2(10)
        # fig = stator_hub.plot2D()
        # plt.show()
        # stator_hub.le_radius_estimate()

        # stator_hub.plot_camber()
       
        # stator_hub.plot_derivative2()
        # stator_hub.plot2D_channel(0.75)
        pass
    
    def test_2D_rotor(self):
        # rotor_hub = airfoil2D(alpha1=40,alpha2=60,axial_chord=5.119,stagger=20)
        
        # rotor_hub.le_thickness_add(0.08)
        # ps_height = [-0.0500,-0.0200,-0.0100]
        # rotor_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2)

        # ss_height=[0.20, 0.200, 0.18, 0.18]
        # rotor_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)
        # rotor_hub.le_thickness_match()
        # rotor_hub.te_create(radius=0.1,wedge_ss=2.5,wedge_ps=2.4)

        # # rotor_hub.flow_guidance2(10)
        # fig = rotor_hub.plot2D()
        # plt.show()
        pass

    def test_3D_Stator(self):
        # Hub Geometry
        # stator_hub = airfoil2D(alpha1=0,alpha2=72,axial_chord=0.038,stagger=58)
        # stator_hub.le_thickness_add(0.04)
        # ps_height = [0.0500,0.0200,-0.0100]
        # ps_height = [0.0500,0.0200,-0.0100]
        # stator_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2) 

        # ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        # stator_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        # stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        # stator_hub.le_thickness_match()
        # stator_hub.flow_guidance2(10)

        # # Tip Geometry
        # stator_tip = airfoil2D(alpha1=5,alpha2=72,axial_chord=0.036,stagger=56)
        # stator_tip.le_thickness_add(0.04)
        # ps_height = [0.0500,0.0200,-0.0100]
        # ps_height_loc = exp_ratio(1.2,len(ps_height)+2,0.95)
        # ps_height_loc = np.append(ps_height_loc,[1])
        # stator_tip.ps_thickness_add(thicknessArray=ps_height,camberPercent=ps_height_loc)

        # ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        # stator_tip.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        # stator_tip.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        # stator_tip.le_thickness_match()
        # stator_tip.flow_guidance2(6)
        # plt.clf()
        # plt.close('all')
        # # Begin 3D design
        # stator3D = airfoil3D([stator_hub,stator_tip],[0,1],0.05)
        # stator3D.stack(stack_type.centroid)
        # # stator3D.lean_add([0, 0.05, 1], [0,0.5,1])
        # stator3D.create_blade(100,80,20)
        # fig = stator3D.plot3D()
        # plt.show()
        pass

    def test_stator3D_wavy_const_te(self):
        '''
            Wavy with constant trailing edge
        '''
        # # Hub Geometry
        # stator_hub = airfoil2D(alpha1=0,alpha2=72,axial_chord=0.038,stagger=58)
        # stator_hub.le_thickness_add(0.04)
        # ps_height = [0.0500,0.0200,-0.0100]
        # stator_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2) 

        # ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        # stator_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        # stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        # stator_hub.le_thickness_match()
        # stator_hub.flow_guidance2(10)
        # # stator_hub.plot2D()

        # # Tip Geometry
        # stator_tip = airfoil2D(alpha1=5,alpha2=72,axial_chord=0.036,stagger=56)
        # stator_tip.le_thickness_add(0.04)
        # ps_height = [0.0500,0.0200,-0.0100]
        # stator_tip.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2)

        # ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        # stator_tip.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        # stator_tip.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        # stator_tip.le_thickness_match()
        # stator_tip.flow_guidance2(6)

        # # Begin 3D design
        # span = 0.05
        # stator3D = airfoil3D([stator_hub,stator_tip],[0,1],0.05)
        # stator3D.stack(stack_type.trailing_edge)
        # stator3D.lean_add([0, 0.05, 0], [0,0.5,1])
        # stator3D.create_blade(100,100,20)
        # # stator3D.plot3D()

        # # Wavy Geometry
        # stator3D_wavy = airfoil_wavy(stator3D)
        # stator3D_wavy.spanwise_spline_fit()
        # t = np.linspace(0,1,1000)
        # ss_ratio = 2*span*np.cos(3*math.pi*t)
        # ps_ratio = 2*span*np.cos(3*math.pi*t)
        # te_ratio = 0.5*span*np.cos(3*math.pi*t)
        # le_ratio = (ss_ratio+ps_ratio)/2.0
        # ssratio_wave = wave_control(ss_ratio)  # These are spanwise waves
        # psratio_wave = wave_control(ps_ratio)  
        # teratio_wave = wave_control(te_ratio)
        # leratio_wave = wave_control(le_ratio)
        
        # LE_wave_angle = leratio_wave*0.0
        # TE_wave_angle = leratio_wave*0.0

        # stator3D_wavy.stretch_thickness_chord_te(SSRatio=ssratio_wave.get_wave(0,1),PSRatio=psratio_wave.get_wave(0.2,0.8,True),
        #     LERatio=leratio_wave.get_wave(0,1),TERatio=teratio_wave.get_wave(0,1),LE_wave_angle=LE_wave_angle,TE_wave_angle=TE_wave_angle,TE_smooth=0.90)
        
        # stator3D.plot3D_ly(only_blade=True)
        pass


    def test_wavy(self):
        '''
            Wavy without constant trailing edge
        '''
        # # Hub Geometry
        # stator_hub = airfoil2D(alpha1=0,alpha2=72,axial_chord=0.038,stagger=58)
        # stator_hub.le_thickness_add(0.04)
        # ps_height = [0.0500,0.0200,-0.0100]
        # stator_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2) 

        # ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        # stator_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        # stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        # stator_hub.le_thickness_match()
        # stator_hub.flow_guidance2(10)
        # # stator_hub.plot2D()

        # # Tip Geometry
        # stator_tip = airfoil2D(alpha1=5,alpha2=72,axial_chord=0.036,stagger=56)
        # stator_tip.le_thickness_add(0.04)
        # ps_height = [0.0500,0.0200,-0.0100]
        # stator_tip.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2)

        # ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        # stator_tip.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        # stator_tip.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        # stator_tip.le_thickness_match()
        # stator_tip.flow_guidance2(6)

        # # Begin 3D design
        # span = 0.05
        # stator3D = airfoil3D([stator_hub,stator_tip],[0,1],0.05)
        # stator3D.stack(stack_type.trailing_edge)
        # stator3D.lean_add([0, 0.05, 0], [0,0.5,1])
        # stator3D.create_blade(100,100,20)
        # # stator3D.plot3D()

        # # Wavy Geometry
        # stator3D_wavy = airfoil_wavy(stator3D)
        # stator3D_wavy.spanwise_spline_fit()
        # t = np.linspace(0,1,1000)
        # ss_ratio = 2*span*np.cos(3*math.pi*t)
        # ps_ratio = 2*span*np.cos(3*math.pi*t)
        # te_ratio = 0.5*span*np.cos(3*math.pi*t)
        # le_ratio = (ss_ratio+ps_ratio)/2.0
        # ssratio_wave = wave_control(ss_ratio)  # These are spanwise waves
        # psratio_wave = wave_control(ps_ratio)  
        # teratio_wave = wave_control(te_ratio)
        # leratio_wave = wave_control(le_ratio)
        
        # LE_wave_angle = leratio_wave*0.0
        # TE_wave_angle = leratio_wave*0.0

        # stator3D_wavy.stretch_thickness_chord(SSRatio=ssratio_wave.get_wave(0,1),PSRatio=psratio_wave.get_wave(0.2,0.8,True),
        #     LERatio=leratio_wave.get_wave(0,1),TERatio=teratio_wave.get_wave(0,1),LE_wave_angle=LE_wave_angle,TE_wave_angle=TE_wave_angle,TE_smooth=0.5)
        
        # stator3D_wavy.plot3D(only_blade=True)
        pass
    
    def test_wavy_whisker(self):
        '''
            Wavy blade that maintains cross sectional area
        '''
        # Hub Geometry
        stator_hub = airfoil2D(alpha1=0,alpha2=72,axial_chord=0.038,stagger=58)
        stator_hub.le_thickness_add(0.04)
        ps_height = [0.0500,0.0200,-0.0100]
        stator_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2) 

        ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        stator_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        stator_hub.le_thickness_match()
        stator_hub.flow_guidance2(10)
        # stator_hub.plot2D()

        # Tip Geometry
        stator_tip = airfoil2D(alpha1=5,alpha2=72,axial_chord=0.036,stagger=56)
        stator_tip.le_thickness_add(0.04)
        ps_height = [0.0500,0.0200,-0.0100]
        stator_tip.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2)

        ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        stator_tip.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        stator_tip.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        stator_tip.le_thickness_match()
        stator_tip.flow_guidance2(6)

        # Begin 3D design
        span = 0.05
        stator3D_wavy = airfoil_wavy([stator_hub,stator_tip],[0,1],0.05)
        stator3D_wavy.stack(stack_type.trailing_edge)
        stator3D_wavy.lean_add([0, 0.05, 0], [0,0.5,1])
        stator3D_wavy.create_blade(100,100,20)
        stator3D_wavy.plot3D()

        # Wavy Geometry
        stator3D_wavy.spanwise_spline_fit()
        t = np.linspace(0,1,1000)
        ps_ratio = 2*span*np.cos(3*math.pi*t)
        te_ratio = 0.5*span*np.cos(3*math.pi*t)
        le_ratio = 3*span*np.cos(12*math.pi*t)
        psratio_wave = wave_control(ps_ratio)  
        teratio_wave = wave_control(te_ratio)
        leratio_wave = wave_control(le_ratio)
        
        LE_wave_angle = leratio_wave*0.0
        TE_wave_angle = leratio_wave*0.0

        stator3D_wavy.whisker_blade(PSRatio=psratio_wave.get_wave(0.2,0.8,True),
            LERatio=leratio_wave.get_wave(0,1),TERatio=teratio_wave.get_wave(0,1),LE_wave_angle=LE_wave_angle,TE_wave_angle=TE_wave_angle,TE_smooth=0.9)
        
        stator3D_wavy.plot3D_ly(only_blade=True)
        pass
    
    def test_import_geometry(self):
        # rotor = airfoil3D.import_geometry(folder='import_7%',axial_chord=124) # Set axial chord to 124 mm 
        # rotor.plot3D_ly()
        pass

    def test_imported_blade_shell(self):
        # rotor = airfoil3D.import_geometry(folder='import_7%',axial_chord=124,span=114,ss_ps_split=105) 
        # rotor.rotate(90)
        # # [ss_x_new,ss_y_new,ss_z,ps_x_new,ps_y_new,ps_z] = rotor.get_shell_2D(percent_span=0.5,shell_thickness=-4)
        # rotor.plot_shell_2D(percent_span=0.0,shell_thickness=-3)
        pass

    def test_blade_shell(self):
        '''
            You can create 2D shell from a 3D Design
        '''
        # # Hub Geometry
        # stator_hub = airfoil2D(alpha1=0,alpha2=72,axial_chord=0.038,stagger=58)
        # stator_hub.le_thickness_add(0.04)
        # ps_height = [0.0500,0.0200,-0.0100]
        # stator_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2) 

        # ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        # stator_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        # stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        # stator_hub.le_thickness_match()
        # stator_hub.flow_guidance2(10)
        # # stator_hub.plot2D()

        # # Tip Geometry
        # stator_tip = airfoil2D(alpha1=5,alpha2=72,axial_chord=0.036,stagger=56)
        # stator_tip.le_thickness_add(0.04)
        # ps_height = [0.0500,0.0200,-0.0100]
        # stator_tip.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2)

        # ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        # stator_tip.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        # stator_tip.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        # stator_tip.le_thickness_match()
        # stator_tip.flow_guidance2(6)

        # # Begin 3D design
        # span = 0.05
        # stator3D = airfoil3D([stator_hub,stator_tip],[0,1],0.05)
        # stator3D.stack(stack_type.trailing_edge)
        # stator3D.lean_add([0, 0.05, 0], [0,0.5,1])
        # stator3D.create_blade(100,100,20)
        # # stator3D.plot3D()
        # # [ss_x_new,ss_y_new,ps_x_new,ps_y_new] = stator3D.get_shell_2D(percent_span=0.5,shell_thickness=-0.002)
        # stator3D.plot_shell_2D(0.2,-0.002)
        pass

    def test_channel_2D(self):
        '''
            Design a channel and places a stator and rotor inside
        '''
        # Stator Hub Geometry
        cax_stator = 0.038
        stator_hub = airfoil2D(alpha1=0,alpha2=72,axial_chord=cax_stator,stagger=58)
        stator_hub.le_thickness_add(0.04)
        ps_height = [0.0500,0.0200,-0.0100]
        stator_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2) 

        ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        stator_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        stator_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        stator_hub.le_thickness_match()
        stator_hub.flow_guidance2(10)
        # stator_hub.plot2D()

        # Stator Tip Geometry
        stator_tip = airfoil2D(alpha1=5,alpha2=72,axial_chord=0.036,stagger=56)
        stator_tip.le_thickness_add(0.04)
        ps_height = [0.0500,0.0200,-0.0100]
        stator_tip.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2)

        ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        stator_tip.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        stator_tip.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        stator_tip.le_thickness_match()
        stator_tip.flow_guidance2(6)

        # Begin Stator 3D design
        stator_span = cax_stator
        stator3D = airfoil3D([stator_hub,stator_tip],[0,1],stator_span)
        stator3D.stack(stack_type.trailing_edge)
        stator3D.lean_add([0, 0.05, 0], [0,0.5,1])
        stator3D.create_blade(100,100,20)
        # stator3D.plot3D()
     
        # Rotor Hub Geometry
        cax_rotor_hub = stator_span
        rotor_hub = airfoil2D(alpha1=30,alpha2=72,axial_chord=cax_rotor_hub,stagger=40)
        rotor_hub.le_thickness_add(0.04)
        ps_height = [0.0500,0.01,0.05]
        rotor_hub.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2) 

        ss_height=[0.3, 0.25, 0.20, 0.25]
        rotor_hub.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        rotor_hub.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        rotor_hub.le_thickness_match()
        rotor_hub.flow_guidance2(3)
        #rotor_hub.plot2D()

        # Rotor Tip Geometry
        rotor_tip = airfoil2D(alpha1=30,alpha2=72,axial_chord=0.036,stagger=45)
        rotor_tip.le_thickness_add(0.04)
        ps_height = [0.0500,0.01,0.05]
        rotor_tip.ps_thickness_add(thicknessArray=ps_height,expansion_ratio=1.2)

        ss_height=[0.2400, 0.2000, 0.1600, 0.1400]
        rotor_tip.ss_thickness_add(thicknessArray=ss_height,camberPercent=0.8,expansion_ratio=1.2)

        rotor_tip.te_create(radius=0.001,wedge_ss=2.5,wedge_ps=2.4)
        rotor_tip.le_thickness_match()
        rotor_tip.flow_guidance2(3)

        # Begin Rotor 3D design
        rotor_span = 0.05
        rotor3D_wavy = airfoil_wavy([rotor_hub,rotor_tip],[0,1],rotor_span)
        rotor3D_wavy.stack(stack_type.trailing_edge)
        rotor3D_wavy.lean_add([0, 0.05, 0], [0,0.5,1])
        rotor3D_wavy.create_blade(100,100,20)
        rotor3D_wavy.flip_cw()
        # rotor3D.plot3D(only_blade=True)


        # Wavy Geometry
        t = np.linspace(0,1,1000)
        ss_ratio = 2*rotor_span*np.cos(5*pi*t)
        ps_ratio = 2*rotor_span*np.cos(3*pi*t)
        te_ratio = 0.5*rotor_span*np.cos(3*pi*t)
        le_ratio = 3*rotor_span*np.cos(12*pi*t)
        ssratio_wave = wave_control(ss_ratio)
        psratio_wave = wave_control(ps_ratio)
        teratio_wave = wave_control(te_ratio)
        leratio_wave = wave_control(le_ratio)
        
        LE_wave_angle = leratio_wave*0.0
        TE_wave_angle = leratio_wave*0.0

        rotor3D_wavy.stretch_thickness_chord_te(SSRatio=ssratio_wave.get_wave(0,1),PSRatio=psratio_wave.get_wave(0.2,0.8,True),
            LERatio=leratio_wave.get_wave(0,1),TERatio=teratio_wave.get_wave(0,1),LE_wave_angle=LE_wave_angle,TE_wave_angle=TE_wave_angle,TE_smooth=0.90)

        # Create 2D Channel
        hub_control_z = [0]
        hub_control_r = [0.5]

        hub_control_z.append(hub_control_z[-1]+cax_stator) # Stator inlet
        hub_control_r.append(0.5)
        hub_control_z.append(hub_control_z[-1]+cax_stator/2) # Stator mid
        hub_control_r.append(hub_control_r[-1]*.98)

        hub_control_z.append(hub_control_z[-1]+ cax_stator/2 + cax_stator*1/6) # in between stator and rotor
        hub_control_r.append(hub_control_r[-1]*.95)

        hub_control_z.append(hub_control_z[-1] + cax_stator*1/6 +cax_rotor_hub/2) # in between rotor mid
        hub_control_r.append(hub_control_r[-1]*1.05)

        hub_control_z.append(hub_control_z[-1]+cax_rotor_hub) #  rotor exit
        hub_control_r.append(hub_control_r[-1])

        hub_control_z.append(hub_control_z[-1]+cax_rotor_hub*2.5) # domain exit
        hub_control_r.append(hub_control_r[-1])


        shroud_control_z = [0]
        shroud_control_r = [hub_control_r[-1]+cax_stator]

        shroud_control_z.append(shroud_control_z[-1]+cax_stator) # Stator inlet
        shroud_control_r.append(hub_control_r[-1]+cax_stator)

        shroud_control_z.append(shroud_control_z[-1]+cax_stator/2) # Stator mid
        shroud_control_r.append(shroud_control_r[-1]*1.05)

        shroud_control_z.append(shroud_control_z[-1]+ cax_stator/2 + cax_stator*1/6) # in between stator and rotor
        shroud_control_r.append(shroud_control_r[-1]*1.0)

        shroud_control_z.append(shroud_control_z[-1]+ cax_stator*1/6 + cax_rotor_hub/2) # rotor mid
        shroud_control_r.append(shroud_control_r[-1]*1.0)

        shroud_control_z.append(shroud_control_z[-1]+cax_rotor_hub) # rotor exit
        shroud_control_r.append(shroud_control_r[-1]*1.0)

        shroud_control_z.append(shroud_control_z[-1]+cax_rotor_hub*2.5) # domain exit
        shroud_control_r.append(shroud_control_r[-1]*1.0)

        hub_bezier = bezier(hub_control_z,hub_control_r)
        shroud_bezier = bezier(shroud_control_z,shroud_control_r)

        t = np.linspace(0,1,500)
        [hz,hr] = hub_bezier.get_point(t)
        [sz,sr] = shroud_bezier.get_point(t)

        channel = passage2D([stator3D,rotor3D_wavy],[cax_stator/3],[41,60])
        channel.add_endwalls(zhub=hz,rhub=hr,zshroud=sz,rshroud=sr,zhub_control=hub_control_z,rhub_control=hub_control_r,rshroud_control=shroud_control_r,zshroud_control=shroud_control_z)
        channel.plot2D_channel()
        channel.blade_fit(cax_stator)
        # channel.airfoils[0].plot3D(only_blade=True)
        # channel.plot3D_ly()
        channel.plot3D()
        # channel.plot2D()
        channel.ExportToDatFile()

        pass
if __name__ == '__main__':
    unittest.main()
