'''
    Code translated from https://whittle.digital/2024/Radial_Compressor_Designer/ 
'''
import math
import json 
import numpy as np 
from .arc import arc

def entropy(P:float,T:float,cp:float,Tref:float=300,Pref:float=1E5,Rgas:float=287.15):
    return cp * math.log(T/Tref) - Rgas * math.log(P/Pref)

def calculate_efficiency(Cgamma:float=0.6):
    fit_data = json.loads('fit_eta_tt.json')
    k = fit_data['k']
    c = fit_data['c']
    xl = fit_data['xl'][0]
    xu = fit_data['xu'][0]
    
    I = len(fit_data.keys)
    J = len(c)
    
    eta = 0
    pval = np.zeros(len(fit_data.keys))
    pval[fit_data['vars'].index('PR_tt')] = PRtt
    pval[fit_data['vars'].index('htr1')] = HTR1
    pval[fit_data['vars'].index('Marel1')] = Marel1     # N1 
    pval[fit_data['vars'].index('tip')] = tau           # Tip Clearance 
    pval[fit_data['vars'].index('Alpha2rel')] = Alrel2  # Outlet Relative Yaw
    pval[fit_data['vars'].index('DHimp')] = DH          # DeHaller Number 
    pval[fit_data['vars'].index('phi1')] = phi1         # Inlet Flow coefficient
    pval[fit_data['vars'].index('Co1')] = Cgamma                      # blade circulation
    
    for j in range(J):
        deta = 1.
        for i in range(I):
            deta = deta * pval[i][k[j][i]]
        
        eta = eta + c[j]*deta
    
    return eta

    
    

def create_passage(PR:float=2.4, phi1:float=0.7, 
                       M1_rel:float=0.6, HTR1:float=0.5,
                       deHaller:float=1, outlet_yaw:float=-64, 
                       blade_circulation:float=0.6, tip_clearance:float=0.01,
                       P01:float=1, T01:float=300, mdot:float=5):
        """_summary_

        Args:
            PR (float, optional): _description_. Defaults to 2.4.
            phi1 (float, optional): inlet flow coefficient. Defaults to 0.7.
            M1_rel (float, optional): _description_. Defaults to 0.6.
            HTR1 (float, optional): Inlet hub to tip ratio. Defaults to 0.5.
            deHaller (float, optional): _description_. Defaults to 1.
            outlet_yaw (float, optional): _description_. Defaults to -64.
            blade_circulation (float, optional): _description_. Defaults to 0.6.
            tip_clearance (float, optional): _description_. Defaults to 0.01.
            P01 (float, optional): _description_. Defaults to 1.
            T01 (float, optional): _description_. Defaults to 300.
            mdot (float, optional): _description_. Defaults to 5.
        """
        import os
        with open('./pyturbo/helper/fit_eta_tt.json','r') as fp:            
            fit_data = json.load(fp)
            k = fit_data['k']
            c = fit_data['c']
            xl = np.array(fit_data['xl'][0])
            xu = np.array(fit_data['xu'][0])
            dx = xu-xl
            dx[fit_data['vars'].index('PR_tt')]
            PR = np.linspace(1.5,3.5,)

        gam = 1.4
        cp = 1005 # J/(Kg K)
        Tref = 300
        Pref = 1E5

        # Meanline 
        gae = gam/(gam-1)
        Rgas = cp/(gae)        
        
        gae = gam/(gam-1); gm1 = gam-1
        
        # Inlet absolute Mach
        Ma1 = M1_rel*(1+1/(phi1**2))**(-0.5)

        # Inlet static temperature
        T1 = T01/(1.+ 0.5*(gam-1)*Ma1**2.)

        # Inlet speed of sound
        a1 = (gam*Rgas*T1)**0.5

        #Inlet density
        P1 = (P01*1e5)/(1.+ 0.5*(gam-1)*Ma1**2.)**gae
        rho1 = P1 / Rgas / T1

        # Inlet velocity
        Vx1 = Ma1 * a1

        # Inlet blade spaeed
        U1 = Vx1 / phi1

        # Inlet area
        A1 = mdot / rho1 / Vx1

        # Inlet radii
        rrms1 = (A1/math.pi/2.*(1.+HTR1**2)/(1-HTR1**2))**0.5
        rhub1 = (A1/math.pi*1./(1./HTR1**2 - 1.))**0.5
        rtip1 = (A1/math.pi*1./(1.-HTR1**2))**0.5

        Vrel1 = a1 * M1_rel

        # Shaft speed
        Omega = U1/ rrms1

        outlet_yaw_rad = outlet_yaw / 360. * 2. * math.pi # Outlet relative yaw

        # Outlet velocities
        Vrel2 = Vrel1 * deHaller
        Vr2 = Vrel2*math.cos(outlet_yaw_rad)
        Vtrel2 = Vrel2*math.sin(outlet_yaw_rad)


        # Predicted effy
        eta_now = calculate_efficiency()

        # Diffuser outlet stagnation state
        P03 = (P01*1e5)* PR
        T03s = T01*(P03/P01/1e5)**(1/gae)
        T03 = T01 + (T03s-T01)/eta_now

        # Loss split to set rotor outlet state
        # Assumed constant
        zeta = 0.75
        s1 = entropy(P01*1e5, T01)
        s3 = entropy(P03, T03)
        s2 = s1 + zeta*(s3-s1)
        print(f"s2-s1: {s2-s1}, s3-s1: {s3-s1}")
        
        T02 = T03
        h02 = cp*(T02-Tref)
        P02 = Pref * math.exp((cp*math.log(T02/Tref) - s2 )/Rgas)

        # Inlet rothalpy
        I1 = cp*(T1-Tref) + 0.5 * Vrel1**2 - 0.5*U1**2

        # Iterate on exit rothalpy
        h2 = cp*(T02-Tref);  # initial guess
        U2 = 0.
        for i in range(10):
            U2 = (2.0 * (h2 + 0.5 * Vrel2**2 - I1))**0.5
            h2new = h02 - 0.5 * (Vr2**2.0 + (Vtrel2 + U2) ** 2.0)
            h2 = h2new
        
        r2 = U2/Omega
        V2 = (2.*(h02-h2))**0.5
        V_cpTo2 = V2 / (cp*T02)**0.5
        xx = V_cpTo2**2
        Ma2 =(xx/(1-0.5*xx)/gm1)**0.5
        T2 = T02/(1 + 0.5*gm1 * Ma2**2)
        P2 = P02*(1 + 0.5*gm1 * Ma2**2)**(-gae)
        rho2 = P2/Rgas/T2

        span2 = mdot/2/math.pi/r2/rho2/Vr2
        
        Alrel1_deg = math.atan2(-U1, Vx1)*180./math.pi
        Alrel2_deg = math.atan2(Vtrel2, Vr2)*180./math.pi

        Power = mdot*cp*(T03-T01)

        RPM = Omega*60./(2*math.pi)

        V3 = 0.75 * V2
        T3 = T03 - V3**2/(2*cp)
        Ma3 = V3*(gam*Rgas*T3)**(-0.5)
        P3 = P03*(1 + 0.5*gm1 * Ma3**2)**(-gae)
        T3s = T1*(P3/P1)**((gam-1)/gam)
        eta_ts = (T3s-T01)/(T03-T01)

        Rhath = 1. - rhub1/r2
        Rhatc = 1. - rtip1/r2
        Span_true = span2/r2
        Xhatc = (Rhath+Rhatc)/2 - Span_true/2   # Xscale
        Xhath = (Rhath+Rhatc)/2 + Span_true/2   # Xscale
        
        shroud = arc(0,1,Rhatc,270,360)
        hub = arc(0,1,Rhatc,270,360)
        xsh, ysh = shroud.get_point(np.linspace(0,1,100))
        
        xhub, yhub = hub.get_point(np.linspace(0,1,100))
        
        xsh *= Xhatc/Rhatc
        xhub *= Xhath/Rhath
        # drawArc(0., 1.0, Rhatc, Xhatc/Rhatc)
        # drawArc(0., 1.0, Rhath, Xhath/Rhath)
        hub = np.hstack([xhub,yhub])
        shroud = np.hstack([xsh,ysh])
        return hub,shroud, V3,T3,P3,Ma3,eta_ts,eta_now,Power,RPM
        