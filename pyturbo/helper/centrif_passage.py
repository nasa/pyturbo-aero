'''
    Code translated from https://whittle.digital/2024/Radial_Compressor_Designer/ 
'''
import math
import json 
import numpy as np 
from .arc import arc

def entropy(P:float,T:float,cp:float,Tref:float=300,Pref:float=1E5,Rgas:float=287.15):
    return cp * math.log(T/Tref) - Rgas * math.log(P/Pref)

def calculate_efficiency():
    fit_data = json.loads('fit_eta_tt.json')
    k = fit_data['k']
    c = fit_data['c']
    xl = fit_data['xl'][0]
    xu = fit_data['xu'][0]
    
    I = len(fit_data.keys)
    J = len(c)
    
    eta = 0
    pval = np.zeros(len(fit_data.keys))
    pval[PRtt.ind] = PRtt.pval
    pval[HTR1.ind] = HTR1.pval
    pval[Marel1.ind] = Marel1.pval
    pval[tau.ind] = tau.pval
    pval[Alrel2.ind] = Alrel2.pval
    pval[DH.ind] = DH.pval
    pval[phi1.ind] = phi1.pval
    pval[Cgamma.ind] = Cgamma.pval # blade circulation
    
    for j in range(J):
        deta = 1.
        for i in range(I):
            deta = deta * pval[i][k[j][i]]
        
        eta = eta + c[j]*deta
    
    return eta

def calculate_meanline(M1_rel:float,phi1:float,T01:float,P01:float,mdot:float,HTR1:float,DH:float,Alrel2:float,PR:float,cp:float=1005,gam:float=1.4,Rgas:float=287.15,Tref:float=300,Pref:float=1E5):
    """Calculate the meanline performance 

    Args:
        M1_rel (float): inlet relative mach number 
        phi1 (float): Inlet Flow coefficient 
        T01 (float): Inlet Total Temperature
        P01 (float): Inlet Total Pressure
        mdot (float): desired massflow rate
        HTR1 (float): Inlet Hub to Tip ratio 
        DH (float): Rotor de Haller number 
        Alrel2 (float): Outlet Relative Yaw
        PR (float): Pressure Ratio
        cp (float, optional): _description_. Defaults to 1005.
        gam (float, optional): _description_. Defaults to 1.4.
        Rgas (float, optional): _description_. Defaults to 287.15.
        Tref (float, optional): _description_. Defaults to 300.
        Pref (float, optional): _description_. Defaults to 1E5.
    """
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

    Alrel2_rad = Alrel2 / 360. * 2. * math.pi # Outlet relative yaw

    # Outlet velocities
    Vrel2 = Vrel1 * DH
    Vr2 = Vrel2*math.cos(Alrel2_rad)
    Vtrel2 = Vrel2*math.sin(Alrel2_rad)


    # Predicted effy
    eta_now = datum[0].y/100.

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

    rpm = Omega*60./(2*math.pi)

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
    
    
    

def create_passage(PR:float=2.4, inlet_flow_coefficient:float=0.7, 
                       M_rel:float=0.6, inlet_hub_tip_ratio:float=0.5,
                       deHaller:float=1, outlet_yaw:float=-64, 
                       blade_circulation:float=0.6, tip_clearance:float=0.01,
                       inlet_P0:float=1, inlet_T0:float=300, mdot:float=5):
        """_summary_

        Args:
            PR (float, optional): _description_. Defaults to 2.4.
            inlet_flow_coefficient (float, optional): _description_. Defaults to 0.7.
            M_rel (float, optional): _description_. Defaults to 0.6.
            inlet_hub_tip_ratio (float, optional): _description_. Defaults to 0.5.
            deHaller (float, optional): _description_. Defaults to 1.
            outlet_yaw (float, optional): _description_. Defaults to -64.
            blade_circulation (float, optional): _description_. Defaults to 0.6.
            tip_clearance (float, optional): _description_. Defaults to 0.01.
            inlet_P0 (float, optional): _description_. Defaults to 1.
            inlet_T0 (float, optional): _description_. Defaults to 300.
            mdot (float, optional): _description_. Defaults to 5.
        """
        gam = 1.4
        cp = 1005 # J/(Kg K)
        gae = gam/(gam-1)