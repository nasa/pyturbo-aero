'''
    Code translated from https://whittle.digital/2024/Radial_Compressor_Designer/ 
'''
import math
import json
from typing import Dict, Tuple 
import numpy as np 
from .arc import arc


def import_json():
    with open('./pyturbo/helper/fit_eta_tt.json','r') as fp:            
        fit_data = json.load(fp)
        
        xl = np.array(fit_data['xl'][0])
        xu = np.array(fit_data['xu'][0])
        dx = xu-xl
        dx[fit_data['vars'].index('PR_tt')]
        indices = dict() 
        indices['PRtt'] = fit_data['vars'].index('PR_tt')
        indices['HTR1'] = fit_data['vars'].index('htr1')
        indices['Marel1'] = fit_data['vars'].index('Marel1')
        indices['tau'] = fit_data['vars'].index('tip')
        indices['DH'] = fit_data['vars'].index('DHimp')
        indices['phi'] = fit_data['vars'].index('phi1')
        indices['Alrel2'] = fit_data['vars'].index('Alpharel2')
        indices['Cgamma'] = fit_data['vars'].index('Co1')
        
        xl_dict = dict()
        xl_dict['PRtt'] = xl[indices['PRtt']]
        xl_dict['HTR1'] = xl[indices['HTR1']] 
        xl_dict['Marel1'] = xl[indices['Marel1']]
        xl_dict['tau'] = xl[indices['tau']]
        xl_dict['DH'] = xl[indices['DH']]
        xl_dict['phi'] = xl[indices['phi']]
        xl_dict['Alrel2'] = xl[indices['Alrel2']]
        xl_dict['Cgamma'] = xl[indices['Cgamma']]

        dx_dict = dict()
        dx_dict['PRtt'] = dx[indices['PRtt']]
        dx_dict['HTR1'] = dx[indices['HTR1']]
        dx_dict['Marel1'] = dx[indices['Marel1']]
        dx_dict['tau'] = dx[indices['tau']]
        dx_dict['DH'] = dx[indices['DH']]
        dx_dict['phi'] = dx[indices['phi']]
        dx_dict['Alrel2'] = dx[indices['Alrel2']]
        dx_dict['Cgamma'] = dx[indices['Cgamma']]
        
        return indices,xl_dict,dx_dict,fit_data
    
    return None

def entropy(P:float,T:float,cp:float,Tref:float=300,Pref:float=1E5,Rgas:float=287.15):
    return cp * math.log(T/Tref) - Rgas * math.log(P/Pref)

def legval(k:int, x:float):
    if (k==0):
        return 1
    elif k==1:
        return x
    elif k==2:
        return (3 * x * x - 1) / 2
    elif k==3:
        return (5 * x * x * x - 3 * x) / 2

def calculate_efficiency(PRtt:float,HTR1:float,Marel1:float,tau:float,Alrel2:float,DH:float,phi1:float,Cgamma:float,
                         fit_data,
                         xl:Dict[str,float],
                         dx:Dict[str,float],
                         indices:Dict[str,int]):
    """Takes unnormalized values and computes the efficiency based on JSON Curve Fit

    Args:
        PRtt (float): _description_
        HTR1 (float): _description_
        Marel1 (float): _description_
        tau (float): _description_
        Alrel2 (float): _description_
        DH (float): _description_
        phi (float): _description_
        Cgamma (float): _description_
    """        
    c = fit_data['c']    
    I = len(fit_data['vars'])
    J = len(c)
    
    PRtt, HTR1, Marel1, tau, DH, phi1, Alrel2, Cgamma = normalize(PRtt,HTR1,Marel1,tau,DH,phi1,Alrel2,Cgamma,xl,dx)
    
    kk = list(range(4))
    # Add Legendre polynomials
    PRtt_pval = np.array([legval(k, PRtt) for k in kk])
    HTR1_pval = np.array([legval(k, HTR1) for k in kk])
    Marel1_pval = np.array([legval(k, Marel1) for k in kk])
    Alrel2_pval = np.array([legval(k, Alrel2) for k in kk])
    DH_pval = np.array([legval(k, DH) for k in kk])
    phi1_pval = np.array([legval(k, phi1) for k in kk])
    tau_pval = np.array([legval(k, tau) for k in kk])
    Cgamma_pval = np.array([legval(k, Cgamma) for k in kk])
    
    k = fit_data['k']
    
    eta = 0
    pval = np.zeros((len(fit_data.keys),4))
    pval[indices['PR_tt'],:] = PRtt_pval
    pval[indices['htr1'],:] = HTR1_pval
    pval[indices['Marel1'],:] = Marel1_pval     # M1 
    pval[indices['tau'],:] = tau_pval           # Tip Clearance 
    pval[indices['Alrel2'],:] = Alrel2_pval     # Outlet Relative Yaw
    pval[indices['DH'],:] = DH_pval             # DeHaller Number 
    pval[indices['Alrel2'],:] = phi1_pval       # Inlet Flow coefficient
    pval[indices['Cgamma'],:] = Cgamma_pval     # blade circulation
    
    # Efficiency
    for j in range(J):
        deta = 1.
        for i in range(I):
            deta = deta * pval[i,k[j][i]]
        eta = eta + c[j]*deta
    return eta
    
def normalize(PRtt:float,HTR1:float,Marel1:float,tau:float,DH:float,phi1:float,Alrel2:float,Cgamma:float,xl:Dict[str,float],dx:Dict[str,float]):   
    PRtt = 2. * (PRtt - xl['PRtt'])/dx['PRtt'] - 1.
    HTR1 = 2. * (HTR1 - xl['HTR1'])/dx['HTR1'] - 1.
    Marel1 = 2. * (Marel1 - xl['Marel1'])/dx['Marel1'] - 1.
    tau = 2. * (tau - xl['tau'])/dx['tau'] - 1.
    DH = 2. * (DH - xl['DH'])/dx['DH'] - 1.
    phi1 = 2. * (phi1 - xl['phi1'])/dx['phi1'] - 1.
    Alrel2 = 2. * (Alrel2 - xl['Alrel2'])/dx['Alrel2'] - 1.
    Cgamma = 2. * (Cgamma - xl['Cgamma'])/dx['Cgamma'] - 1.

    return PRtt, HTR1,Marel1,tau,DH,phi1,Alrel2,Cgamma


        

def create_passage(fit_data,xl:Dict[str,float],dx:Dict[str,float],indices:Dict[str,float],PR:float=2.4, phi1:float=0.7, 
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
        outlet_yaw (float, optional): Blade exit angle [degrees]. Defaults to -64.
        blade_circulation (float, optional): . Defaults to 0.6.
        tip_clearance (float, optional): _description_. Defaults to 0.01.
        P01 (float, optional): Total Inlet Pressure [bar]. Defaults to 1.
        T01 (float, optional): _description_. Defaults to 300.
        mdot (float, optional): _description_. Defaults to 5.
    """
    
               
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
    eta_now = calculate_efficiency(PR,HTR1,M1_rel,tip_clearance,Alrel2_deg,deHaller,phi1,blade_circulation,fit_data,xl,dx,indices)

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
        