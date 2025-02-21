'''
    Code translated from https://whittle.digital/2024/Radial_Compressor_Designer/ 
'''
import math
import json
from typing import Dict, Tuple 
import numpy as np 
from pyturbo.helper import arc
import matplotlib.pyplot as plt 
import os, pathlib
import requests

def import_json():
    """Import Whittle Lab Data

    Returns:
        Tuple containing:
        
            *indices* (Dict[str,int]): Dictionary containing indices in fitdata that match varible names
            *xl_dict* (Dict[str,int]): Dictionary containing lower values of the data that match variable names
            *dx_dict* (Dict[str,int]): Dictionary containing dx values of the data that match variable names
            *fit_data* (Dict[str,Any]): all the data
    """

    default_home = os.path.join(os.path.expanduser("~"), ".cache")
    os.environ['pyturbo-aero'] = os.path.join(default_home,'whittle_labs_radial_compressor')
    path = pathlib.Path(os.path.join(os.environ['pyturbo-aero'],"fit_eta_tt"+".json"))

    try:
        if not path.exists():
            os.makedirs(os.environ['pyturbo-aero'],exist_ok=True)
            url = "https://whittle.digital/2024/Radial_Compressor_Designer/fit_eta_tt.json"
            response = requests.get(url, stream=True)
            with open(path.absolute(), mode="wb") as file:
                for chunk in response.iter_content(chunk_size=10 * 1024):
                    file.write(chunk)
    except Exception as ex:  
        pass
    try:
        with open(path.absolute(),'r') as fp:            
            fit_data = json.load(fp)
            indices = dict() 

            xl = np.array(fit_data['xl'][0])
            xu = np.array(fit_data['xu'][0])
            dx = xu-xl
            dx[fit_data['vars'].index('PR_tt')]
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
    except Exception as ex:  
        pass
    indices = dict() 
    xl_dict = dict()
    dx_dict = dict()
    fit_data = None
    return indices,xl_dict,dx_dict,fit_data
    

def entropy(P:float,T:float,cp:float,Tref:float=300,Pref:float=1E5,Rgas:float=287.15) -> float:
    """Computes the entropy rise from Temperature and Pressure 

    Args:
        P (float): Pressure in Pascal
        T (float): Temperature in Kelvin
        cp (float): Coefficient of Pressure
        Tref (float, optional): Reference Temperature [Kelvin]. Defaults to 300.
        Pref (float, optional): Reference Pressure [Pa]. Defaults to 1E5.
        Rgas (float, optional): Ideal Gas constant [J/(Kg K)]. Defaults to 287.15.

    Returns:
        float: entropy difference delta_s
    """
    return cp * math.log(T/Tref) - Rgas * math.log(P/Pref)

def legval(k:int, x:float) ->float:
    """Legendre Polynomials

    Args:
        k (int): polynomial power 
        x (float): value

    Returns:
        float: evalulated legendre polynomial 
    """
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
        PRtt (float): Total Pressure Ratio 
        HTR1 (float): _description_
        Marel1 (float): Relative inlet mach number
        tau (float): _description_
        Alrel2 (float): _description_
        DH (float): deHaller number
        phi (float): inlet flow coefficient
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
    pval = np.zeros((len(fit_data['vars']),4))
    pval[indices['PRtt'],:] = PRtt_pval
    pval[indices['HTR1'],:] = HTR1_pval
    pval[indices['Marel1'],:] = Marel1_pval     # M1 
    pval[indices['tau'],:] = tau_pval           # Tip Clearance 
    pval[indices['Alrel2'],:] = Alrel2_pval     # Outlet Relative Yaw
    pval[indices['DH'],:] = DH_pval             # DeHaller Number 
    pval[indices['phi'],:] = phi1_pval       # Inlet Flow coefficient
    pval[indices['Cgamma'],:] = Cgamma_pval     # blade circulation
    
    # Efficiency
    for j in range(J):
        deta = 1.
        for i in range(I):
            deta = deta * pval[i,k[j][i]]
        eta = eta + c[j]*deta
    return eta
    
def normalize(PRtt:float,HTR1:float,Marel1:float,tau:float,DH:float,phi1:float,Alrel2:float,Cgamma:float,xl:Dict[str,float],dx:Dict[str,float]):
    """Normalize the data given absolute data

    Args:
        PRtt (float): Total Pressure Ratio
        HTR1 (float): Inlet hub to tip ratio
        Marel1 (float): Inlet relative mach number
        tau (float): Tip Clearance as a percent
        DH (float): deHaller Number
        phi1 (float): Inlet flow coefficient
        Alrel2 (float): Blade exit angle [degrees]
        Cgamma (float): Recirculation 
        xl (Dict[str,float]): Dictionary containing lower values of the data that match variable names
        dx (Dict[str,float]): Dictionary containing dx values of the data that match variable names

    Returns:
        Normalized Data: PRtt, HTR1, Marel1, tau, DH, phi1, Alrel2, Cgamma
    """
    PRtt = 2. * (PRtt - xl['PRtt'])/dx['PRtt'] - 1.
    HTR1 = 2. * (HTR1 - xl['HTR1'])/dx['HTR1'] - 1.
    Marel1 = 2. * (Marel1 - xl['Marel1'])/dx['Marel1'] - 1.
    tau = 2. * (tau - xl['tau'])/dx['tau'] - 1.
    DH = 2. * (DH - xl['DH'])/dx['DH'] - 1.
    phi1 = 2. * (phi1 - xl['phi'])/dx['phi'] - 1.
    Alrel2 = 2. * (Alrel2 - xl['Alrel2'])/dx['Alrel2'] - 1.
    Cgamma = 2. * (Cgamma - xl['Cgamma'])/dx['Cgamma'] - 1.

    return PRtt, HTR1,Marel1,tau,DH,phi1,Alrel2,Cgamma

# Global variables 
indices,xl_dict,dx_dict,fit_data = import_json()

def create_passage(PR:float=2.4, phi1:float=0.7, 
                       M1_rel:float=0.6, HTR1:float=0.5,
                       deHaller:float=1, outlet_yaw:float=-64, 
                       blade_circulation:float=0.6, tip_clearance:float=0.01,
                       P01:float=1, T01:float=300, mdot:float=5,gam:float=1.4,Rgas:float=287.15):
    """Create a passage for the centrif using whittle labs code 

    Args:
        PR (float, optional): Total Pressure Ratio. Defaults to 2.4.
        phi1 (float, optional): Inlet flow coefficient. Defaults to 0.7.
        M1_rel (float, optional): Inlet relative mach number. Defaults to 0.6.
        HTR1 (float, optional): Inlet hub to tip ratio. Defaults to 0.5.
        deHaller (float, optional): deHaller Number. Defaults to 1.
        outlet_yaw (float, optional): Blade exit angle [degrees]. Defaults to -64.
        blade recirculation (float, optional): How much recirculation flow in an centrif compressor. Defaults to 0.6.
        tip_clearance (float, optional): Tip Clearance as a percent. Defaults to 0.01.
        P01 (float, optional): Total Inlet Pressure [bar]. Defaults to 1.
        T01 (float, optional): Total Inlet Temperature [K]. Defaults to 300.
        mdot (float, optional): massflow rate [kg/s]. Defaults to 5.
        gam (float, optional): Ratio of Cp/Cv. Defaults to 1.4.
        Rgas (float, optional): Ideal Gas Constant [J/(Kg*K)]. Defaults to 287.15.
        
    Returns:
        Tuple containing:
            *hub* (npt.NDArray): hub curve nx2
            *shroud* (npt.NDArray): shroud curve nx2
            *V3* (float): Exit velocity (m/s)
            *T3* (float): Exit static temperature (Kelvin)
            *P3 (float): Exit static pressure (Pascal)
            *Ma3* (float): Exit absolute mach number
            *eta_ts*, (float): Total to static efficiency
            *eta_now* (float): Total to total efficiency
            *Power* (float): Power in watts
            *RPM* (float): Revolutions per min
            *Alrel1_deg* (float): Inlet flow angle
            *Alrel2_deg* (float): Exit flow angle
    """
    cp = gam*Rgas/(gam-1) # J/(Kg K)
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
    eta_now = calculate_efficiency(PR,HTR1,M1_rel,tip_clearance,outlet_yaw,deHaller,phi1,blade_circulation,fit_data,xl_dict,dx_dict,indices)

    # Diffuser outlet stagnation state
    P03 = (P01*1e5)* PR
    T03s = T01*(P03/P01/1e5)**(1/gae)
    T03 = T01 + (T03s-T01)/eta_now

    # Loss split to set rotor outlet state
    # Assumed constant
    zeta = 0.75
    s1 = entropy(P01*1e5, T01, cp, Tref, Pref, Rgas)
    s3 = entropy(P03, T03, cp, Tref, Pref, Rgas)
    s2 = s1 + zeta*(s3-s1)
    # print(f"s2-s1: {s2-s1}, s3-s1: {s3-s1}")
    
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
    hub = arc(0,1,Rhath,270,360)
    xsh, ysh = shroud.get_point(np.linspace(0,1,100))
    
    xhub, yhub = hub.get_point(np.linspace(0,1,100))
    
    xsh *= Xhatc/Rhatc
    xhub *= Xhath/Rhath
    # drawArc(0., 1.0, Rhatc, Xhatc/Rhatc)
    # drawArc(0., 1.0, Rhath, Xhath/Rhath)
    hub = np.vstack([xhub,yhub]).transpose()
    shroud = np.vstack([xsh,ysh]).transpose()
    return hub,shroud,V3,T3,P3,Ma3,eta_ts,eta_now,Power,RPM,Alrel1_deg,Alrel2_deg
        
if __name__ == "__main__":
    hub,shroud,V3,T3,P3,Ma3,eta_ts,eta_now,Power,RPM = create_passage(PR=2.0,phi1=0.6,M1_rel=0.6,HTR1=0.5,
                   deHaller=1,outlet_yaw=-60,blade_circulation=0.6,tip_clearance=0.01)
    plt.figure()    
    plt.plot(hub[:,0],hub[:,1],'k')
    plt.plot(shroud[:,0],shroud[:,1],'b')
    plt.show()
