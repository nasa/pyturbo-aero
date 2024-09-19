import matplotlib.pyplot as plt
import os, glob
import pandas as pd 


folder = 'tutorials/centrif/NASA-HECC-Data-Archive'
coord_path = 'HECC_Vaned_Configuration/Coordinates/HECC_As-Manufactured_Cold_Coordinates_(Cold_With_Fillets)'
plot_folder = os.path.join(folder,'hecc_plots')

main_blade = list(glob.glob(
    os.path.join(folder,coord_path,'HECCcoordinates_impellerMainBladeColdWithFillets_section*.csv')))

splitter_blade = list(glob.glob(
    os.path.join(folder,coord_path,'HECCcoordinates_impellerSplitterBladeColdWithFillets_section*.csv'))) 
main_blade.sort()
splitter_blade.sort()

os.makedirs(plot_folder,exist_ok=True)
i=1
plt.figure(num=2,clear=True)
for main,splitter in zip(main_blade,splitter_blade):
    m = pd.read_csv(main, sep=',', header='infer')
    s = pd.read_csv(splitter, sep=',', header='infer')
    
    plt.figure(num=1,clear=True)    # 2D Plots
    plt.plot(m.values[:,0],m.values[:,1],label='main blade')
    plt.plot(s.values[:,0],s.values[:,1],label='splitter blade')
    plt.legend()
    plt.title(f"Main and Splitter {i:02d}")
    plt.xlabel('x-axial')
    plt.ylabel('rtheta')
    plt.savefig(os.path.join(plot_folder,f'main_w_splitter_sections_{i:02d}.jpg'))
    
    plt.figure(num=2,clear=False)    # 2D Plots
    plt.plot(m.values[:,0],m.values[:,2],label='main blade')
    plt.plot(s.values[:,0],s.values[:,2],label='splitter blade')
    plt.legend()
    plt.title(f"Main and Splitter {i:02d}")
    plt.xlabel('x-axial')
    plt.ylabel('rtheta')
    plt.savefig(os.path.join(plot_folder,f'main_w_splitter_{i:02d}.jpg'))
print(os.path.exists(os.path.join(folder,coord_path)))