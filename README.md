# Introduction
PyTurbo is a Full 2D to 3D Turbomachinery blade and passage generation tool. Designs are generated and blade profiles are exported to json format.

## Installation 
Installation from pip
> `pip install pyturbo-aero`

Installation from source
> `python setup.py install`

Importing the python package after installing
> `import pyturbo_aero as pta`

[Link to documentation](https://nasa.github.io/pyturbo-aero)

# Tutorial
- [2D Airfoil Design](https://colab.research.google.com/github/nasa/pyturbo-aero/blob/main/tutorials/2D_DesignTutorial.ipynb)
- [3D Airfoil Design](https://colab.research.google.com/github/nasa/pyturbo-aero/blob/main/tutorials/3D_DesignTutorial.ipynb)
- [Stage Tutorial](https://colab.research.google.com/github/nasa/pyturbo-aero/blob/main/tutorials/3D_StageTutorial.ipynb)
- [Wavy Tutorial](https://colab.research.google.com/github/nasa/pyturbo-aero/blob/main/tutorials/3D_Wavy_Example.ipynb)
- [Radial Machines](https://colab.research.google.com/github/nasa/pyturbo-aero/blob/main/tutorials/Radial_Machines.ipynb)
  -- Just an note, I am making lots of update to centrif.py so if you notice the tutorials breaking, please submit an issue and I'll fix it.
- [Rotor37](https://colab.research.google.com/github/nasa/pyturbo-aero/blob/main/tutorials/rotor37/rotor37.ipynb) Tutorial on how to import rotor 37 txt files.
  - See Example from turbo-design on how to import rotor 37 iges files along with hub and shroud curves [Tutorial](https://colab.research.google.com/github/nasa/turbo-design/blob/main/examples/EEE/eee_hpt.ipynb)
- [Bi-Directional Turbine Design](https://colab.research.google.com/github/nasa/pyturbo-aero/blob/main/tutorials/3D_SymmetricTutorial.ipynb) This might help with power extraction for pulsating applications such as a shock tube
# License
[NASA Open Source Agreement](https://opensource.org/licenses/NASA-1.3)


# Disclaimer
This tool should only be used for design exploration. The final component design should always be done with CAD. This tool is not to be used as a final design tool. 
