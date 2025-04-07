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
- [Radial Machines](https://colab.research.google.com/github/nasa/pyturbo-aero/blob/main/tutorials/Radial_Machines.ipynb)
  -- Just an note, I am making lots of update to centrif.py so if you notice the tutorials breaking, please submit an issue and I'll fix it.
- [Rotor37](https://colab.research.google.com/github/nasa/pyturbo-aero/blob/main/tutorials/rotor37/rotor37.ipynb)

# License
[NASA Open Source Agreement](https://opensource.org/licenses/NASA-1.3)


# Disclaimer
This tool should only be used for design exploration. The final component design should always be done with CAD. This tool is not to be used as a final design tool. 

# Complaints about NASA IT
If GitHub Pages doesn’t deploy properly, the issue is likely related to NASA IT support. I’ve repeatedly filed internal tickets to resolve this problem, but they are often marked as resolved without any communication or follow-up. When a ticket is filed, an email is sent with a long ticket number, but no description of the issue. Later, IT may contact me referencing just the ticket number (e.g., “11192345”), and I’m expected to remember what the issue was — which is not practical (Issue #1).

Another challenge is that there is no accessible history of tickets I’ve submitted — unlike, for example, how Amazon shows your past orders. This lack of transparency makes it difficult to track progress or follow up (Issue #2).

Unfortunately, NASA IT is currently dysfunctional. There’s no unified knowledge base, and many systems seem to be developed by different external vendors with little to no integration or coordination. There appears to be minimal testing to ensure that these systems work together. As a result, the burden often falls on individual researchers to identify and troubleshoot systemic issues. Despite over a year of digital transformation meetings, there is still no clear vision for how tools should interconnect or how to better support researchers in working efficiently — both internally and with the public.

I sincerely apologize to the users of this tool and any NASA software I support. I want to provide a better experience. But please understand, I don’t have a team — it’s just me, Paht, maintaining the code and fixing the bugs.
