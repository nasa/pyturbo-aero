.. GlennOPT documentation master file, created by
   sphinx-quickstart on Mon Apr  5 15:19:34 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GlennOPT Documentation
==================================
GlennOPT is a general purpose multi-objective optimizer for computer simulations locally or on the super computer. One of the biggest problems when simulating many designs on a super computer are the unexpected problems that occur such as unexpected server crashes, out of storage, updates that stop the job from executing. 

GlennOPT is designed with this in mind. When an optimization crashes, you can restart GlennOPT and it will scan the execution folder either the DOE or POP folders and any simulation that did not yield an `output.txt` file will be restarted from it's local evaluation script. 

The number of files that GlennOPT produces is quite significant. It saves a copy of every single execution of the objective function. This is useful for generating data for machine learning. While many of these evaluations may not yield an optimized result, some may even fail. These extra data is useful for debugging/understanding the trends of your design space.

.. note::
   GlennOPT does not solve CFD. It cannot provide direct gradients that relate your objective to the design parameters. The best method for doing this would be to use one of the surrogate models such as `NSOPT` and specify a minimization method from scipy that an estimate gradients 
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html 

Background
==============================

Individual 
-----------------------------
In this documentation and the code you will hear a lot about `Individual`. This represents your design parameters for a single evaluation. So lets take a simple function :math:`f(x) = x_1 + x_2 + (x_3)^2` the :math:`x` in :math:`f(x)` is a vector containing :math:`x_1`, :math:`x_2`, :math:`x_3`. This vector of :math:`x` is the evaluation parameters of an individual. An individual in GlennOPT is a class containing the evaluation parameters, performance parameters, objective values, and constraints.

Say your objectives include these two functions :math:`f_1(x) and :math:f_2(x)` where x is a vector shared between these functions. The individual will contain f_1 and f_2. If you have other performance parameter such as Pressure(x) and Speed(x) these can also be tracked within an individual. 

When evaluating using GlennOPT, you specify how many individuals per population and how many populations to run for. Think of it as keeping track of people - you have a population of the group and the number of generations which you are recording data for. The individual represents a single person and all the properties and objectives. 

Now that you have a rough background, please check out the tutorials and documentation on single and multi-objective.

Optimization Folder Structure
-----------------------------------------

It is recommended to keep the following folder structure when performing the optimization. The calculation folder will be created automatically by glennopt. However you need to put

::

   Calculation (Created automatically)
      -DOE 
         -IND000
            -Evaluation.py (copied from Data folder)
            -input.dat (contains your evaluation parameters that goes into your optimization function)
      -POP000
      ...
      -POP010
   Data
      -Evaluation.py (Called by the optimizer, this can be your optimization function)
   optimization_setup.py (starts the optimization)


Note about NAS (NASA Advance Super Computer) or any Torque/SLURM queuing system 
################################################################################
When executing on NAS it was found to be best to launch simulations directly from GlennOPT and have GlennOPT wait for the output.txt results. Think of doing a `qsub` to launch glennopt and then having glennopt do `qsub` to launch other executions. This is better when debugging failed simulations. 


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/optimizers
   

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/base
   modules/sode
   modules/nsga3
   modules/nsga3_ml
   modules/nsopt
   modules/helpers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




