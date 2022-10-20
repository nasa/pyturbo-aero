Installation
===============

.. note::
    We do not recommend installation as root user on your system's default installation. It's best to either use miniconda or anaconda or some virtual environment for python 

    Please setup an `Anaconda/Miniconda <https://conda.io/docs/user-guide/install/index.html/>`_ environment or create a `Docker image <https://www.docker.com/>`_.

    Anaconda Installation can be found herehttps://www.anaconda.com/products/individual

Please follow the steps below for a successful installation.

Installation via Pip
-------------------------

#. Install the relevant packages:

    .. code-block:: none

        pip install glennopt

Installation via Source
-------------------------

Clone the repository and you will need the poetry package manager to build and install the project. `Poetry Installation <https://python-poetry.org/docs/#installation>`

Commands for building and installing the library

    .. code-block:: none

        poetry build
        poetry install 

        