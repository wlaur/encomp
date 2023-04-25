Setup
=====

``encomp`` requires Python 3.9 or higher.


Installing with ``pip``
-----------------------

To install ``encomp`` with ``pip``:

.. code-block:: bash

    pip install encomp

This will install ``encomp`` along with its dependencies into the active Python environment.

.. warning::

    The *CoolProp* package might cause issues when installing with ``pip``.
    In this case, use ``conda`` to install a prebuilt version from ``conda-forge``:

    .. code-block:: bash

        conda install conda-forge::coolprop


Development environment
-----------------------

A development environment can be set up using ``conda``.
First make sure that ``conda`` is installed (from Anaconda or Miniconda).

Clone the repository:

.. code-block:: bash

    git clone --depth 1 https://github.com/wlaur/encomp
    cd encomp


Create a new ``conda`` environment named ``encomp-env`` with all dependencies (including development dependencies):

.. code-block:: bash

    conda env create -f environment.yml

Activate the environment and install ``encomp`` from local source files:

.. code-block:: bash

    conda activate encomp-env
    pip install .


Removing ``conda`` environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To completely remove the ``conda`` environment for ``encomp``:

.. code-block:: bash

    conda remove -y --name encomp-env --all


Building documentation
~~~~~~~~~~~~~~~~~~~~~~

The Sphinx documentation is built from source with the following commands:

.. code-block:: bash

    sphinx-apidoc -f -o docs/source encomp encomp/tests
    call docs/make clean
    call docs/make html

.. tip::
    The script ``utils.py`` contains commands for some common tasks.


Testing
-------

The tests are run with ``pytest``.
Some configuration options are defined in ``pytest.ini``.


.. code-block:: bash

    # run pytest from the root of the repository
    pytest .

To disable the ``mypy`` tests, add the flag ``-p no:mypy-testing``:

.. code-block:: bash


    # run from the root of the repository
    pytest . -p no:mypy-testing

.. todo::

    The ``pytest-mypy-testing`` plugin does not seem to work on Windows.

Use ``coverage`` to generate a coverage report (ignore the ``mypy`` tests for this):

.. code-block:: bash

    coverage run -m pytest . -p no:mypy-testing
    coverage html

The test report will be generated in the ``htmlcov`` subdirectory.
This directory is not included in version control.

.. todo::

    Coverage reports don't work in WSL, the file paths seem to get mixed up between Windows and WSL.
