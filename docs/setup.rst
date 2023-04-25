Setup
=====

``encomp`` requires Python 3.10 or higher.


Installing with ``pip``
-----------------------

To install ``encomp`` with ``pip``:

.. code-block:: bash

    pip install encomp

To also install optional dependencies:

.. code-block:: bash

    pip install encomp[optional]


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
    poetry install --extras optional


Removing ``conda`` environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To completely remove the ``conda`` environment for ``encomp``:

.. code-block:: bash

    conda remove -y --name encomp-env --all


Building documentation
~~~~~~~~~~~~~~~~~~~~~~

The Sphinx documentation is built from source with the following commands:

.. code-block:: bash

    python scripts/utils.py docs


Testing
-------

The tests are run with ``pytest``.
Some configuration options are defined in ``pytest.ini``.


.. code-block:: bash

    # run pytest from the root of the repository
    pytest .

Use ``coverage`` to generate a coverage report (ignore the ``mypy`` tests for this):

.. code-block:: bash

    coverage run -m pytest . -p no:mypy-testing
    coverage html

The test report will be generated in the ``htmlcov`` subdirectory.
This directory is not included in version control.
