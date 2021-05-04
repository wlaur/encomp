from setuptools import setup

from encomp import __version__

setup(
    name='encomp',
    version=__version__,
    description='General toolkit for engineering computations',
    license='MIT',
    author='William Laurén',
    author_email='lauren.william.a@gmail.com',
    packages=[
        'encomp',
        'encomp.tests',
    ],
    python_requires='~=3.8',
    include_package_data=True  # package data is specified in MANIFEST.in
)
