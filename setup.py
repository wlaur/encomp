from setuptools import setup

from encomp import __version__

# load requirements.txt (base install) dependencies
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()


# include readme
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='encomp',
    version=__version__,
    description='General-purpose library for engineering computations',
    url='https://github.com/wlaur/encomp',
    license='MIT',
    author='William Laurén',
    author_email='lauren.william.a@gmail.com',
    packages=[
        'encomp',
        'encomp.tests',
    ],
    python_requires='~=3.9',
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True  # package data is specified in MANIFEST.in
)
