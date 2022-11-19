from setuptools import setup

from encomp import __version__


def get_requirements(fname: str) -> list[str]:
    with open(fname, encoding='utf-8') as f:
        return [n for n in f.read().splitlines() if n.strip()]


install_requires = get_requirements('requirements.txt')
requirements_optional = get_requirements('requirements-optional.txt')
requirements_dev = get_requirements('requirements-dev.txt')


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
    package_data={'encomp': ['py.typed', '*.pyi']},
    python_requires='~=3.10',
    install_requires=install_requires,
    extras_require={
        'optional': requirements_optional,
        'dev': requirements_dev,
        'full': requirements_optional + requirements_dev
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True  # additional package data is specified in MANIFEST.in
)
