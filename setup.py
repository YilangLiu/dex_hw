from setuptools import find_packages
from distutils.core import setup

setup(
    name='vega_manipulation',
    version='0.1.0',
    author='Yilang Liu',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='yilang.liu@yale.edu',
    description='Genesis environments for Vega',
    install_requires=[
                      'torch',
                      'matplotlib',
                      'torchvision',
                      'numpy',
                      'tensorboard',
                      'xlsxwriter',
                      'pandas']
)
