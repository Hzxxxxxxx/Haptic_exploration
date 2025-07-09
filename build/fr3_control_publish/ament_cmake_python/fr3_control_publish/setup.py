from setuptools import find_packages
from setuptools import setup

setup(
    name='fr3_control_publish',
    version='0.0.0',
    packages=find_packages(
        include=('fr3_control_publish', 'fr3_control_publish.*')),
)
