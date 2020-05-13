import os
from setuptools import setup, find_packages

setup(
    name='kernet',
    version="0.1.1",
    author="Shiyu Duan",
    description=("Connectionist models powered by kernel machines."),
    license="MIT",
    url="https://github.com/michaelshiyu/kerNET",
    long_description=open(os.path.join(os.path.dirname(__file__), "README.md"), encoding='utf-8').read(),
    packages=find_packages(),
)