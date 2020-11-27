"""
©Copyright 2020 University of Florida Research Foundation, Inc. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
from setuptools import setup, find_packages

setup(
    name='kernet',
    version="0.2.0",
    author="Shiyu Duan",
    description=("Connectionist models powered by kernel machines."),
    license="CC BY-NC-SA 4.0",
    url="https://github.com/michaelshiyu/kernet",
    long_description=open(os.path.join(os.path.dirname(
        __file__), "README.md"), encoding='utf-8').read(),
    packages=find_packages(),
)
