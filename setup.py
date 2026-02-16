"""
Richter — GPU-Accelerated Seismic Wave Propagation Library
Setup script for pip installation.
"""
from setuptools import setup, find_packages

setup(
    name="richter",
    version="0.1.0",
    description="High-performance 3D acoustic wave propagation on GPUs",
    author="Nate Willis",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib"],
        "segy": ["segyio"],
    },
)
