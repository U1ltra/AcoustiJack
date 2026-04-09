from setuptools import setup, find_packages

setup(
    name="RIGOL_scpi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyvisa",
    ],
    python_requires=">=3.7",
)
