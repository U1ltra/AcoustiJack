from setuptools import setup, find_packages

setup(
    name="AD9833_spi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "spidev",
    ],
    python_requires=">=3.7",
)