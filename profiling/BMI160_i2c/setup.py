from setuptools import setup, find_packages

setup(
    name="BMI160_i2c",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "smbus2",
    ],
    python_requires=">=3.7",
)