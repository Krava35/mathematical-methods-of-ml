from setuptools import setup, find_packages

setup(
    name="lab2",  # Name of your package/project
    version="0.1",  # Initial version
    description="A Python library for BFGS and L-BFGS",  # Short description
    author="Kravchenko Daniil and Kurchaviy Vitaly",  # Your name
    packages=find_packages(),  # Automatically find packages in your project
    python_requires='>=3.6',  # Minimum Python version required
)