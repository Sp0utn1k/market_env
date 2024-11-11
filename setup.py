from setuptools import setup, find_packages
import os

def package_files(directory):
    """
    Recursively gather all files in a directory to include in the package.
    """
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

# Include all files in the 'configs' directory
extra_files = package_files('market_env/configs')

setup(
    name='market_env',
    version='0.1.0',
    description='Market Simulation Environment for RL Agents',
    author='Your Name',
    packages=find_packages(),
    package_data={'market_env': extra_files},
    include_package_data=True,
    install_requires=[
        'gymnasium',
        'numpy',
        'pandas',
        'joblib',
    ],
    python_requires='>=3.7',
)
