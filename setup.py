from setuptools import setup, find_packages

setup(
    name='market_env',
    version='0.1.0',
    description='Market Simulation Environment for RL Agents',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
        'pandas',
        'joblib',
    ],
    python_requires='>=3.7',
)
