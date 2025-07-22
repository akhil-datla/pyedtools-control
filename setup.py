from setuptools import setup, find_packages

setup(
    name='pyedtools-control',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'control>=0.9.0',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    author='PyEdTools Team',
    description='Educational control systems toolkit',
    license='MIT'
)
