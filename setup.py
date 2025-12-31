from setuptools import setup, find_packages

setup(
    name='pyedtools-control',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'control>=0.9.0',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    python_requires='>=3.8',
    author='Akhil Datla',
    description='A student-friendly Python library for learning classical control systems',
    long_description=open('README.md').read() if __import__('os').path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    license='MIT',
    keywords=['control systems', 'education', 'PID', 'transfer function', 'bode', 'root locus'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
    ],
)
