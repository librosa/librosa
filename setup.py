from setuptools import setup, find_packages
import sys

from importlib.machinery import SourceFileLoader

version = SourceFileLoader('librosa.version',
                           'librosa/version.py').load_module()

with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(
    name='librosa',
    version=version.version,
    description='Python module for audio and music processing',
    author='Brian McFee',
    author_email='brian.mcfee@nyu.edu',
    url='https://librosa.org',
    download_url='https://github.com/librosa/librosa/releases',
    packages=find_packages(),
    package_data={'': ['example_data/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Framework :: Matplotlib",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords='audio music sound',
    license='ISC',
    install_requires=[
        'audioread >= 2.0.0',
        'numpy >= 1.15.0',
        'packaging >= 18',
        'scipy >= 1.0.0',
        'scikit-learn >= 0.14.0, != 0.19.0',
        'joblib >= 0.14',
        'decorator >= 3.0.0',
        'resampy >= 0.2.2',
        'numba >= 0.43.0',
        'soundfile >= 0.9.0',
        'pooch >= 1.0'
    ],
    python_requires='>=3.6',
    extras_require={
        'docs': ['numpydoc', 'sphinx!=1.3.1', 'sphinx_rtd_theme==0.5.*',
                 'matplotlib >= 2.0.0',
                 'sphinx-multiversion==0.2.3',
                 'sphinx-gallery>=0.7',
                 'sphinxcontrib-svg2pdfconverter',
                 'presets'],
        'tests': ['matplotlib >= 2.1',
                  'pytest-mpl',
                  'pytest-cov',
                  'pytest',
                  'contextlib2',
                  'samplerate'],
        'display': ['matplotlib >= 1.5'],
    }
)
