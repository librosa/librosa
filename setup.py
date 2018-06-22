from setuptools import setup, find_packages

import imp

version = imp.load_source('librosa.version', 'librosa/version.py')

with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(
    name='librosa',
    version=version.version,
    description='Python module for audio and music processing',
    author='Brian McFee',
    author_email='brian.mcfee@nyu.edu',
    url='http://github.com/librosa/librosa',
    download_url='http://github.com/librosa/librosa/releases',
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
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='audio music sound',
    license='ISC',
    install_requires=[
        'audioread >= 2.0.0',
        'numpy >= 1.8.0',
        'scipy >= 0.14.0',
        'scikit-learn >= 0.14.0, != 0.19.0',
        'joblib >= 0.12',
        'decorator >= 3.0.0',
        'six >= 1.3',
        'resampy >= 0.2.0',
        'numba >= 0.38.0',
    ],
    extras_require={
        'docs': ['numpydoc', 'sphinx!=1.3.1', 'sphinx_rtd_theme',
                 'matplotlib >= 2.0.0',
                 'sphinxcontrib-versioning >= 2.2.1',
                 'sphinx-gallery'],
        'tests': ['matplotlib >= 2.1'],
        'display': ['matplotlib >= 1.5'],
    }
)
