from setuptools import setup, find_packages

import imp

version = imp.load_source('librosa.version', 'librosa/version.py')

setup(
    name='librosa',
    version=version.version,
    description='Python module for audio and music processing',
    author='Brian McFee',
    author_email='brian.mcfee@nyu.edu',
    url='http://github.com/bmcfee/librosa',
    download_url='http://github.com/bmcfee/librosa/releases',
    packages=find_packages(),
    package_data={'': ['example_data/*']},
    long_description="""A python module for audio and music processing.""",
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
    ],
    keywords='audio music sound',
    license='ISC',
    install_requires=[
        'audioread',
        'numpy >= 1.8.0',
        'scipy >= 0.13.0',
        'scikit-learn >= 0.14.0',
        'matplotlib',
        'joblib',
        'decorator',
        'six',
    ],
    extras_require={
        'resample': 'scikits.samplerate>=0.3',
        'docs': ['numpydoc', 'seaborn', 'sphinx_rtd_theme']
    }
)
