*******
librosa
*******
`librosa` is a python package for music and audio analysis.  It provides the building
blocks necessary to create music information retrieval systems.

For a quick introduction to using librosa, please refer to the :doc:`tutorial`.
For a more advanced introduction which describes the package design principles, please refer to the
`librosa paper <http://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf>`_ at
`SciPy 2015 <http://scipy2015.scipy.org>`_.

Citing librosa
==============

If you want to cite librosa in a scholarly work, there are two ways to do it.

- If you are using the library for your work, for the sake of reproducibility, please cite the version you used by retrieving the appropriate DOI and citation information from Zenodo:

.. image:: https://zenodo.org/badge/6309729.svg
   :target: https://zenodo.org/badge/latestdoi/6309729

- If you wish to cite librosa for its design, motivation etc., please cite the paper
  published at SciPy 2015. [#]_

.. [#] McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto.
    "librosa: Audio and music signal analysis in python."
    In Proceedings of the 14th python in science conference, pp. 18-25. 2015.


.. toctree::
    :caption: Getting started
    :maxdepth: 1

    install
    tutorial
    troubleshooting


.. toctree::
    :caption: API documentation
    :maxdepth: 1

    core
    display
    feature
    onset
    beat
    decompose
    effects
    segment
    sequence
    util


.. toctree::
    :caption: Advanced topics
    :maxdepth: 2

    filters
    cache
    ioformats
    advanced
    recordings

.. toctree::
    :caption: Reference
    :maxdepth: 1

    changelog
    genindex
    glossary
