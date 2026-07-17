*******
librosa
*******
`librosa` is a Python library for audio and music signal processing. It provides the foundational algorithms and tools required for building music information retrieval (MIR) systems.

Installing librosa
==================

.. include:: install.rst


Citing librosa
==============

If you want to cite librosa in a scholarly work, there are two ways to do it.

- If you are using the library for your work, for the sake of reproducibility, please cite the version you used by retrieving the appropriate DOI and citation information from Zenodo |zenodo_badge|.

.. |zenodo_badge| image:: https://zenodo.org/badge/6309729.svg
    :target: https://zenodo.org/badge/latestdoi/6309729
    :alt: DOI badge

You can also find the DOI for your currently installed
version by running the following command in Python:

.. code-block:: python

    import librosa
    print(librosa.cite())


- If you wish to cite librosa for its design, motivation etc., please cite the `librosa paper <https://doi.org/10.25080/Majora-7b98e3ed-003>`_ published at `SciPy 2015 <https://scipy2015.scipy.org>`_. [#]_

.. [#] McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto.
    "librosa: Audio and music signal analysis in python."
    In Proceedings of the 14th python in science conference, pp. 18-25. 2015.


.. toctree::
    :maxdepth: 2
    :hidden:

    tutorial
    api/index
    advanced/index
    changelog
    contributing
    reference/index
