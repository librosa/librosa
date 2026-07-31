There are two ways to cite librosa in scholarly work, depending on whether you are citing it for its use in your work
or for its design and motivation.

We additionally encourage you to cite the original publications describing the methods implemented in librosa, in addition to citing librosa itself.  References can often be found within function documentation.

.. tab-set::
    .. tab-item:: Cite for use

        If you are citing for use in your own work, please cite the version you used by retrieving the appropriate DOI and citation information from Zenodo |zenodo_badge|.  This ensures that all contributing authors are credited for the work you are building on.

        .. |zenodo_badge| image:: https://zenodo.org/badge/6309729.svg
            :target: https://zenodo.org/badge/latestdoi/6309729
            :alt: DOI badge

        You can also find the DOI for your currently installed
        version by running the following command in Python:

        .. code-block:: python

            print(librosa.cite())
        
        And if you want a BibTeX entry for your paper, you can run:

        .. code-block:: python

            print(librosa.cite(bib=True))


    .. tab-item:: Cite for design

        If you wish to cite librosa for its design, motivation etc., please cite the `librosa paper <https://doi.org/10.25080/Majora-7b98e3ed-003>`_ published at `SciPy 2015 <https://scipy2015.scipy.org>`_. [#]_

        .. [#] McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto.
            "librosa: Audio and music signal analysis in python."
            In Proceedings of the 14th python in science conference, pp. 18-25. 2015.


As a rule of thumb:

    - Use a versioned librosa citation to document the software implementation used in your experiments.
    - Cite the original methodological papers for the algorithms that are central to your work.
    - Cite the SciPy paper when discussing librosa itself, its design, or its contribution as a software library.
