Example files
^^^^^^^^^^^^^

*librosa* includes a small selection of example recordings which are primarily used
to demonstrate different functions of the library.
Beginning with version 0.8, these examples are automatically retrieved from a remote
server upon request.
Example recordings are cached locally after the first request, so each file should
only be downloaded once.


Downloading files directly
--------------------------

If you want to ensure that example data are always present on your computer, you can
clone the data repository directly::

    $ git clone https://github.com/librosa/data.git /path/to/librosa-data

To ensure that *librosa* can find the example files, you can set the
`LIBROSA_DATA_DIR` environment variable prior to importing *librosa*::

    $ export LIBROSA_DATA_DIR=/path/to/librosa-data/audio
    $ python my_librosa_script.py

or, directly in Python::

    >>> import os
    >>> os.environ['LIBROSA_DATA_DIR'] = '/path/to/librosa-data/audio'
    >>> import librosa
    
This will bypass any remote network access and use the local copies of the data
files directly.


Description of examples
-----------------------

The function `librosa.util.list_examples()` provides a brief description of each
track, and `librosa.util.example_info()` will provide some metadata and licensing
information for a given track.

The following table describes in more detail what the recordings are, and how they
are mainly used in the documentation.
`Key` indicates the descriptor used to identify the track when calling `librosa.example()`.

==========  =========================================== ============================================================================
Key         Full name                                   Description
==========  =========================================== ============================================================================
brahms      Brahms - Hungarian Dance #5                 A short performance of this piece, with soft note onsets and variable tempo.
choice      Admiral Bob - Choice                        A short drum and bass loop, good for demonstrating decomposition methods.
fishin      Karissa Hobbs - Let's Go Fishin'            A folk/pop song with verse/chorus/verse structure and vocals.
nutcracker  Tchaikovsky - Dance of the Sugar Plum Fairy Orchestral piece included to demonstrate tempo and harmony features.
trumpet     Mihai Sorohan - Trumpet loop                Monophonic trumpet recording, good for demonstrating pitch features.
vibeace     Kevin Macleod - Vibe Ace                    A vibraphone, piano, and bass combo. Previously the only included example.
==========  =========================================== ============================================================================
