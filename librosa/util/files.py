#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for dealing with files"""

import os
import glob
import pkg_resources

import six

EXAMPLE_AUDIO = 'example_data/Kevin_MacLeod_-_Vibe_Ace.ogg'


__all__ = ['example_audio_file', 'find_files']


def example_audio_file():
    '''Get the path to an included audio example file.

    Examples
    --------
    >>> # Load the waveform from the example track
    >>> y, sr = librosa.load(librosa.util.example_audio_file())

    Returns
    -------
    filename : str
        Path to the audio example file included with librosa

    .. raw:: html

      <div xmlns:cc="http://creativecommons.org/ns#"
        xmlns:dct="http://purl.org/dc/terms/"
        about="http://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace_1278">
        <span property="dct:title">Vibe Ace</span>
        (<a rel="cc:attributionURL" property="cc:attributionName"
            href="http://freemusicarchive.org/music/Kevin_MacLeod/"
            >Kevin MacLeod</a>)
        / <a rel="license"
             href="http://creativecommons.org/licenses/by/3.0/"
             >CC BY 3.0</a>
      </div>
    '''

    return pkg_resources.resource_filename(__name__, EXAMPLE_AUDIO)


def find_files(directory, ext=None, recurse=True, case_sensitive=False,
               limit=None, offset=0):
    '''Get a sorted list of (audio) files in a directory or directory sub-tree.

    Examples
    --------
    >>> # Get all audio files in a directory sub-tree
    >>> files = librosa.util.find_files('~/Music')

    >>> # Look only within a specific directory, not the sub-tree
    >>> files = librosa.util.find_files('~/Music', recurse=False)

    >>> # Only look for mp3 files
    >>> files = librosa.util.find_files('~/Music', ext='mp3')

    >>> # Or just mp3 and ogg
    >>> files = librosa.util.find_files('~/Music', ext=['mp3', 'ogg'])

    >>> # Only get the first 10 files
    >>> files = librosa.util.find_files('~/Music', limit=10)

    >>> # Or last 10 files
    >>> files = librosa.util.find_files('~/Music', offset=-10)

    Parameters
    ----------
    directory : str
        Path to look for files

    ext : str or list of str
        A file extension or list of file extensions to include in the search.

        Default: `['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']`

    recurse : boolean
        If `True`, then all subfolders of `directory` will be searched.

        Otherwise, only `directory` will be searched.

    case_sensitive : boolean
        If `False`, files matching upper-case version of
        extensions will be included.

    limit : int > 0 or None
        Return at most `limit` files. If `None`, all files are returned.

    offset : int
        Return files starting at `offset` within the list.

        Use negative values to offset from the end of the list.

    Returns
    -------
    files : list of str
        The list of audio files.
    '''

    if ext is None:
        ext = ['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']

    elif isinstance(ext, six.string_types):
        ext = [ext]

    # Cast into a set
    ext = set(ext)

    # Generate upper-case versions
    if not case_sensitive:
        # Force to lower-case
        ext = set([e.lower() for e in ext])
        # Add in upper-case versions
        ext |= set([e.upper() for e in ext])

    files = []

    if recurse:
        for walk in os.walk(directory):
            files.extend(__get_files(walk[0], ext))
    else:
        files = __get_files(directory, ext)

    files.sort()
    files = files[offset:]
    if limit is not None:
        files = files[:limit]

    return files


def __get_files(dir_name, extensions):
    '''Helper function to get files in a single directory'''

    # Expand out the directory
    dir_name = os.path.abspath(os.path.expanduser(dir_name))

    myfiles = []

    for sub_ext in extensions:
        globstr = os.path.join(dir_name, '*' + os.path.extsep + sub_ext)
        myfiles.extend(glob.glob(globstr))

    return myfiles
