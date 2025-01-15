#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for dealing with files"""
from __future__ import annotations
from typing import List, Optional, Union, Any, Set

import os
import glob
import json
import msgpack
import contextlib
import sys

from importlib import resources

import pooch

from .exceptions import ParameterError
from ..version import version as librosa_version


__all__ = [
    "find_files",
    "example",
    "ex",
    "list_examples",
    "example_info",
]


# Instantiate the pooch
__data_path = os.environ.get("LIBROSA_DATA_DIR", pooch.os_cache("librosa"))
__GOODBOY = pooch.create(
    __data_path, base_url="https://librosa.org/data/audio/", registry=None
)


@contextlib.contextmanager
def _resource_file(package: str, resource: str):
    """Provide a context manager for accessing resources in a package.

    It acts as a shim to provide a consistent interface for accessing resources
    since the 3.9 series deprecated the "path" method in favor of the "files" method.
    """
    if sys.version_info < (3, 9):
        with resources.path(package, resource) as path:
            yield path

    else:
        with resources.as_file(resources.files(package).joinpath(resource)) as path:
            yield path


with _resource_file("librosa.util.example_data", "registry.txt") as reg:
    __GOODBOY.load_registry(str(reg))
    # We want to bypass version checks here to allow asynchronous updates for new releases
    __GOODBOY.registry['version_index.msgpack'] = None

with _resource_file("librosa.util.example_data", "index.json") as index:
    with index.open("r") as _fdesc:
        __TRACKMAP = json.load(_fdesc)


def example(key: str, *, hq: bool = False) -> str:
    """Retrieve the example recording identified by 'key'.

    The first time an example is requested, it will be downloaded from
    the remote repository over HTTPS.
    All subsequent requests will use a locally cached copy of the recording.

    For a list of examples (and their keys), see `librosa.util.list_examples`.

    By default, local files will be cached in the directory given by
    `pooch.os_cache('librosa')`.  You can override this by setting
    an environment variable ``LIBROSA_DATA_DIR`` prior to importing librosa:

    >>> import os
    >>> os.environ['LIBROSA_DATA_DIR'] = '/path/to/store/data'
    >>> import librosa

    Parameters
    ----------
    key : str
        The identifier for the track to load
    hq : bool
        If ``True``, return the high-quality version of the recording.
        If ``False``, return the 22KHz mono version of the recording.

    Returns
    -------
    path : str
        The path to the requested example file

    Examples
    --------
    Load "Hungarian Dance #5" by Johannes Brahms

    >>> y, sr = librosa.load(librosa.example('brahms'))

    Load "Vibe Ace" by Kevin MacLeod (the example previously packaged with librosa)
    in high-quality mode

    >>> y, sr = librosa.load(librosa.example('vibeace', hq=True))

    See Also
    --------
    librosa.util.list_examples
    pooch.os_cache
    """
    if key not in __TRACKMAP:
        raise ParameterError(f"Unknown example key: {key}")

    if hq:
        ext = ".hq.ogg"
    else:
        ext = ".ogg"

    return str(__GOODBOY.fetch(__TRACKMAP[key]["path"] + ext))


ex = example
"""Alias for example"""


def list_examples() -> None:
    """List the available audio recordings included with librosa.

    Each recording is given a unique identifier (e.g., "brahms" or "nutcracker"),
    listed in the first column of the output.

    A brief description is provided in the second column.

    See Also
    --------
    util.example
    util.example_info
    """
    print("AVAILABLE EXAMPLES")
    print("-" * 68)
    for key in sorted(__TRACKMAP.keys()):
        if key == "pibble":
            # Shh... she's sleeping
            continue
        print(f"{key:10}\t{__TRACKMAP[key]['desc']}")


def example_info(key: str) -> None:
    """Display licensing and metadata information for the given example recording.

    The first time an example is requested, it will be downloaded from
    the remote repository over HTTPS.
    All subsequent requests will use a locally cached copy of the recording.

    For a list of examples (and their keys), see `librosa.util.list_examples`.

    By default, local files will be cached in the directory given by
    `pooch.os_cache('librosa')`.  You can override this by setting
    an environment variable ``LIBROSA_DATA_DIR`` prior to importing librosa.

    Parameters
    ----------
    key : str
        The identifier for the recording (see `list_examples`)

    See Also
    --------
    librosa.util.example
    librosa.util.list_examples
    pooch.os_cache
    """
    if key not in __TRACKMAP:
        raise ParameterError(f"Unknown example key: {key}")

    license_file = __GOODBOY.fetch(__TRACKMAP[key]["path"] + ".txt")

    with open(license_file, "r") as fdesc:
        print(f"{key:10s}\t{__TRACKMAP[key]['desc']:s}")
        print("-" * 68)
        for line in fdesc:
            print(line)


def find_files(
    directory: Union[str, os.PathLike[Any]],
    *,
    ext: Optional[Union[str, List[str]]] = None,
    recurse: bool = True,
    case_sensitive: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[str]:
    """Get a sorted list of (audio) files in a directory or directory sub-tree.

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

    >>> # Avoid including search patterns in the path string
    >>> import glob
    >>> directory = '~/[202206] Music'
    >>> directory = glob.escape(directory)  # Escape the special characters
    >>> files = librosa.util.find_files(directory)

    Parameters
    ----------
    directory : str
        Path to look for files

    ext : str or list of str
        A file extension or list of file extensions to include in the search.

        Default: ``['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']``

    recurse : boolean
        If ``True``, then all subfolders of ``directory`` will be searched.

        Otherwise, only ``directory`` will be searched.

    case_sensitive : boolean
        If ``False``, files matching upper-case version of
        extensions will be included.

    limit : int > 0 or None
        Return at most ``limit`` files. If ``None``, all files are returned.

    offset : int
        Return files starting at ``offset`` within the list.

        Use negative values to offset from the end of the list.

    Returns
    -------
    files : list of str
        The list of audio files.
    """
    if ext is None:
        ext = ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]

    elif isinstance(ext, str):
        ext = [ext]

    # Cast into a set
    ext = set(ext)

    # Generate upper-case versions
    if not case_sensitive:
        # Force to lower-case
        ext = {e.lower() for e in ext}
        # Add in upper-case versions
        ext |= {e.upper() for e in ext}

    fileset = set()

    if recurse:
        for walk in os.walk(directory):  # type: ignore
            fileset |= __get_files(walk[0], ext)
    else:
        fileset = __get_files(directory, ext)

    files = list(fileset)
    files.sort()
    files = files[offset:]
    if limit is not None:
        files = files[:limit]

    return files


def __get_files(dir_name: Union[str, os.PathLike[Any]], extensions: Set[str]):
    """Get a list of files in a single directory"""
    # Expand out the directory
    dir_name = os.path.abspath(os.path.expanduser(dir_name))

    myfiles = set()

    for sub_ext in extensions:
        globstr = os.path.join(dir_name, "*" + os.path.extsep + sub_ext)
        myfiles |= set(glob.glob(globstr))

    return myfiles


def cite(version: Optional[str]=None) -> str:
    """Print the citation information for librosa.

    Parameters
    ----------
    version : str or None
        The version of librosa to cite. If None, the current version is used.

    Returns
    -------
    doi : str
        The DOI for the given version of librosa.

    Raises
    ------
    ParameterError
        If the requested version is not found in the citation index.

    Examples
    --------
    >>> librosa.cite("0.10.1")
    "https://doi.org/10.5281/zenodo.8252662"
    """
    if version is None:
        version = librosa_version

    version_data = __GOODBOY.fetch("version_index.msgpack")
    with open(version_data, "rb") as fdesc:
        version_index = msgpack.load(fdesc)

    if version not in version_index:
        if "dev" in version:
            raise ParameterError(f"Version {version} is not yet released and therefore does not yet have a citable DOI.")
        else:
            raise ParameterError(f"Version {version} not found in the citation index")

    return f"https://doi.org/{version_index[version]}"
