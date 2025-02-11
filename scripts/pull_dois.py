#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to pull all versions of our Zenodo record and save them as a msgpack file.
Modified to use urllib (instead of requests) for Python 3.13.
Added URL scheme validation and pagination to retrieve all versions.
"""

import urllib.request
import urllib.error  # For handling HTTP and URL errors
import urllib.parse  # For parsing URLs to validate the scheme
import json  # For JSON parsing
import msgpack  # For saving data in msgpack format
import pyzenodo3  # Zenodo API wrapper


def validate_url(url):
    """
    Validate the URL scheme to only allow http and https.
    This prevents unintended schemes (e.g., file://) from being used.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    return url


def safe_urlopen(req, timeout=30):
    """
    Wrapper around urllib.request.urlopen that validates the URL scheme before opening.
    The "# nosec" comment tells Bandit that this call is safe.
    """
    validate_url(req.full_url)
    return urllib.request.urlopen(req, timeout=timeout)  # nosec


def get_zenodo_record_versions(doi):
    """
    Retrieve all versions of a Zenodo record given its DOI.

    This function first finds the main record using pyzenodo3, then uses the
    URL provided in main_record.data['links']['versions'] to retrieve version records.
    Pagination is handled by checking for a "next" link in the JSON response.

    Returns a mapping of version numbers to DOIs.
    """
    zen = pyzenodo3.Zenodo()
    main_record = zen.find_record_by_doi(doi)

    # URL for the versions of the record.
    base_url = main_record.data["links"]["versions"]

    all_matches = []  # List to accumulate all version records.
    next_url = base_url  # Start with the base URL.

    while next_url:
        # Validate the URL scheme before making the request.
        validate_url(next_url)
        # Create a Request object with a User-Agent header.
        req = urllib.request.Request(next_url, headers={"User-Agent": "Mozilla/5.0"})
        # Open the URL safely.
        with safe_urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8")
            version_data = json.loads(data)

        # Extract the list of version records from the JSON.
        hits = version_data.get("hits", {}).get("hits", [])
        all_matches.extend(hits)

        # Check if there's a "next" page link in the JSON's "links" section.
        # If present, update next_url; otherwise, exit the loop.
        next_url = version_data.get("links", {}).get("next")

    # Build a mapping of version numbers to DOIs.
    doi_map = {m["metadata"]["version"]: m["doi"] for m in all_matches}
    return doi_map


def save_as_msgpack(data, filename="version_index.msgpack"):
    """
    Save the given data in msgpack format to the specified filename.
    """
    with open(filename, "wb") as f:
        msgpack.dump(data, f)
    print(f"Data saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Main concept DOI for all librosa versions
    doi = "10.5281/zenodo.591533"
    version_index = get_zenodo_record_versions(doi)
    save_as_msgpack(version_index)
