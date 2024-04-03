#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to pull all versions of our Zenodo record and save them as a msgpack file.

This should be run after each release and the result sent to our data
repo.
"""
import pyzenodo3
import requests
import msgpack

def get_zenodo_record_versions(doi):

    zen = pyzenodo3.Zenodo()
    main_record = zen.find_record_by_doi(doi)

    links_url = main_record.data['links']['versions']

    version_data = requests.get(links_url, timeout=(3.05, 27)).json()

    matches = version_data['hits']['hits']
    doi_map = {m['metadata']['version']: m['doi'] for m in matches}

    return doi_map

def save_as_msgpack(data, filename="version_index.msgpack"):
    # Save the dictionary in msgpack format
    with open(filename, "wb") as f:
        msgpack.dump(data, f)
    print(f"Data saved to {filename}")

# Example usage
if __name__ == '__main__':
    # Main concept DOI for all librosa versions
    doi = "10.5281/zenodo.591533" 
    version_index = get_zenodo_record_versions(doi)
    save_as_msgpack(version_index)
