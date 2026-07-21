#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated script to fetch DOIs and bibtex entries from zenodo.org and DataCite, with improved error handling and compliance with API rate limits.

Revised 2026-07-21, coauthored by gemini 3.1 pro.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import msgpack


def get_zenodo_record_versions(concept_doi: str) -> dict:
    if not concept_doi.startswith("10.5281/zenodo."):
        raise ValueError("Invalid Zenodo DOI format. Expected '10.5281/zenodo.<id>'")

    concept_id = concept_doi.split(".")[-1]

    base_url = "https://zenodo.org/api/records"
    params = {
        "q": f"conceptrecid:{concept_id}",
        "all_versions": "true",
        "size": 25
    }

    next_url = f"{base_url}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "Librosa-Zenodo-Indexer/1.4 (Python/urllib)"}
    doi_map = {}

    while next_url:
        parsed = urllib.parse.urlparse(next_url)
        if parsed.scheme != "https":
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

        req = urllib.request.Request(next_url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=30) as response:  # nosec
                page_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"API request failed: {e.code} {e.reason} - {error_body}") from e

        hits = page_data.get("hits", {}).get("hits", [])
        for hit in hits:
            version = hit.get("metadata", {}).get("version")
            hit_doi = hit.get("doi") or hit.get("pids", {}).get("doi", {}).get("identifier")

            if version and hit_doi:
                doi_map[version] = hit_doi

        next_url = page_data.get("links", {}).get("next")

    return doi_map

def get_bibtex_index(doi_map: dict) -> dict:
    bib_map = {}
    headers = {"User-Agent": "Librosa-Zenodo-Indexer/1.4 (Python/urllib)"}

    for version, hit_doi in doi_map.items():
        # Query DataCite directly to bypass urllib redirect header truncation
        url = f"https://api.datacite.org/application/x-bibtex/{hit_doi}"
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=30) as response:  # nosec
                bib_map[version] = response.read().decode("utf-8").strip()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Failed to fetch BibTeX for {hit_doi}: {e.code} {e.reason}") from e

        # Enforce 0.6s delay to respect DataCite's ~100 req/min limit
        time.sleep(0.6)

    return bib_map

def save_as_msgpack(data: dict, filename: str):
    with open(filename, "wb") as f:
        msgpack.dump(data, f)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    doi = "10.5281/zenodo.591533"

    version_index = get_zenodo_record_versions(doi)
    save_as_msgpack(version_index, "version_index.msgpack")

    bib_index = get_bibtex_index(version_index)
    save_as_msgpack(bib_index, "bib_index.msgpack")
