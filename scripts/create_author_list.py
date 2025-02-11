#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to retrieve contributors from GitHub API.
Modified to use urllib (instead of requests) for Python 3.13.
Added URL scheme validation to mitigate Bandit's B310 warning.
"""

import urllib.request
import urllib.error  # For handling HTTP and URL errors
import urllib.parse  # For parsing URLs to validate the scheme
import json  # For JSON parsing
import operator


def validate_url(url):
    """
    Validate the URL scheme to only allow http and https.
    This mitigates potential security risks by disallowing other schemes like file://.
    """
    parsed = urllib.parse.urlparse(url)
    assert parsed.scheme in ("http", "https"), f"Invalid URL scheme: {parsed.scheme}"
    return url


def parse_link_header(link_header):
    """
    Parse the HTTP Link header into a dictionary.
    The Link header contains pagination info (e.g., next, last).
    """
    links = {}
    if not link_header:
        return links
    # Split the header into comma-separated parts
    parts = link_header.split(",")
    for part in parts:
        section = part.strip().split(";")
        if len(section) < 2:
            continue
        url_part = section[0].strip()
        # Remove angle brackets from the URL
        if url_part.startswith("<") and url_part.endswith(">"):
            url = url_part[1:-1]
        else:
            url = url_part
        # Look for the rel parameter in the remaining parts
        rel = None
        for param in section[1:]:
            param = param.strip()
            if param.startswith("rel="):
                # Remove surrounding quotes from the rel value
                rel = param.split("=", 1)[1].strip('"')
        if rel:
            links[rel] = {"url": url, "rel": rel}
    return links


def get_contributors(repo_owner, repo_name):
    # Construct the initial URL for GitHub API stats endpoint.
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/stats/contributors"
    contributors = {}

    # Loop over pages if the API paginates the results.
    while url:
        # Validate the URL scheme before making the request.
        try:
            # Create a Request object and add a User-Agent header (GitHub requires it).
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            # Use urlopen with a timeout (urllib supports only one timeout value).
            # Here we set a combined timeout of 30 seconds.
            parsed = urllib.parse.urlparse(url)
            assert parsed.scheme in (
                "http",
                "https",
            ), f"Invalid URL scheme: {parsed.scheme}"
            with urllib.request.urlopen(req, timeout=30) as response:  # nosec
                # Get HTTP status code. Note: Non-200 codes would raise an HTTPError.
                status_code = response.getcode()
                # Read and decode the response body.
                data = response.read().decode("utf-8")
                # Parse the JSON data.
                json_data = json.loads(data)

                # Although HTTPError is raised for non-200 statuses,
                # we check status_code here for consistency with the original script.
                if status_code != 200:
                    raise Exception(
                        f"Failed to retrieve contributors: {json_data.get('message', 'Unknown error')}"
                    )

                # Process each contributor in the JSON response.
                for contributor in json_data:
                    contributions = contributor["weeks"]
                    mod_lines = sum([week["d"] + week["a"] for week in contributions])
                    contributors[contributor["author"]["login"]] = mod_lines

                # Retrieve the Link header from the response to handle pagination.
                link_header = response.info().get("Link")
                links = parse_link_header(link_header)
                # If there is a "next" link, update the URL; otherwise, we're done.
                if "next" in links:
                    url = links["next"]["url"]
                else:
                    url = None

        except urllib.error.HTTPError as e:
            # Read the error body to extract a useful error message.
            error_body = e.read().decode("utf-8")
            try:
                error_json = json.loads(error_body)
                error_message = error_json.get("message", "Unknown error")
            except json.JSONDecodeError:
                error_message = error_body
            raise Exception(f"HTTPError {e.code}: {error_message}")
        except urllib.error.URLError as e:
            # Handle URL errors (e.g., connection issues).
            raise Exception(f"URLError: {e.reason}")

    # Return the contributors sorted by the number of modified lines in descending order.
    return sorted(contributors.items(), key=operator.itemgetter(1), reverse=True)


if __name__ == "__main__":
    repo_owner = "librosa"
    repo_name = "librosa"

    contributors = get_contributors(repo_owner, repo_name)
    for contributor, mod_lines in contributors:
        print(f"{contributor:30s}|\t{mod_lines:7d} lines added/deleted")
