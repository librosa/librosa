#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a zenodo.json metadata file.

This should only be run after AUTHORS.md has been updated.

Note that the output of this script should not be taken directly, as it cannot capture ORCID
information.

ORCID ids should be migrated manually before overwriting the old .zenodo.json file.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import httpx

ORCID_RE = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[0-9Xx]")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Zenodo metadata.")
    parser.add_argument(
        "--authors",
        type=Path,
        default=Path("../AUTHORS.md"),
        help="Path to the canonical authors mapping file."
    )
    parser.add_argument("--owner", type=str, default="librosa")
    parser.add_argument("--repo", type=str, default="librosa")
    return parser.parse_args()


def parse_authors(filepath: Path) -> dict:
    """Parse AUTHORS.md into {login_lower: {'name': str, 'orcid': str|None}}."""
    authors_map = {}
    if not filepath.exists():
        print(f"Warning: {filepath} not found.", file=sys.stderr)
        return authors_map

    pattern = re.compile(
        r"^\*\s+(.+?)\s+<https://github\.com/([^>]+)>(?:\s+\[ORCID:\s*(\d{4}-\d{4}-\d{4}-\d{3}[0-9xX])\])?",
        re.IGNORECASE
    )

    for line in filepath.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        name = match.group(1).strip()
        login = match.group(2).strip().lower()
        orcid = match.group(3).strip().upper() if match.group(3) else None
        authors_map[login] = {"name": name, "orcid": orcid}

    return authors_map


async def fetch_repo_metadata(client: httpx.AsyncClient, owner: str, repo: str, headers: dict) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    return {
        "title": data.get("name", repo),
        "description": data.get("description", "No description provided.")
    }


async def fetch_all_contributors(client: httpx.AsyncClient, owner: str, repo: str, headers: dict) -> list[str]:
    """
    Fetch all contributors from paginated REST endpoint (not capped at 100 total users).
    Returns lowercase logins in API order.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=100"
    logins = []

    while url:
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        for contributor in response.json():
            if contributor.get("type") == "User" and contributor.get("login"):
                logins.append(contributor["login"].lower())

        next_url = None
        link_header = response.headers.get("Link", "")
        for part in link_header.split(","):
            if 'rel="next"' in part:
                m = re.search(r"<([^>]+)>", part)
                if m:
                    next_url = m.group(1)
                    break
        url = next_url

    # stable dedupe
    seen = set()
    deduped = []
    for login in logins:
        if login not in seen:
            seen.add(login)
            deduped.append(login)
    return deduped


async def fetch_contributor_churn(client: httpx.AsyncClient, owner: str, repo: str, headers: dict) -> dict[str, int]:
    """
    Fetch churn (adds + deletes) from /stats/contributors.
    Returns {login_lower: churn}.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/stats/contributors"

    # GitHub may return 202 while generating stats
    for _ in range(8):
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            break
        if response.status_code == 202:
            await asyncio.sleep(3)
            continue
        response.raise_for_status()
    else:
        raise TimeoutError("GitHub API timed out building contributor stats.")

    churn = {}
    for contributor in response.json():
        author = contributor.get("author") or {}
        login = (author.get("login") or "").lower()
        if not login or author.get("type") != "User":
            continue

        total = 0
        for w in contributor.get("weeks", []):
            total += int(w.get("a", 0) or 0) + int(w.get("d", 0) or 0)
        churn[login] = total

    return churn


def extract_orcid(node: dict) -> str | None:
    # 1) social links
    for account in (node.get("socialAccounts") or {}).get("nodes", []):
        url = (account.get("url") or "").strip()
        if "orcid.org" in url.lower():
            oid = url.rstrip("/").split("/")[-1]
            m = ORCID_RE.search(oid)
            if m:
                return m.group(0).upper()

    # 2) bio
    bio = node.get("bio") or ""
    m = ORCID_RE.search(bio)
    if m:
        return m.group(0).upper()

    return None


async def fetch_user_profiles_graphql(client: httpx.AsyncClient, logins: list[str], headers: dict) -> dict:
    """
    Fetch login/company/bio/socialAccounts for many users via GraphQL.
    Returns {login_lower: node}.
    """
    url = "https://api.github.com/graphql"
    user_data = {}
    chunk_size = 50

    for i in range(0, len(logins), chunk_size):
        chunk = logins[i:i + chunk_size]

        lines = ["query {"]
        for idx, login in enumerate(chunk):
            lines.append(
                f"""
                u_{idx}: user(login: "{login}") {{
                    login
                    company
                    bio
                    socialAccounts(first: 20) {{
                        nodes {{ url }}
                    }}
                }}
                """
            )
        lines.append("}")

        response = await client.post(url, headers=headers, json={"query": "\n".join(lines)})
        response.raise_for_status()
        payload = response.json()

        if payload.get("errors") and not payload.get("data"):
            raise RuntimeError(f"GraphQL error: {payload['errors']}")

        for node in (payload.get("data") or {}).values():
            if node and node.get("login"):
                user_data[node["login"].lower()] = node

    return user_data


async def main(args):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable is strictly required.")

    headers = {
        "User-Agent": "librosa-zenodo-builder",
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    canonical_authors = parse_authors(args.authors)

    async with httpx.AsyncClient(timeout=30.0) as client:
        repo_meta_task = fetch_repo_metadata(client, args.owner, args.repo, headers)
        all_contributors_task = fetch_all_contributors(client, args.owner, args.repo, headers)
        churn_task = fetch_contributor_churn(client, args.owner, args.repo, headers)

        repo_meta, all_contributors, churn_map = await asyncio.gather(
            repo_meta_task, all_contributors_task, churn_task
        )

        # union set for profile lookup (contributors + any AUTHORS entries)
        profile_logins = list(dict.fromkeys(all_contributors + list(canonical_authors.keys())))
        profiles = await fetch_user_profiles_graphql(client, profile_logins, headers)

    # Primary ordering: churn descending
    # tie-break by login for deterministic output
    churn_sorted = sorted(churn_map.items(), key=lambda kv: (-kv[1], kv[0]))
    ordered_logins = [login for login, _ in churn_sorted]

    # Ensure not truncated: include contributors absent from stats at tail (0 churn unknown)
    for login in all_contributors:
        if login not in churn_map:
            ordered_logins.append(login)

    # stable dedupe
    seen = set()
    ordered_logins = [x for x in ordered_logins if not (x in seen or seen.add(x))]

    creators = []
    seen_authors = set()

    # Build creators from ordered contributor list, but only if in AUTHORS.md
    for login in ordered_logins:
        if login not in canonical_authors:
            continue

        seen_authors.add(login)
        entry = canonical_authors[login]
        profile = profiles.get(login, {})

        creator = {"name": entry["name"]}

        company = profile.get("company")
        if company:
            creator["affiliation"] = company

        # Prefer AUTHORS.md ORCID, fallback to profile extraction
        orcid = entry.get("orcid") or extract_orcid(profile)
        if orcid:
            creator["orcid"] = orcid.upper()

        creators.append(creator)

    # Append AUTHORS.md entries not seen in contributor API results
    for login, entry in canonical_authors.items():
        if login in seen_authors:
            continue

        profile = profiles.get(login, {})
        creator = {"name": entry["name"]}

        company = profile.get("company")
        if company:
            creator["affiliation"] = company

        orcid = entry.get("orcid") or extract_orcid(profile)
        if orcid:
            creator["orcid"] = orcid.upper()

        creators.append(creator)

    result = {
        "title": repo_meta["title"],
        "description": repo_meta["description"],
        "upload_type": "software",
        "license": "ISC",
        "creators": creators,
    }

    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    parsed_args = parse_args()
    asyncio.run(main(parsed_args))
