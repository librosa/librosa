#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare release materials for Librosa.

Automates step 2 (changelog fragment) and step 3 (AUTHORS.md updates) of the
Librosa publishing process using GitHub CLI (`gh`).

Steps automated:
1. Identifies the target version from `librosa/version.py` (or CLI override).
2. Queries GitHub for all issues and pull requests attached to the milestone
   matching that version number.
3. Groups PRs with any issues they resolve, fix, close, or address.
4. Generates a ReStructuredText summary fragment suitable for pasting into
   `docs/changelog.rst`.
5. Compares PR authors against `AUTHORS.md` and generates properly formatted
   lines for any new contributors to append to `AUTHORS.md`.
"""

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


def check_gh_cli() -> str:
    """Verify that the `gh` command-line tool is installed and available.

    Returns
    -------
    gh_path : str
        The absolute path to the `gh` executable.

    Raises
    ------
    RuntimeError
        If `gh` is not found in PATH or not executable.
    """
    gh_path = shutil.which("gh")
    if not gh_path:
        raise RuntimeError(
            "The GitHub CLI tool (`gh`) is required but was not found in PATH. "
            "Please install it from https://cli.github.com/ and authenticate with `gh auth login`."
        )
    return gh_path


def run_gh_command(gh_path: str, args: list[str]) -> str:
    """Run a `gh` CLI command and return its stdout.

    Parameters
    ----------
    gh_path : str
        Path to the `gh` executable.
    args : list of str
        Arguments to pass to `gh`.

    Returns
    -------
    output : str
        Standard output from the command.

    Raises
    ------
    RuntimeError
        If the command fails.
    """
    cmd = [gh_path, *args]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {err_msg}") from exc


def get_repo_root() -> Path:
    """Find the root directory of the librosa repository.

    Returns
    -------
    repo_root : Path
        Absolute path to the repository root.
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if (repo_root / "librosa" / "version.py").exists():
        return repo_root
    # Fallback to current working directory if plausible
    cwd = Path.cwd().resolve()
    if (cwd / "librosa" / "version.py").exists():
        return cwd
    return repo_root


def detect_repo(gh_path: str, repo_root: Path) -> str:
    """Detect the GitHub repository owner/name.

    Parameters
    ----------
    gh_path : str
        Path to the `gh` executable.
    repo_root : Path
        Path to the repository root.

    Returns
    -------
    repo : str
        Repository in "owner/repo" format (e.g. "librosa/librosa").
    """
    git_path = shutil.which("git")
    if git_path:
        try:
            proc = subprocess.run(  # noqa: S603
                [git_path, "-C", str(repo_root), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=True,
            )
            url = proc.stdout.strip()
            # Parse git@github.com:owner/repo.git or https://github.com/owner/repo.git
            match = re.search(r"github\.com[:/]([^/]+)/([^/\.]+)(?:\.git)?", url)
            if match:
                return f"{match.group(1)}/{match.group(2)}"
        except (subprocess.SubprocessError, OSError):
            pass

    try:
        out = run_gh_command(gh_path, ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
        repo = out.strip()
        if repo:
            return repo
    except RuntimeError:
        pass

    return "librosa/librosa"


def get_target_version(version_path: Path) -> str:
    """Read the target version number from version.py.

    Parameters
    ----------
    version_path : Path
        Path to `version.py`.

    Returns
    -------
    version : str
        Target version string.

    Raises
    ------
    ValueError
        If the version string cannot be found.
    """
    if not version_path.exists():
        raise ValueError(f"Version file not found: {version_path}")

    content = version_path.read_text(encoding="utf-8")
    match = re.search(r'^[ \t]*version[ \t]*=[ \t]*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find `version = ...` definition in {version_path}")
    return match.group(1).strip()


def find_milestone(gh_path: str, repo: str, version: str) -> dict:
    """Find the milestone object on GitHub matching the target version.

    Parameters
    ----------
    gh_path : str
        Path to the `gh` executable.
    repo : str
        Repository in "owner/repo" format.
    version : str
        Target version string.

    Returns
    -------
    milestone : dict
        Milestone data dictionary from GitHub API.

    Raises
    ------
    RuntimeError
        If no matching milestone is found.
    """
    output = run_gh_command(gh_path, ["api", f"repos/{repo}/milestones?state=all", "--paginate"])
    milestones = json.loads(output)

    norm_target = version.lstrip("v").strip()
    for m in milestones:
        m_title = m.get("title", "").strip()
        if m_title.lstrip("v") == norm_target or m_title == version:
            return m

    available = [m.get("title", "") for m in milestones]
    raise RuntimeError(
        f"No milestone matching version '{version}' found in repository '{repo}'. "
        f"Available milestones: {', '.join(available)}"
    )


def fetch_milestone_items(gh_path: str, repo: str, milestone_number: int) -> tuple[list[dict], list[dict]]:
    """Fetch all issues and pull requests attached to a milestone.

    Parameters
    ----------
    gh_path : str
        Path to the `gh` executable.
    repo : str
        Repository in "owner/repo" format.
    milestone_number : int
        The milestone number.

    Returns
    -------
    prs : list of dict
        Pull requests attached to the milestone.
    issues : list of dict
        Issues attached to the milestone.
    """
    output = run_gh_command(
        gh_path,
        ["api", f"repos/{repo}/issues?milestone={milestone_number}&state=all", "--paginate"],
    )
    items = json.loads(output)
    prs = [item for item in items if "pull_request" in item]
    issues = [item for item in items if "pull_request" not in item]
    return prs, issues


def fetch_user_name(gh_path: str, login: str, cache: dict[str, str]) -> str:
    """Retrieve the display name for a GitHub username, with fallback to username.

    Parameters
    ----------
    gh_path : str
        Path to the `gh` executable.
    login : str
        GitHub user login.
    cache : dict of str to str
        In-memory cache of username -> display name.

    Returns
    -------
    name : str
        Display name or login.
    """
    if login in cache:
        return cache[login]

    try:
        output = run_gh_command(gh_path, ["api", f"users/{login}"])
        user_data = json.loads(output)
        name = (user_data.get("name") or "").strip()
        cache[login] = name if name else login
    except Exception:
        cache[login] = login

    return cache[login]


def get_existing_authors(authors_path: Path) -> set[str]:
    """Parse existing GitHub user logins from AUTHORS.md.

    Parameters
    ----------
    authors_path : Path
        Path to `AUTHORS.md`.

    Returns
    -------
    authors : set of str
        Set of lowercase GitHub logins already present in AUTHORS.md.
    """
    if not authors_path.exists():
        return set()

    content = authors_path.read_text(encoding="utf-8")
    matches = re.findall(r"<https://github\.com/([^>]+)>", content, re.IGNORECASE)
    return {m.strip().lower() for m in matches if m.strip()}


def extract_linked_issues(
    prs: list[dict],
    milestone_issue_numbers: set[int],
    gh_path: str,
    repo: str,
) -> tuple[list[dict], list[int]]:
    """Group PRs with the issues they resolve or reference.

    Parameters
    ----------
    prs : list of dict
        List of PR items.
    milestone_issue_numbers : set of int
        Issue numbers tagged with this milestone.
    gh_path : str
        Path to the `gh` executable.
    repo : str
        Repository in "owner/repo" format.

    Returns
    -------
    grouped_prs : list of dict
        List of dicts with keys 'pr' and 'linked_issues' (sorted list of int).
    unclaimed_issues : list of int
        Milestone issue numbers that were not linked to any PR.
    """
    # Fetch GraphQL closingIssuesReferences where available
    closing_refs_by_pr: dict[int, set[int]] = {}
    try:
        output = run_gh_command(
            gh_path,
            [
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "all",
                "--limit",
                "100",
                "--json",
                "number,closingIssuesReferences",
            ],
        )
        pr_closing_info = json.loads(output)
        for item in pr_closing_info:
            p_num = item["number"]
            closing_refs = item.get("closingIssuesReferences") or []
            closing_refs_by_pr[p_num] = {
                ref["number"] for ref in closing_refs if ref.get("number") in milestone_issue_numbers
            }
    except RuntimeError:
        pass

    pattern = re.compile(
        r"(?i)(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|address(?:e[sd])?|ref(?:erence)?s?|refactor/)"
        r"\s*[:#\s]*(?:issue\s*)?#?(\d+)"
    )

    claimed_issues: set[int] = set()
    grouped_prs: list[dict] = []

    for pr in prs:
        pr_num = pr["number"]
        text = f"{pr.get('title', '')}\n{pr.get('body', '')}"
        regex_matches = {int(n) for n in pattern.findall(text)}
        graphql_matches = closing_refs_by_pr.get(pr_num, set())

        matched = (regex_matches | graphql_matches) & milestone_issue_numbers
        # Ensure issue isn't claimed twice
        available_matches = sorted(matched - claimed_issues)
        claimed_issues.update(available_matches)

        grouped_prs.append({
            "pr": pr,
            "linked_issues": available_matches,
        })

    unclaimed_issues = sorted(milestone_issue_numbers - claimed_issues)
    return grouped_prs, unclaimed_issues


def categorize_entry(labels: list[dict]) -> str:
    """Categorize a changelog entry based on GitHub labels.

    Parameters
    ----------
    labels : list of dict
        List of GitHub label dictionaries.

    Returns
    -------
    category : str
        Category heading name.
    """
    names = [lbl.get("name", "").lower() for lbl in labels]

    if any("feature" in n for n in names):
        return "New features"
    if any("bug" in n for n in names):
        return "Bug fixes"
    if any("enhancement" in n for n in names):
        return "Enhancements"
    if any("doc" in n for n in names):
        return "Documentation"
    if any(
        n in ["testing", "management", "maintenance", "ci", "infrastructure", "build"]
        or "test" in n
        or "maint" in n
        for n in names
    ):
        return "Maintenance"
    if any("api change" in n for n in names):
        return "API changes"
    if any("deprecation" in n for n in names):
        return "Deprecations"

    return "Other changes"


def format_bullet(
    numbers: list[int],
    title: str,
    author: str,
    wrapper: textwrap.TextWrapper,
) -> list[str]:
    """Format a single bullet entry for the changelog fragment.

    Parameters
    ----------
    numbers : list of int
        List of PR/issue numbers, primary first.
    title : str
        Title description.
    author : str
        Author attribution string.
    wrapper : textwrap.TextWrapper
        Configured text wrapper.

    Returns
    -------
    lines : list of str
        Wrapped bullet lines.
    """
    ref_str = ", ".join(f"`#{n}`_" for n in numbers)
    clean_title = title.strip().rstrip(".")
    # Remove common prefix noise if present (e.g. [Bug]: or [Maint]:)
    clean_title = re.sub(r"^\[(?:Bug|Maint|Feature|Doc)\]:\s*", "", clean_title, flags=re.IGNORECASE)

    raw_bullet = f"{ref_str} {clean_title}. *{author}*"
    return wrapper.wrap(raw_bullet)


def generate_changelog_fragment(
    version: str,
    release_date: str,
    grouped_prs: list[dict],
    unclaimed_issues: list[dict],
    milestone_issues_dict: dict[int, dict],
    gh_path: str,
    repo: str,
    user_cache: dict[str, str],
) -> str:
    """Generate the ReStructuredText changelog fragment.

    Parameters
    ----------
    version : str
        Target release version.
    release_date : str
        Release date string (YYYY-MM-DD).
    grouped_prs : list of dict
        Grouped PR items.
    unclaimed_issues : list of dict
        Issues that were not linked to any PR.
    milestone_issues_dict : dict of int to dict
        All milestone issue dicts keyed by issue number.
    gh_path : str
        Path to the `gh` executable.
    repo : str
        Repository in "owner/repo" format.
    user_cache : dict of str to str
        Username to display name cache.

    Returns
    -------
    fragment : str
        ReST changelog text.
    """
    category_order = [
        "New features",
        "Enhancements",
        "Bug fixes",
        "Documentation",
        "Maintenance",
        "API changes",
        "Deprecations",
        "Other changes",
    ]
    categorized_entries: dict[str, list[tuple[int, list[str]]]] = {cat: [] for cat in category_order}
    all_refs: dict[int, str] = {}  # number -> 'pull' or 'issues'

    wrapper = textwrap.TextWrapper(
        width=80,
        initial_indent="    - ",
        subsequent_indent="      ",
        break_long_words=False,
    )

    for item in grouped_prs:
        pr = item["pr"]
        pr_num = pr["number"]
        linked = item["linked_issues"]

        all_refs[pr_num] = "pull"
        for l_num in linked:
            all_refs[l_num] = "issues"

        # Combine labels from PR and linked issues
        combined_labels = list(pr.get("labels", []))
        for l_num in linked:
            combined_labels.extend(milestone_issues_dict[l_num].get("labels", []))
        cat = categorize_entry(combined_labels)

        author_login = pr.get("user", {}).get("login", "")
        author_name = fetch_user_name(gh_path, author_login, user_cache)

        bullet_lines = format_bullet(
            numbers=[pr_num, *linked],
            title=pr.get("title", ""),
            author=author_name,
            wrapper=wrapper,
        )
        categorized_entries[cat].append((pr_num, bullet_lines))

    for issue in unclaimed_issues:
        num = issue["number"]
        all_refs[num] = "issues"
        cat = categorize_entry(issue.get("labels", []))

        author_login = issue.get("user", {}).get("login", "")
        author_name = fetch_user_name(gh_path, author_login, user_cache)

        bullet_lines = format_bullet(
            numbers=[num],
            title=issue.get("title", ""),
            author=author_name,
            wrapper=wrapper,
        )
        categorized_entries[cat].append((num, bullet_lines))

    lines: list[str] = []
    header = f"v{version}"
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(release_date)
    lines.append("")

    for cat in category_order:
        entries = categorized_entries[cat]
        if not entries:
            continue
        lines.append(cat)
        # Sort descending by primary number
        entries.sort(key=lambda x: x[0], reverse=True)
        for _, bullet_lines in entries:
            lines.extend(bullet_lines)
        lines.append("")

    # Link definitions at bottom
    for num in sorted(all_refs.keys(), reverse=True):
        kind = all_refs[num]
        lines.append(f".. _#{num}: https://github.com/{repo}/{kind}/{num}")

    return "\n".join(lines)


def find_new_authors(
    prs: list[dict],
    existing_authors: set[str],
    gh_path: str,
    user_cache: dict[str, str],
) -> list[tuple[str, str]]:
    """Identify PR authors not present in AUTHORS.md.

    Parameters
    ----------
    prs : list of dict
        Pull requests.
    existing_authors : set of str
        Existing lowercase logins from AUTHORS.md.
    gh_path : str
        Path to the `gh` executable.
    user_cache : dict of str to str
        Username to display name cache.

    Returns
    -------
    new_authors : list of tuple of str
        List of (display_name, login) tuples.
    """
    new_authors_dict: dict[str, str] = {}

    for pr in prs:
        user = pr.get("user", {})
        login = user.get("login", "")
        if not login:
            continue
        # Skip bot accounts
        if user.get("type") == "Bot" or login.endswith("[bot]"):
            continue

        if login.lower() not in existing_authors and login not in new_authors_dict:
            display_name = fetch_user_name(gh_path, login, user_cache)
            new_authors_dict[login] = display_name

    return [(name, login) for login, name in new_authors_dict.items()]


def format_author_lines(new_authors: list[tuple[str, str]]) -> str:
    """Format new contributors into AUTHORS.md bullet format.

    Parameters
    ----------
    new_authors : list of tuple of str
        List of (display_name, login).

    Returns
    -------
    formatted : str
        Markdown list of authors.
    """
    lines = [f"* {name} <https://github.com/{login}>" for name, login in new_authors]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Automate changelog fragment and AUTHORS.md updates for Librosa releases."
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Target release version. Defaults to reading `librosa/version.py`.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="GitHub repository (owner/repo). Defaults to git remote or librosa/librosa.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.date.today().isoformat(),
        help="Release date in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument(
        "--authors-file",
        type=Path,
        default=None,
        help="Path to AUTHORS.md (default: repo root AUTHORS.md).",
    )
    parser.add_argument(
        "--version-file",
        type=Path,
        default=None,
        help="Path to version.py (default: repo root librosa/version.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to write the changelog fragment to.",
    )
    parser.add_argument(
        "--changelog-only",
        action="store_true",
        help="Only display/write the changelog fragment.",
    )
    parser.add_argument(
        "--authors-only",
        action="store_true",
        help="Only display new authors for AUTHORS.md.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute release preparation workflow."""
    args = parse_args()

    gh_path = check_gh_cli()
    repo_root = get_repo_root()

    repo = args.repo or detect_repo(gh_path, repo_root)

    version_file = args.version_file or (repo_root / "librosa" / "version.py")
    target_version = args.version or get_target_version(version_file)

    authors_file = args.authors_file or (repo_root / "AUTHORS.md")

    sys.stderr.write(f"Target version: {target_version}\n")
    sys.stderr.write(f"Repository:     {repo}\n")

    # Locate milestone
    milestone = find_milestone(gh_path, repo, target_version)
    m_num = milestone["number"]
    m_title = milestone.get("title", target_version)
    sys.stderr.write(f"Milestone:      {m_title} (#{m_num})\n")

    # Fetch milestone items
    prs, issues = fetch_milestone_items(gh_path, repo, m_num)
    sys.stderr.write(f"Found:          {len(prs)} PR(s), {len(issues)} Issue(s)\n\n")

    milestone_issues_dict = {i["number"]: i for i in issues}
    milestone_issue_numbers = set(milestone_issues_dict.keys())

    # Link issues to PRs
    grouped_prs, unclaimed_issue_nums = extract_linked_issues(
        prs, milestone_issue_numbers, gh_path, repo
    )
    unclaimed_issues = [milestone_issues_dict[n] for n in unclaimed_issue_nums]

    user_cache: dict[str, str] = {}

    # 1. Changelog fragment
    changelog_fragment = generate_changelog_fragment(
        version=target_version,
        release_date=args.date,
        grouped_prs=grouped_prs,
        unclaimed_issues=unclaimed_issues,
        milestone_issues_dict=milestone_issues_dict,
        gh_path=gh_path,
        repo=repo,
        user_cache=user_cache,
    )

    if args.output:
        args.output.write_text(changelog_fragment + "\n", encoding="utf-8")
        sys.stderr.write(f"Wrote changelog fragment to {args.output}\n")

    # 2. Authors update
    existing_authors = get_existing_authors(authors_file)
    new_authors = find_new_authors(prs, existing_authors, gh_path, user_cache)
    authors_output = format_author_lines(new_authors)

    # Output printing
    divider = "=" * 80

    if not args.authors_only:
        if not args.changelog_only:
            print(divider)
            print(f"Changelog fragment for docs/changelog.rst (v{target_version}):")
            print(divider)
        print(changelog_fragment)
        if not args.changelog_only:
            print()

    if not args.changelog_only:
        if not args.authors_only:
            print(divider)
            print("New contributors to append to AUTHORS.md:")
            print(divider)
        if new_authors:
            print(authors_output)
        else:
            print("(No new contributors found; all PR authors are already in AUTHORS.md)")

        if not args.authors_only:
            print(divider)
            print("Next steps reminder (from publishing instructions):")
            print("1. Verify and paste the changelog fragment into `docs/changelog.rst`.")
            print("2. Verify and paste any new authors above into `AUTHORS.md`.")
            print("3. Run `python scripts/create_zenodo_json.py` to generate the updated .zenodo.json.")
            print(divider)


if __name__ == "__main__":
    main()
