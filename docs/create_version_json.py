#!/usr/bin/env python3
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate versions.json for PyData Sphinx Theme switcher."
    )
    parser.add_argument(
        "--output",
        default="build/html/versions.json",
        help="Output path for versions.json.",
    )
    parser.add_argument(
        "--preferred",
        required=True,
        help="The tag/version that is preferred (gets preferred: true).",
    )
    parser.add_argument(
        "--doc-root",
        default="https://librosa.org/doc/",
        help="Base URL root for documentation.",
    )
    parser.add_argument(
        "versions",
        nargs="+",
        help="List of all version strings (e.g., dev 0.11.0 0.10.2).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = []

    for v in args.versions:
        # dev is typically routed directly as a subfolder, as are release tags
        url = f"{args.doc_root}/{v}/"
        entry = {"name": v, "version": v, "url": url}

        if v == args.preferred:
            entry["preferred"] = True

        output.append(entry)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
