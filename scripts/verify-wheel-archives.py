"""Reject damaged wheel archives before uploading or publishing them."""

import argparse
import sys
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("wheels", nargs="+")
    args = parser.parse_args()

    if args.expected_count is not None and len(args.wheels) != args.expected_count:
        print(
            f"Expected {args.expected_count} wheel archive(s), found {len(args.wheels)}",
            file=sys.stderr,
        )
        return 1

    failed = False
    for name in args.wheels:
        path = Path(name)
        try:
            with zipfile.ZipFile(path) as archive:
                bad_file = archive.testzip()
                if bad_file is not None:
                    raise zipfile.BadZipFile(f"bad CRC in {bad_file}")
        except (OSError, zipfile.BadZipFile) as exc:
            print(f"Invalid wheel archive {path}: {exc}", file=sys.stderr)
            failed = True
        else:
            print(f"Valid wheel archive: {path}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
