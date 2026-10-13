#!/usr/bin/env python3
"""Fail when an SEO change touches DistilRL's protected learning assets.

The SEO implementation deliberately keeps the 24 chapter Markdown files and
the original chapter illustrations byte-for-byte unchanged.  This check works
both in a pull request (pass the PR base SHA with ``--base``) and in a local
checkout (``origin/main`` is selected automatically when it exists).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


PROTECTED_DIRECTORIES = (
    "docs/Contents",
    "docs/Contents_zh",
)


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    """Run Git without a shell so paths and revisions cannot be reinterpreted."""

    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_root(start: Path) -> Path:
    result = run_git(start, "rev-parse", "--show-toplevel")
    return Path(result.stdout.decode("utf-8").strip()).resolve()


def revision_exists(repo: Path, revision: str) -> bool:
    if not revision or set(revision) == {"0"}:
        return False
    result = run_git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}", check=False)
    return result.returncode == 0


def select_base(repo: Path, requested: str | None) -> str:
    if requested:
        if revision_exists(repo, requested):
            return requested
        raise ValueError(f"base revision does not exist locally: {requested}")

    candidates = (
        os.environ.get("SEO_BASE_SHA"),
        os.environ.get("GITHUB_BASE_SHA"),
        "origin/main",
        "main",
    )
    for candidate in candidates:
        if candidate and revision_exists(repo, candidate):
            return candidate
    raise ValueError("cannot find a baseline; fetch origin/main or pass --base <revision>")


def is_protected(path_string: str) -> bool:
    """Return whether a repository-relative POSIX path is protected."""

    path = PurePosixPath(path_string)
    normalized = path.as_posix().lstrip("./")

    if any(normalized == directory or normalized.startswith(f"{directory}/") for directory in PROTECTED_DIRECTORIES):
        return True

    parts = PurePosixPath(normalized).parts
    if len(parts) < 5 or parts[:3] != ("docs", "_static", "img"):
        return False
    # Protect every existing and future chapter directory, but not the logo or
    # new marketing images placed in docs/_static/seo.
    return parts[3].startswith("chapter")


def nul_paths(result: subprocess.CompletedProcess[bytes]) -> set[str]:
    return {
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    }


def changed_paths(repo: Path, base: str, head: str) -> dict[str, set[str]]:
    changes: dict[str, set[str]] = {}

    committed = nul_paths(
        run_git(
            repo,
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            f"{base}...{head}",
            "--",
        )
    )
    for path in committed:
        changes.setdefault(path, set()).add(f"committed diff ({base}...{head})")

    # Also inspect staged and unstaged local edits.  In CI this is normally
    # empty; locally it prevents an accidental content edit from being missed.
    worktree = nul_paths(
        run_git(repo, "diff", "--name-only", "-z", "--no-renames", head, "--")
    )
    for path in worktree:
        changes.setdefault(path, set()).add(f"working tree diff ({head})")

    untracked = nul_paths(run_git(repo, "ls-files", "--others", "--exclude-standard", "-z"))
    for path in untracked:
        changes.setdefault(path, set()).add("untracked")

    return changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensure DistilRL chapter text and original chapter images are unchanged."
    )
    parser.add_argument(
        "--base",
        help="Git revision used as the protected-content baseline (defaults to PR env or origin/main).",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Git revision containing the candidate change (default: HEAD).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path inside the repository (default: current directory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo = git_root(args.repo_root.resolve())
        base = select_base(repo, args.base)
        if not revision_exists(repo, args.head):
            raise ValueError(f"head revision does not exist locally: {args.head}")
        changes = changed_paths(repo, base, args.head)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"protected-content check could not run: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
        return 2

    violations = {path: reasons for path, reasons in changes.items() if is_protected(path)}
    if violations:
        print("Protected learning content changed; SEO work must not modify these paths:", file=sys.stderr)
        for path in sorted(violations):
            print(f"  - {path} [{', '.join(sorted(violations[path]))}]", file=sys.stderr)
        print(
            "Protected scopes: docs/Contents/**, docs/Contents_zh/**, "
            "docs/_static/img/chapter*/**",
            file=sys.stderr,
        )
        return 1

    print(
        f"Protected-content check passed against {base}: "
        "24 chapter sources and original chapter image directories are unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
