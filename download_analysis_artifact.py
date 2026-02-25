#!/usr/bin/env python3
"""
download_analysis_artifact.py

Downloads the latest ``analysis_results.parquet`` artifact produced by the
``Analyze Stocks and Generate Report`` GitHub Actions workflow and saves it to
``data/results/it/analysis_results.parquet``.

Authentication
--------------
The script resolves a GitHub token in this order:

1. ``GITHUB_TOKEN`` environment variable  (preferred in CI)
2. ``gh auth token`` CLI command           (transparent for local dev)

The token needs at minimum ``repo`` scope (or ``actions:read`` for public repos).

Usage
-----
    # Option A: explicit token
    GITHUB_TOKEN=ghp_... python download_analysis_artifact.py

    # Option B: relies on `gh` CLI auth (already configured)
    python download_analysis_artifact.py

Invariants
----------
- The workflow name must match the ``name:`` field in the YAML file exactly.
- The artifact name prefix must match what ``actions/upload-artifact`` uses.
- The parquet filename inside the zip must be ``analysis_results.parquet``.
"""

import io
import logging
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration — single source of truth for all tuneable constants
# ---------------------------------------------------------------------------

REPO             = "laceto/myfinance2"
WORKFLOW_FILE    = "analyze_and_report.yml"
ARTIFACT_PREFIX  = "analysis-results-"
PARQUET_FILENAME = "analysis_results.parquet"
OUTPUT_PATH      = Path("data/results/it/analysis_results.parquet")

GITHUB_API_VERSION = "2022-11-28"
DOWNLOAD_CHUNK_SIZE = 64 * 1024  # 64 KB per chunk

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GitHub API client
# ---------------------------------------------------------------------------

class GitHubClient:
    """Thin wrapper around the GitHub REST API using a persistent session."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        })

    def get(self, path: str, **params) -> dict:
        """GET a JSON endpoint. Raises on non-2xx status."""
        url = f"{self.BASE_URL}{path}"
        response = self._session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def stream_to_file(self, path: str, dest: Path) -> None:
        """Stream a binary endpoint to *dest* without buffering it in RAM.

        Invariant: *dest*'s parent directory must already exist.
        """
        url = f"{self.BASE_URL}{path}"
        with self._session.get(url, stream=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            written = 0
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    fh.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = written / total * 100
                        print(f"\r  {written / 1_048_576:.1f} / {total / 1_048_576:.1f} MB  ({pct:.0f}%)", end="", flush=True)
        print()  # newline after progress


# ---------------------------------------------------------------------------
# Domain logic
# ---------------------------------------------------------------------------

def resolve_token() -> str:
    """Return a GitHub token from the environment or the ``gh`` CLI.

    Failure modes:
    - Neither source is available → exits with a clear message.
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        log.debug("Using token from GITHUB_TOKEN environment variable.")
        return token

    log.debug("GITHUB_TOKEN not set — trying `gh auth token`.")
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, check=True,
        )
        token = result.stdout.strip()
        if token:
            log.debug("Using token from `gh auth token`.")
            return token
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    log.error("No GitHub token found.")
    log.error("Either set the GITHUB_TOKEN environment variable")
    log.error("or authenticate with the GitHub CLI: gh auth login")
    sys.exit(1)


def find_latest_successful_run(client: GitHubClient, repo: str, workflow: str) -> dict:
    """Return the metadata dict of the most recent successful run.

    Failure mode: no successful run exists → RuntimeError.
    """
    data = client.get(
        f"/repos/{repo}/actions/workflows/{workflow}/runs",
        status="success",
        per_page=1,
    )
    runs = data.get("workflow_runs", [])
    if not runs:
        raise RuntimeError(
            f"No successful runs found for workflow '{workflow}' in repo '{repo}'."
        )
    run = runs[0]
    log.info(
        "Latest successful run — id: %s  created_at: %s  head_branch: %s",
        run["id"], run["created_at"], run["head_branch"],
    )
    return run


def find_artifact(client: GitHubClient, repo: str, run_id: int, prefix: str) -> dict:
    """Return the first artifact whose name starts with *prefix*.

    Failure mode: no matching artifact → RuntimeError listing what *is* available.
    """
    data = client.get(f"/repos/{repo}/actions/runs/{run_id}/artifacts")
    artifacts = data.get("artifacts", [])
    matches = [a for a in artifacts if a["name"].startswith(prefix)]

    if not matches:
        available = [a["name"] for a in artifacts]
        raise RuntimeError(
            f"No artifact with prefix '{prefix}' in run {run_id}. "
            f"Available artifacts: {available}"
        )

    artifact = matches[0]
    size_mb = artifact["size_in_bytes"] / 1_048_576
    log.info("Found artifact — name: %s  size: %.1f MB", artifact["name"], size_mb)
    return artifact


def download_and_extract(
    client: GitHubClient,
    repo: str,
    artifact: dict,
    output_path: Path,
    expected_filename: str,
) -> None:
    """Stream the artifact zip to a temp file, extract *expected_filename*, clean up.

    Why stream to a temp file instead of buffering:
        The zip can be ~300 MB. Buffering it in RAM alongside the extracted
        content would peak at ~600 MB. Streaming keeps memory usage at O(chunk).

    Failure modes:
    - Expected filename not found in zip → RuntimeError listing zip contents.
    - Output directory cannot be created → OSError from mkdir.
    """
    artifact_id = artifact["id"]
    artifact_path = f"/repos/{repo}/actions/artifacts/{artifact_id}/zip"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        log.info("Streaming artifact zip to temp file %s …", tmp_path)
        client.stream_to_file(artifact_path, tmp_path)

        log.info("Extracting '%s' from zip …", expected_filename)
        with zipfile.ZipFile(tmp_path) as zf:
            names = zf.namelist()
            if expected_filename not in names:
                raise RuntimeError(
                    f"Expected '{expected_filename}' inside artifact zip. "
                    f"Zip contents: {names}"
                )
            with zf.open(expected_filename) as src, open(output_path, "wb") as dst:
                dst.write(src.read())
    finally:
        tmp_path.unlink(missing_ok=True)

    size_mb = output_path.stat().st_size / 1_048_576
    log.info("Saved → %s  (%.1f MB)", output_path, size_mb)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    token    = resolve_token()
    client   = GitHubClient(token)

    run      = find_latest_successful_run(client, REPO, WORKFLOW_FILE)
    artifact = find_artifact(client, REPO, run["id"], ARTIFACT_PREFIX)

    download_and_extract(
        client,
        repo=REPO,
        artifact=artifact,
        output_path=OUTPUT_PATH,
        expected_filename=PARQUET_FILENAME,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
