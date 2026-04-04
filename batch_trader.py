"""
batch_trader.py — Batch AI analysis for all Italian equity tickers.

Reads all symbols from analysis_results.parquet, builds per-ticker snapshots,
and submits them to the OpenAI Batch API via kitai (50% cost vs synchronous).

Batch jobs complete asynchronously within 24 h. Job IDs are printed and written
to data/results/it/batches/batch_jobs_YYYYMMDD_HHMMSS.json immediately after
submission. Use that file to recover if the process is interrupted.

Usage:
    python batch_trader.py --mode bo
    python batch_trader.py --mode ma
    python batch_trader.py --mode both
    python batch_trader.py --mode bo --question "Is this a good entry?"
    python batch_trader.py --mode both --poll-interval 60

Output:
    data/results/it/batches/batch_bo_results.json  — per-ticker BO breakout analyses
    data/results/it/batches/batch_ma_results.json  — per-ticker MA crossover analyses

Each output file is a JSON array of objects:
    {"ticker": "A2A.MI", "analysis": {...TraderAnalysis fields...}, "error": null}
    {"ticker": "XY.MI",  "analysis": null,                          "error": "reason"}

Environment:
    OPENAI_API_KEY must be set (loaded from .env via python-dotenv).

Recovery after interruption:
    job_id = "batch_abc123"   # copy from the log above the restart
    from kitai.batch import poll_until_complete, download_batch_results
    completed = poll_until_complete(client, [job_id], poll_interval=30.0)
    results   = download_batch_results(client, job_id)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv

from ask_bo_trader import (
    RESULTS_PATH,
    SYSTEM_PROMPT as BO_SYSTEM_PROMPT,
    MODEL         as BO_MODEL,
    build_snapshot as build_bo_snapshot,
    TraderAnalysis,
)
from ask_ma_trader import (
    SYSTEM_PROMPT  as MA_SYSTEM_PROMPT,
    MODEL          as MA_MODEL,
    build_snapshot as build_ma_snapshot,
    MATraderAnalysis,
)
from kitai.batch import (
    submit_batch_job,
    poll_until_complete,
    download_batch_results,
    check_batch_job,
)

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("data/results/it/batches")
BO_OUTPUT  = OUTPUT_DIR / "batch_bo_results.json"
MA_OUTPUT  = OUTPUT_DIR / "batch_ma_results.json"

BATCH_ENDPOINT = "/v1/chat/completions"

# ---------------------------------------------------------------------------
# Task building
# ---------------------------------------------------------------------------


def _make_strict_schema(node: dict) -> dict:
    """
    Recursively transform a Pydantic model_json_schema() into an OpenAI
    strict-mode compatible schema.

    OpenAI strict mode requirements:
      1. Every object node has "additionalProperties": false.
      2. Every property of an object is listed in "required" (Optional fields
         keep their anyOf: [{type: X}, {type: "null"}] but must still appear
         in required — OpenAI allows null as a valid value).

    Pydantic already emits anyOf for Union[X, None] fields, so we only need
    to add the two object-level keys and recurse into nested schemas.
    """
    node = copy.deepcopy(node)
    _strict_in_place(node)
    return node


def _strict_in_place(node: dict) -> None:
    """Apply strict mode requirements in-place, recursively."""
    if not isinstance(node, dict):
        return

    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False
        if "properties" in node:
            # All properties must appear in required (null-typed fields included).
            node["required"] = list(node["properties"].keys())
            for child in node["properties"].values():
                _strict_in_place(child)

    # Recurse into $defs — Pydantic hoists nested models here.
    for sub in node.get("$defs", {}).values():
        _strict_in_place(sub)

    # Recurse into anyOf / allOf / oneOf (used for Union types).
    for key in ("anyOf", "allOf", "oneOf"):
        for sub in node.get(key, []):
            _strict_in_place(sub)

    # Recurse into array items.
    if "items" in node:
        _strict_in_place(node["items"])


def _response_format(model_class) -> dict:
    """
    Build an OpenAI strict json_schema response_format dict from a Pydantic model.

    strict=True forces the model to emit every declared field, preventing the
    Pydantic validation failures caused by omitted required fields.
    The schema is transformed by _make_strict_schema to satisfy OpenAI's
    strict-mode requirements (additionalProperties: false, all props in required).
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name":   model_class.__name__,
            "schema": _make_strict_schema(model_class.model_json_schema()),
            "strict": True,
        },
    }


def _make_task(
    ticker:          str,
    snapshot:        dict,
    system_prompt:   str,
    model:           str,
    response_format: dict,
    question:        str | None,
) -> dict:
    """
    Build one kitai batch task dict for a single ticker snapshot.

    custom_id = ticker symbol so results map back to tickers without a
    separate index lookup.

    Args:
        ticker:          Yahoo Finance symbol, e.g. "A2A.MI".
        snapshot:        Output of build_snapshot — last-bar data payload.
        system_prompt:   Trader system prompt (BO or MA).
        model:           OpenAI model ID.
        response_format: json_schema dict from _response_format().
        question:        Optional follow-up question appended to user message.

    Returns:
        {"custom_id": ticker, "body": {...chat completions body...}}
    """
    user_content = f"Ticker: {ticker}\n\nSnapshot:\n{json.dumps(snapshot, indent=2)}"
    if question:
        user_content += f"\n\nQuestion: {question}"

    return {
        "custom_id": ticker,
        "method":    "POST",
        "url":       BATCH_ENDPOINT,
        "body": {
            "model":           model,
            "max_tokens":      1024,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            "response_format": response_format,
        },
    }


def _build_tasks(
    df:            pd.DataFrame,
    tickers:       list[str],
    build_fn,
    system_prompt: str,
    model:         str,
    schema_class,
    question:      str | None,
    mode_label:    str,
) -> list[dict]:
    """
    Build batch tasks for every ticker, skipping those where snapshot
    construction fails (e.g. insufficient history, missing columns).

    Args:
        df:            Full analysis_results DataFrame (all tickers).
        tickers:       Ordered list of unique ticker symbols to process.
        build_fn:      build_bo_snapshot or build_ma_snapshot.
        system_prompt: Trader system prompt string.
        model:         OpenAI model ID.
        schema_class:  Pydantic model class (TraderAnalysis or MATraderAnalysis).
        question:      Optional follow-up question.
        mode_label:    "BO" or "MA" — used in log messages only.

    Returns:
        List of kitai task dicts (one per successfully built snapshot).

    Failure modes:
        - Ticker not in df:           skipped with WARNING.
        - build_fn raises ValueError:  skipped with WARNING.
        - Other exception:             skipped with WARNING (stack trace at DEBUG).
    """
    fmt   = _response_format(schema_class)
    tasks = []
    for ticker in tickers:
        df_t = df[df["symbol"] == ticker]
        if df_t.empty:
            log.warning("%s: ticker %s not found in parquet — skipped.", mode_label, ticker)
            continue
        try:
            snapshot = {"ticker": ticker, **build_fn(df_t)}
        except ValueError as exc:
            log.warning("%s: snapshot failed for %s: %s", mode_label, ticker, exc)
            continue
        except Exception as exc:
            log.warning("%s: unexpected error for %s: %s", mode_label, ticker, exc)
            log.debug("", exc_info=True)
            continue
        tasks.append(
            _make_task(ticker, snapshot, system_prompt, model, fmt, question)
        )

    log.info("%s: built %d / %d tasks.", mode_label, len(tasks), len(tickers))
    return tasks


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _parse_results(raw: list[dict], schema_class, mode_label: str) -> list[dict]:
    """
    Parse raw kitai batch results into validated Pydantic model dicts.

    Each item in the returned list has:
        "ticker":   str   — the custom_id (Yahoo Finance symbol)
        "analysis": dict | None — model_dump() of the parsed Pydantic object,
                                  or None when parsing failed
        "error":    str | None  — error message when analysis is None

    Per-item failures are logged and included in the output (error set) so
    the result file always covers all submitted tickers.

    Args:
        raw:         list[dict] from download_batch_results.
        schema_class: Pydantic model class to validate against.
        mode_label:  "BO" or "MA" — used in log messages only.

    Returns:
        List of result dicts, one per raw item.
    """
    out = []
    for item in raw:
        ticker = item.get("custom_id", "unknown")

        # --- Per-item API error (quota, bad input, etc.) ---
        if item.get("error"):
            log.warning("%s: API error for %s: %s", mode_label, ticker, item["error"])
            out.append({"ticker": ticker, "analysis": None, "error": str(item["error"])})
            continue

        # --- Parse content into the Pydantic schema ---
        try:
            content  = item["response"]["body"]["choices"][0]["message"]["content"]
            analysis = schema_class.model_validate_json(content)
            out.append({"ticker": ticker, "analysis": analysis.model_dump(), "error": None})
        except Exception as exc:
            # Strict mode should prevent this, but if validation still fails,
            # preserve the raw JSON so the model's output is not silently lost.
            log.warning("%s: validation failed for %s: %s — storing raw response.", mode_label, ticker, exc)
            try:
                raw_json = json.loads(content)
                out.append({"ticker": ticker, "analysis": raw_json, "error": str(exc)})
            except Exception:
                out.append({"ticker": ticker, "analysis": None, "error": str(exc)})

    success = sum(1 for r in out if r["error"] is None)
    log.info("%s: parsed %d / %d items successfully.", mode_label, success, len(out))
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save(results: list[dict], path: Path) -> None:
    """Write results to a JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)
    log.info("Saved %d records → %s", len(results), path)


def _save_job_ids(job_ids: dict[str, str]) -> Path:
    """
    Persist batch job IDs to a timestamped JSON file in OUTPUT_DIR.

    File name: batch_jobs_YYYYMMDD_HHMMSS.json
    Content:   {"submitted_at": "<iso>", "jobs": {"bo": "batch_...", "ma": "batch_..."}}

    This file is the recovery artefact — paste the IDs into poll_until_complete
    if the main process is interrupted before results are downloaded.

    Returns:
        Path to the written file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now(timezone.utc)
    filename = f"batch_jobs_{ts.strftime('%Y%m%d_%H%M%S')}.json"
    path     = OUTPUT_DIR / filename
    payload  = {"submitted_at": ts.isoformat(), "jobs": job_ids}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Job IDs saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit batch AI analysis for all Italian equity tickers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["bo", "ma", "both"],
        default="both",
        help="Analysis mode: bo (range breakout), ma (MA crossover), both (default).",
    )
    p.add_argument(
        "--question",
        default=None,
        help="Optional follow-up question appended to every ticker snapshot.",
    )
    p.add_argument(
        "--data",
        default=str(RESULTS_PATH),
        help=f"Path to analysis_results.parquet (default: {RESULTS_PATH})",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Seconds between batch status checks while polling (default: 30).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        log.error("Parquet not found: %s", data_path)
        sys.exit(1)

    log.info("Loading %s", data_path)
    df      = pd.read_parquet(data_path)
    tickers = sorted(df["symbol"].unique().tolist())
    log.info("Found %d tickers to process.", len(tickers))

    client  = openai.OpenAI()
    job_ids: dict[str, str] = {}   # mode → batch_id

    # -----------------------------------------------------------------------
    # 1. Build tasks and submit
    # -----------------------------------------------------------------------

    if args.mode in ("bo", "both"):
        bo_tasks = _build_tasks(
            df, tickers,
            build_fn=build_bo_snapshot,
            system_prompt=BO_SYSTEM_PROMPT,
            model=BO_MODEL,
            schema_class=TraderAnalysis,
            question=args.question,
            mode_label="BO",
        )
        if bo_tasks:
            job_ids["bo"] = submit_batch_job(
                client,
                bo_tasks,
                endpoint=BATCH_ENDPOINT,
                metadata={"mode": "bo", "n_tickers": str(len(bo_tasks))},
            )
            log.info("BO batch submitted: %s  (%d tasks)", job_ids["bo"], len(bo_tasks))

    if args.mode in ("ma", "both"):
        ma_tasks = _build_tasks(
            df, tickers,
            build_fn=build_ma_snapshot,
            system_prompt=MA_SYSTEM_PROMPT,
            model=MA_MODEL,
            schema_class=MATraderAnalysis,
            question=args.question,
            mode_label="MA",
        )
        if ma_tasks:
            job_ids["ma"] = submit_batch_job(
                client,
                ma_tasks,
                endpoint=BATCH_ENDPOINT,
                metadata={"mode": "ma", "n_tickers": str(len(ma_tasks))},
            )
            log.info("MA batch submitted: %s  (%d tasks)", job_ids["ma"], len(ma_tasks))

    if not job_ids:
        log.error("No batch jobs submitted — check warnings above.")
        sys.exit(1)

    # Persist job IDs to disk before blocking on poll so they survive a crash.
    ids_file = _save_job_ids(job_ids)
    print()
    print("=" * 50)
    print("  Batch job IDs (also saved to disk):")
    for mode, jid in job_ids.items():
        print(f"  {mode.upper()}: {jid}")
    print(f"  File: {ids_file}")
    print("=" * 50)
    print()

    # -----------------------------------------------------------------------
    # 2. Poll until all jobs reach a terminal state
    # -----------------------------------------------------------------------

    log.info("Polling for completion (interval=%ss)…", args.poll_interval)
    completed_ids = poll_until_complete(
        client,
        list(job_ids.values()),
        poll_interval=args.poll_interval,
    )

    # -----------------------------------------------------------------------
    # 3. Download, parse, and save each completed job
    # -----------------------------------------------------------------------

    for mode, job_id in job_ids.items():
        if job_id not in completed_ids:
            info = check_batch_job(client, job_id)
            log.error(
                "%s batch %s did not complete. Status: %s  Counts: %s",
                mode.upper(), job_id, info["status"], info.get("request_counts"),
            )
            if info.get("error_file_id"):
                log.error(
                    "Error file:\n%s",
                    client.files.content(info["error_file_id"]).text,
                )
            continue

        log.info("Downloading %s results (%s)…", mode.upper(), job_id)
        raw = download_batch_results(client, job_id)

        if mode == "bo":
            parsed = _parse_results(raw, TraderAnalysis,   "BO")
            _save(parsed, BO_OUTPUT)
        else:
            parsed = _parse_results(raw, MATraderAnalysis, "MA")
            _save(parsed, MA_OUTPUT)

    log.info("Done.")


if __name__ == "__main__":
    main()
