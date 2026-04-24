"""
agents/_subagents.py — Worker registry.

WORKER_NAMES is the single source of truth for which strategies run.
Adding a new strategy = add one entry here; no wiring changes needed in agent.py.
"""

from __future__ import annotations

from agents.graph_nodes import create_subgraph

WORKER_NAMES: list[str] = ["breakout", "ma"]


def build_subgraphs() -> dict:
    """Compile one subgraph per worker name."""
    return {name: create_subgraph(name) for name in WORKER_NAMES}
