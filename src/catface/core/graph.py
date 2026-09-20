"""Adjacency construction for the 48-node cat-face graph (E3, RSCH-3).

The anatomical adjacency is hand-authored and hand-reviewed, not
geometrically derived: `build_cat_edges` parses
`models/graph_edge_schemes/graph_edges_manual_v3.txt`,
an edge list a human wrote after reviewing rendered overlays of an earlier
geometric version (`scripts/graph_review_manual.py`). This module therefore
does I/O (it reads that file) -- see docs/DECISIONS.md's 2026-09-20 entry for
the full rationale and what changed.

Also home to `random_edges`/`build_random_adjacency`, the random-adjacency
ablation generator RSCH-3 requires as a control.
"""

import re
from pathlib import Path

import numpy as np

Edge = tuple[int, int]

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EDGES_PATH = ROOT / "models" / "graph_edge_schemes" / "graph_edges_manual_v3.txt"

_EDGE_LINE = re.compile(r"^\((\d+)\s*,\s*(\d+)\)")


def _dedup(edges: list[Edge]) -> list[Edge]:
    seen: set[Edge] = set()
    out = []
    for a, b in edges:
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def build_cat_edges(path: Path = DEFAULT_EDGES_PATH) -> list[Edge]:
    """Parse the hand-authored edge list at `path`. Legend lines
    (`#   12: nose, top, left`) and group-header comments (`# -- nose_ring --`)
    don't match the leading `(i, j)` pattern and are ignored; a trailing
    `# comment` after a matched edge is fine. Deduplicates and drops
    self-loops."""
    edges: list[Edge] = []
    for line in path.read_text().splitlines():
        if m := _EDGE_LINE.match(line):
            edges.append((int(m.group(1)), int(m.group(2))))
    return _dedup(edges)


def edges_to_adjacency(edges: list[Edge], num_nodes: int = 48) -> np.ndarray:
    """Symmetric 0/1 adjacency, no self-loops."""
    A = np.zeros((num_nodes, num_nodes))
    for a, b in edges:
        A[a, b] = A[b, a] = 1.0
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """D^-1/2 (A+I) D^-1/2, per docs/backlog.md RSCH-3."""
    A_hat = A + np.eye(len(A))
    d_inv_sqrt = 1.0 / np.sqrt(A_hat.sum(axis=1))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def build_cat_adjacency(path: Path = DEFAULT_EDGES_PATH) -> tuple[list[Edge], np.ndarray]:
    edges = build_cat_edges(path)
    return edges, normalize_adjacency(edges_to_adjacency(edges))


def random_edges(num_nodes: int, num_edges: int, seed: int) -> list[Edge]:
    """`num_edges` distinct unordered pairs drawn without replacement from
    all C(num_nodes, 2) possible pairs -- no self-loops, no duplicates by
    construction. Deterministic given the same seed."""
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    chosen = rng.choice(len(all_pairs), size=num_edges, replace=False)
    return [all_pairs[k] for k in chosen]


def build_random_adjacency(num_nodes: int, num_edges: int, seed: int) -> tuple[list[Edge], np.ndarray]:
    edges = random_edges(num_nodes, num_edges, seed)
    return edges, normalize_adjacency(edges_to_adjacency(edges, num_nodes))
