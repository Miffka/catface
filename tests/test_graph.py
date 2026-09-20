from pathlib import Path

import numpy as np

from catface.core.graph import (
    build_cat_adjacency,
    build_cat_edges,
    edges_to_adjacency,
    normalize_adjacency,
)

ROOT = Path(__file__).resolve().parents[1]


def test_edges_to_adjacency_symmetric_no_self_loops():
    A = edges_to_adjacency([(0, 1), (1, 2)], num_nodes=4)
    assert A.shape == (4, 4)
    assert np.array_equal(A, A.T)
    assert np.all(np.diag(A) == 0)
    assert A[0, 1] == 1 and A[1, 2] == 1 and A[0, 2] == 0


def test_normalize_adjacency_known_triangle():
    # Triangle (all-pairs connected, 3 nodes): A+I is all-ones, degree 3 for
    # every node, so D^-1/2 (A+I) D^-1/2 = (1/3) * ones(3,3).
    A = edges_to_adjacency([(0, 1), (1, 2), (0, 2)], num_nodes=3)
    normalized = normalize_adjacency(A)
    assert np.allclose(normalized, np.full((3, 3), 1 / 3))


def test_build_cat_edges_parses_legend_headers_dupes_and_self_loops(tmp_path):
    path = tmp_path / "edges.txt"
    path.write_text(
        "\n".join(
            [
                "#   0: nose, top, left",
                "# -- some_group --",
                "(0,1)",
                "(1,0)",  # reversed-order duplicate of (0,1)
                "(0,1)",  # exact duplicate
                "(2,2)",  # self-loop, dropped
                "(3,4) # trailing comment",
            ]
        )
    )
    edges = build_cat_edges(path)
    assert set(edges) == {(0, 1), (3, 4)}


def test_build_cat_edges_default_path_covers_all_48_nodes():
    edges = build_cat_edges()
    touched = {i for edge in edges for i in edge}
    assert touched == set(range(48))
    assert len(edges) == 76  # 78 raw (i,j) lines in models/graph_edges_manual.txt, 76 unique


def test_build_cat_adjacency_shape_and_no_torch_geometric():
    edges, A_hat = build_cat_adjacency()
    assert A_hat.shape == (48, 48)
    assert np.allclose(A_hat, A_hat.T)
    assert len(edges) > 0

    import ast

    tree = ast.parse((ROOT / "src" / "catface" / "core" / "graph.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            assert not any(n and "torch_geometric" in n for n in names)
